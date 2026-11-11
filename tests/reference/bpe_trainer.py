from __future__ import annotations

from enum import Enum
from typing import Union
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse
import time
import json

from utils import MCounter, WHITESPACE, PAD, UNK, BOS, EOS
from language import Token, Word

import logging
logger = logging.getLogger(__name__)


class EventType(Enum):
    MERGE = 0
    SPLIT = 1


class BPE:

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
        coverage: float = 0.9999,
        threshold: float = 0.9999,
    ):
        self.desired_vocab_size = vocab_size
        self.pad_token = Token(pad_id, PAD, 0, special=True)
        self.unk_token = Token(unk_id, UNK, 0, special=True)
        self.bos_token = Token(bos_id, BOS, 0, special=True)
        self.eos_token = Token(eos_id, EOS, 0, special=True)
        self.id2token = {
            token.id: token for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        }
        self.str2token = {
            token.str: token for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        }
        self.str2token = defaultdict(lambda: self.unk_token, self.str2token)
        self.max_special_token_id = max(self.id2token.keys())
        self.actual_vocab_size = len(self.id2token)
        self.new_id = self.max_special_token_id + 1
        self.coverage = coverage
        self.threshold = threshold
        self.events = list()

    @staticmethod
    def _get_words(file: str) -> list[Word]:
        logger.info(f'Loading corpus from {file}...')
        start_time = time.time()
        with open(file) as f:
            counter = MCounter()
            for i, line in enumerate(f):
                counter.update(line.strip('\n').split())
                if i > 0 and i % 500000 == 0:
                    logger.info(f'Processed {i} lines.')
            num_lines = i
        logger.info(
            f'Loaded {len(counter)} unique words from {num_lines} sentences in {time.time() - start_time:.2f}s.'
        )
        return [Word(i, WHITESPACE + word, freq) for i, (word, freq) in enumerate(counter.items())]

    @staticmethod
    def _get_characters(words: list[Word]) -> MCounter:
        counter = MCounter()
        for i, word in enumerate(words):
            counter.update(MCounter(word.str) * word.freq)
            if i > 0 and i % 500000 == 0:
                logger.info(f'Processed {i} words.')
        return counter

    def _filter_characters(self, characters: MCounter) -> MCounter:
        if self.coverage < 1:
            corpus_size = sum(characters.values())
            freq_to_remove = corpus_size - round(self.coverage * corpus_size)
            if freq_to_remove > 0:
                cum_sum = np.cumsum([freq for _, freq in reversed(characters.most_common())])
                num_to_remove = np.searchsorted(cum_sum, freq_to_remove)
                characters_to_remove = [c for c, _ in characters.most_common()[-num_to_remove:]]
                for c in characters_to_remove:
                    characters.pop(c)
                logger.info(f'Replaced {num_to_remove} rare characters with UNK.')
        return characters

    def _initialize_vocab(self, words: list[Word]) -> None:
        logger.info('Initializing the vocabulary...')
        characters = self._get_characters(words)
        filtered_characters = self._filter_characters(characters)
        for i, character in enumerate(filtered_characters):
            token = Token(self.new_id + i, character, filtered_characters[character])
            self.id2token[token.id] = token
            self.str2token[token.str] = token
        self.new_id += len(filtered_characters)
        self.actual_vocab_size += len(filtered_characters)
        logger.info(f'Initialized vocabulary with {len(filtered_characters)} unique characters.')

    @staticmethod
    def _validate_pair(pair: np.ndarray) -> bool:
        return not any(token.special for token in pair)

    def _encode_words(self, words: list[Word]) -> None:
        logger.info('Encoding words...')
        for i, word in enumerate(words):
            word.encode(self.str2token)
            if i > 0 and i % 500000 == 0:
                logger.info(f'Processed {i} words.')

    def _initialize_pairs(self, words: list[Word]) -> MCounter:
        pairs = MCounter()
        logger.info('Counting character pairs...')
        for i, word in enumerate(words):
            pairs.update(word.pairs)
            if i > 0 and i % 500000 == 0:
                logger.info(f'Processed {i} words.')
        to_remove = set()
        for pair in pairs:
            if not self._validate_pair(pair):
                to_remove.add(pair)
        for pair in to_remove:
            pairs.pop(pair)
        return pairs

    @staticmethod
    def _update_pairs_on_merge(
        new_token: Token,
        pair: tuple[Token, Token],
        pairs_for_update: MCounter,
        pairs: MCounter
    ):
        pairs.update(pairs_for_update)
        for p, freq in pairs_for_update.items():
            if new_token not in p:
                raise ValueError(f'Pair {p} does not contain the new token {new_token}.')
            if new_token is p[0]:
                if new_token is p[1]:
                    to_update = (pair[1], pair[0])
                else:
                    to_update = (pair[1], p[1])
            else:
                to_update = (p[0], pair[0])
            if to_update in pairs:
                pairs[to_update] -= freq
                if pairs[to_update] <= 0:
                    pairs.pop(to_update)

    @staticmethod
    def _update_pairs_on_remove(token: Token, split: list[Token], pairs_for_update: MCounter, pairs: MCounter):
        for pair, freq in pairs_for_update.items():
            if token is pair[0]:
                if token is pair[1]:
                    to_update = (split[-1], split[0])
                else:
                    to_update = (split[-1], pair[1])
            else:
                to_update = (pair[0], split[0])
            pairs[to_update] += freq
            pairs.pop(pair)

    def _remove_if_possible(self, token: Token, merged_freq: int, pairs: MCounter) -> bool:
        if merged_freq / (token.freq + merged_freq) > self.threshold:
            split = token.split_if_possible()
            if split is not None:
                self.actual_vocab_size -= 1
                for t in split:
                    t.freq += token.freq
                for pair in zip(split[:-1], split[1:]):
                    pairs[pair] += token.freq
                pairs_for_update = MCounter()
                for word in token.words:
                    if token not in word.tokens:
                        raise ValueError(f'Token {token} not found in the token list {word.tokens} of word {word}.')
                    pairs_for_update.update({pair: freq for pair, freq in word.pairs.items() if
                                            self._validate_pair(pair) and token in pair})
                    word.split_token(token, split)
                self._update_pairs_on_remove(token, split, pairs_for_update, pairs)
                token.remove()
                return True
        return False

    def _merge_token_in_words(
            self,
            token_to_merge: Token,
            pair_to_merge: tuple[Token, Token],
            pairs: MCounter,
    ) -> int:
        actual_freq = 0
        pairs_for_update = MCounter()
        for word in pair_to_merge[0].words & pair_to_merge[1].words:
            if pair_to_merge in word.pairs:
                word.pairs.pop(pair_to_merge)
                actual_freq += word.merge_pair(pair_to_merge, token_to_merge)
                pairs_for_update.update(
                    {p: f for p, f in word.pairs.items() if self._validate_pair(p) and token_to_merge in p}
                )
        self._update_pairs_on_merge(token_to_merge, pair_to_merge, pairs_for_update, pairs)
        token_to_merge.freq += actual_freq
        if pair_to_merge[0] is pair_to_merge[1]:
            pair_to_merge[0].freq -= 2 * actual_freq
            removed = self._remove_if_possible(pair_to_merge[0], actual_freq, pairs)
            if removed:
                logger.info(
                    f'Removed token {pair_to_merge[0].str} with frequency {pair_to_merge[0].freq} '
                    f'after merging into {token_to_merge.str} with frequency {token_to_merge.freq}.'
                )
                self.events.append((EventType.SPLIT, pair_to_merge[0], pair_to_merge[0].walk()))
        else:
            for token in pair_to_merge:
                if not token.present:
                    raise ValueError(f'Token {token} is not present in the vocabulary.')
                token.freq -= actual_freq
                token_freq = token.freq
                removed = self._remove_if_possible(token, actual_freq, pairs)
                if removed:
                    logger.info(
                        f'Removed token {token.str} with frequency {token_freq} '
                        f'after merging into {token_to_merge.str} with frequency {token_to_merge.freq}.'
                    )
                    self.events.append((EventType.SPLIT, token, token.walk()))
        return actual_freq

    def _merge_pair(self, pair: tuple[Token, Token], pairs: MCounter) -> int:
        pairs.pop(pair)
        merged_str = pair[0].str + pair[1].str
        if merged_str in self.str2token:
            new_token = self.str2token[merged_str]
            if not new_token.present:
                new_token.restore()
                logger.info(f'Restored previously removed token {new_token.str}.')
            else:
                logger.info(f'Additional merges for {new_token.str}.')
        else:
            new_token = Token(self.new_id, merged_str, 0, left=pair[0], right=pair[1])
            self.id2token[new_token.id] = new_token
            self.str2token[new_token.str] = new_token
            self.new_id += 1
        self.events.append((EventType.MERGE, pair, new_token))
        actual_freq = self._merge_token_in_words(new_token, pair, pairs)
        return actual_freq

    def _dump(self, file: Union[Path, str]) -> None:
        logger.info(f'Dumping model to {file}...')
        assigned_ids = sorted(self.id2token.keys())
        id_mapping = dict()
        id_counter = 0
        for i in assigned_ids:
            if self.id2token[i].present:
                id_mapping[i] = id_counter
                id_counter += 1
        with open(file, 'w') as f:
            json.dump({
                'tokens': [token.to_dict() for token in self.id2token.values()],
                'id2int': id_mapping,
                'int2id': {v: k for k, v in id_mapping.items()},
                'merges': [{'id': i, 'pair': [token.to_dict() for token in merge[1]],
                            'new_token': merge[2].to_dict()}
                           for i, merge in enumerate(self.events) if merge[0] == EventType.MERGE],
                'splits': [{'id': i, 'token': merge[1].to_dict(),
                            'split': [token.to_dict() for token in merge[2]]}
                           for i, merge in enumerate(self.events) if merge[0] == EventType.SPLIT],
            }, f, indent=4)

    def fit(self, input_file: Union[Path, str], model_file: Union[Path, str], logging_step: int = 200) -> None:
        words = self._get_words(input_file)
        self._initialize_vocab(words)
        self._encode_words(words)
        pairs = self._initialize_pairs(words)
        merge_time = []
        while self.actual_vocab_size < self.desired_vocab_size:
            start_time = time.time()
            pair, count = pairs.most_common(1)[0]
            if count <= 0:
                logger.info(f'No more pairs to merge. Stopping with vocab size of {self.actual_vocab_size}.')
                break
            freq = self._merge_pair(pair, pairs)
            self.actual_vocab_size += 1
            merge_time.append(time.time() - start_time)
            if self.actual_vocab_size % logging_step == 0:
                logger.info(
                    f'VOCABULARY SIZE: {self.actual_vocab_size}. '
                    f'Merged {pair[0].str} + {pair[1].str} with frequency {freq}. '
                    f'Average merge time {np.mean(merge_time):.2f}s.'
                )
                merge_time = []
        self._dump(model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to the input file.')
    parser.add_argument('--model_file', type=str, help='Path to the output model file.')
    parser.add_argument('--vocab_size', type=int, help='Desired vocabulary size.')
    parser.add_argument('--threshold', type=float, help='Desired threshold.')
    parser.add_argument('--coverage', type=float, default=0.9999, help='Desired coverage.')
    parser.add_argument('--pad_id', type=int, default=0, help='ID of the padding token.')
    parser.add_argument('--unk_id', type=int, default=1, help='ID of the unknown token.')
    parser.add_argument('--bos_id', type=int, default=2, help='ID of the beginning-of-sequence token.')
    parser.add_argument('--eos_id', type=int, default=3, help='ID of the end-of-sequence token.')
    parser.add_argument('--logging_step', type=int, default=200, help='Logging step.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    bpe = BPE(
        args.vocab_size,
        pad_id=args.pad_id,
        unk_id=args.unk_id,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        coverage=args.coverage,
        threshold=args.threshold
    )
    bpe.fit(args.input_file, args.model_file, args.logging_step)

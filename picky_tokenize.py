import json
import time
import argparse
import numpy as np
from utils import WHITESPACE, UNK
from language import Token, Word
from collections import defaultdict
from functools import lru_cache


import logging


class BPEModel:

    def __init__(self, bpe_model_path):
        self.id2token = dict()
        self.str2token = dict()
        self.id2int = dict()
        self.int2id = dict()
        self.merge_map = defaultdict(list)
        self.split_map = defaultdict(list)
        self.splits = dict()
        self.events = []
        self._load_bpe_model(bpe_model_path)

    def _token_from_dict(self, token_dict: dict) -> Token:
        return Token(
            token_dict['id'],
            token_dict['str'],
            token_dict['freq'],
            token_dict['special'],
            token_dict['present'],
            self.id2token[token_dict['left']] if token_dict['left'] is not None else None,
            self.id2token[token_dict['right']] if token_dict['right'] is not None else None,
            [self.id2token[i] for i in token_dict['split']] if len(token_dict['split']) > 1 else None
        )

    def _load_bpe_model(self, bpe_model_path: str) -> None:
        with open(bpe_model_path, 'r') as f:
            bpe_model = json.load(f)
        for token_dict in sorted(bpe_model['tokens'], key=lambda x: x['id']):
            token = self._token_from_dict(token_dict)
            self.id2token[token.id] = token
            self.str2token[token.str] = token
        self.str2token = defaultdict(lambda: self.str2token[UNK], self.str2token)
        self.events = bpe_model['merges'] + bpe_model['splits']
        self.events.sort(key=lambda x: x['id'])
        for merge in bpe_model['merges']:
            self.merge_map[(self.str2token[merge['pair'][0]['str']], self.str2token[merge['pair'][1]['str']])].append(merge['id'])
        for merge in self.merge_map:
            self.merge_map[merge] = np.array(self.merge_map[merge])
        for split in bpe_model['splits']:
            self.split_map[self.str2token[split['token']['str']]].append(split['id'])
            self.splits[split['id']] = [self.str2token[token['str']] for token in split['split']]
        for split in self.split_map:
            self.split_map[split] = np.array(self.split_map[split])
        self.id2int = bpe_model['id2int']
        self.int2id = bpe_model['int2id']

    @lru_cache(maxsize=None)
    def _encode_word_by_event_sequence(self, word: str) -> list[Token]:
        processed_word = word
        if processed_word in self.str2token and self.str2token[processed_word].present:
            return [self.str2token[processed_word]]
        word = Word(0, processed_word)
        word.encode(self.str2token)
        for event in self.events:
            pairs = word.pairs
            if 'pair' in event:
                pair = (self.str2token[event['pair'][0]['str']], self.str2token[event['pair'][1]['str']])
                if pair in pairs:
                    word.merge_pair(pair, self.str2token[event['new_token']['str']])
            else:
                token = self.str2token[event['token']['str']]
                if token in word.tokens:
                    word.split_token(token, [self.str2token[t['str']] for t in event['split']])
        return word.tokens

    @lru_cache(maxsize=None)
    def _encode_word_by_events(self, word: str) -> list[Token]:
        proceesed_word = word
        if proceesed_word in self.str2token and self.str2token[proceesed_word].present:
            return [self.str2token[proceesed_word]]
        previous_event = -1
        word = Word(0, proceesed_word)
        word.encode(self.str2token)
        while True:
            pairs = [pair for pair in word.pairs if pair in self.merge_map]
            pairs = [(pair, self.merge_map[pair][np.searchsorted(self.merge_map[pair], previous_event)]) for pair in pairs
                     if np.any(self.merge_map[pair] >= previous_event)]
            removals = [token for token in word.tokens if token in self.split_map]
            removals = [(token, self.split_map[token][np.searchsorted(self.split_map[token], previous_event)])
                        for token in removals if np.any(self.split_map[token] >= previous_event)]
            if not pairs and not removals:
                break
            pair_to_merge, token_to_remove = None, None
            merge_event_id, split_event_id = None, None
            if pairs:
                pair_to_merge, merge_event_id = min(pairs, key=lambda p: p[1])
            if removals:
                token_to_remove, split_event_id = min(removals, key=lambda t: t[1])
            if merge_event_id is None and split_event_id is None:
                break
            if token_to_remove is None or (pair_to_merge is not None and merge_event_id < split_event_id):
                word.merge_pair(pair_to_merge, self.str2token[pair_to_merge[0].str + pair_to_merge[1].str], update_tokens=False)
                previous_event = merge_event_id
            else:
                word.split_token(token_to_remove, self.splits[split_event_id], update_tokens=False)
                previous_event = split_event_id
        return word.tokens

    def encode_file(
        self,
        input_file: str,
        output_file: str,
        return_type: str = 'str',
    ) -> None:
        start_time = time.time()
        result = []
        with open(input_file, 'r') as file:
            logger.info('Encoding text...')
            for i, line in enumerate(file):
                words = line.strip().split()
                tokens = [token for word in words for token in self._encode_word_by_events(WHITESPACE + word)]
                if return_type == 'str':
                    result.append(' '.join([token.str for token in tokens]))
                elif return_type == 'int':
                    result.append(' '.join([str(self.id2int[str(token.id)]) for token in tokens]))
                else:
                    raise NotImplementedError(f'Unknown return type: {return_type}. Available options: str, int.')
                if i > 0 and i % 100000 == 0:
                    logger.info(f'Encoded {i} lines. Elapsed time: {time.time() - start_time:.2f} seconds.')
        logger.info(f'Encoded text in {time.time() - start_time:.2f} seconds.')
        start_time = time.time()
        with open(output_file, 'w') as file:
            logger.info('Writing encoded text...')
            file.write('\n'.join(result))
        logger.info(f'Wrote encoded text in {time.time() - start_time:.2f} seconds.')

    def decode(self, text: str, input_type: str = 'str') -> str:
        sentences = [sentence.strip().split(' ') for sentence in text.strip().split('\n')]
        if input_type == 'int':
            sentences = [[self.id2token[self.int2id[token]].str for token in sentence] for sentence in sentences]
        elif input_type != 'str':
            raise NotImplementedError(f'Unknown input type: {input_type}. Available options: str, int.')
        return '\n'.join([''.join(sentence).replace(WHITESPACE, ' ').strip() for sentence in sentences])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe_model', type=str, required=True, help='Path to the BPE model.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--return_type', type=str, default='str', help='Return type: str or int.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    model = BPEModel(args.bpe_model)
    model.encode_file(args.input_file, args.output_file, args.return_type)

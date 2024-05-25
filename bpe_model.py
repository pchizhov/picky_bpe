import json
import time
from utils import WHITESPACE, UNK
from language import Token, Word
from collections import defaultdict


import logging
logger = logging.getLogger(__name__)


class BPEModel:

    def __init__(self, bpe_model_path):
        self.id2token = dict()
        self.str2token = dict()
        self.id2int = dict()
        self.int2id = dict()
        self.merges = dict()
        self._load_bpe_model(bpe_model_path)

    def _load_bpe_model(self, bpe_model_path: str) -> None:
        with open(bpe_model_path, 'r') as f:
            bpe_model = json.load(f)
        for token_dict in sorted(bpe_model['tokens'], key=lambda x: x['id']):
            token = Token(
                token_dict['id'],
                token_dict['str'],
                token_dict['freq'],
                token_dict['special'],
                token_dict['present'],
                self.id2token[token_dict['left']] if token_dict['left'] is not None else None,
                self.id2token[token_dict['right']] if token_dict['right'] is not None else None,
                [self.id2token[i] for i in token_dict['split']] if len(token_dict['split']) > 1 else None
            )
            self.id2token[token.id] = token
            self.str2token[token.str] = token
            if not token.atomic:
                self.merges[(token.left, token.right)] = token.id
            self.str2token = defaultdict(lambda: self.str2token[UNK], self.str2token)
        self.id2int = bpe_model['id2int']
        self.int2id = bpe_model['int2id']

    def _encode_word(self, word: str) -> list[Token]:
        processed_word = word
        if processed_word in self.str2token:
            tokens = [self.str2token[processed_word]]
        else:
            word = Word(0, processed_word)
            word.encode(self.str2token)
            while True:
                if len(word.tokens) == 1:
                    break
                token_pairs = [pair for pair in word.pairs if pair in self.merges]
                if not token_pairs:
                    break
                pair_to_merge = min(token_pairs, key=lambda p: self.merges[p])
                word.merge_pair(pair_to_merge, self.id2token[self.merges[pair_to_merge]])
            tokens = word.tokens
        result = []
        for token in tokens:
            if not token.present:
                result.extend(token.split)
            else:
                result.append(token)
        return result

    def _encode(self, text: str) -> list[list[Token]]:
        sentences = [sentence.strip().split(' ') for sentence in text.strip().split('\n')]
        words = set([WHITESPACE + word for sentence in sentences for word in sentence])
        encoded_words = {word: self._encode_word(word) for word in words}
        return [sum([encoded_words[WHITESPACE + word] for word in sentence], start=[]) for sentence in sentences]

    def encode(self, text: str, return_type: str = 'str') -> str:
        encoded = self._encode(text)
        if return_type == 'str':
            return '\n'.join([' '.join([token.str for token in sentence]) for sentence in encoded])
        elif return_type == 'int':
            return '\n'.join([' '.join([str(self.id2int[str(token.id)]) for token in sentence]) for sentence in encoded])
        else:
            raise NotImplementedError(f'Unknown return type: {return_type}.')

    def encode_file(self, file_path: str, return_type: str = 'str') -> str:
        start = time.time()
        with open(file_path, 'r') as file:
            logger.info(f'Reading file {file_path}...')
            words = set()
            for i, line in enumerate(file):
                words.update([WHITESPACE + word for word in line.strip().split(' ')])
                if i > 0 and i % 100000 == 0:
                    logger.info(f'Processed {i} lines.')
        logger.info(f'Loaded {len(words)} words in {time.time() - start:.2f} seconds.')
        start_time = time.time()
        logger.info('Encoding words...')
        encoded_words = dict()
        for i, word in enumerate(words):
            encoded_words[word] = self._encode_word(word)
            if i > 0 and i % 100000 == 0:
                logger.info(f'Encoded {i} words.')
        logger.info(f'Encoded {len(words)} words in {time.time() - start_time:.2f} seconds.')
        start_time = time.time()
        result = []
        with open(file_path, 'r') as file:
            logger.info('Encoding text...')
            for i, line in enumerate(file):
                words = line.strip().split(' ')
                tokens = [token for word in words for token in encoded_words[WHITESPACE + word]]
                if return_type == 'str':
                    result.append(' '.join([token.str for token in tokens]))
                elif return_type == 'int':
                    result.append(' '.join([str(self.id2int[str(token.id)]) for token in tokens]))
                else:
                    raise NotImplementedError(f'Unknown return type: {return_type}. Available options: str, int.')
                if i > 0 and i % 100000 == 0:
                    logger.info(f'Encoded {i} lines. Elapsed time: {time.time() - start_time:.2f} seconds.')
        logger.info(f'Encoded text in {time.time() - start_time:.2f} seconds.')
        return '\n'.join(result)

    def decode(self, text: str, input_type: str = 'str') -> str:
        sentences = [sentence.strip().split(' ') for sentence in text.strip().split('\n')]
        if input_type == 'int':
            sentences = [[self.id2token[self.int2id[token]].str for token in sentence] for sentence in sentences]
        elif input_type != 'str':
            raise NotImplementedError(f'Unknown input type: {input_type}. Available options: str, int.')
        return '\n'.join([''.join(sentence).replace(WHITESPACE, ' ').strip() for sentence in sentences])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    bpe_model = BPEModel('model.json')
    encoding_type = 'str'
    encoded = bpe_model.encode('housekeeping', return_type=encoding_type)
    decoded = bpe_model.decode(encoded, input_type=encoding_type)
    print(encoded)
    print(decoded)
    print(bpe_model.encode_file('botchan.txt', return_type=encoding_type))

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from typing import Iterable, Optional, Union

from tokenizers import decoders
from transformers import PreTrainedTokenizer

from ._encoder import HEADER, PickyEncoder
from ._pretokenize import count_pretokens, make_pre_tokenizer, pre_tokenize
from ._trainer import PickyTrainer, TrainResult

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'picky_merges.txt'}


class PickyBPETokenizer(PreTrainedTokenizer):
    """Byte-level Picky BPE tokenizer with a Hugging Face ``PreTrainedTokenizer`` interface.

    Train with :meth:`train_new_from_iterator` / :meth:`train_from_files`, persist with
    ``save_pretrained`` and reload with ``from_pretrained``.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        pattern: Optional[str] = None,
        add_prefix_space: bool = False,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        **kwargs,
    ):
        with open(vocab_file, encoding='utf-8') as f:
            self.encoder: dict[str, int] = json.load(f)
        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        with open(merges_file, encoding='utf-8') as f:
            self._event_lines = f.read().split('\n')
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self._pretokenizer = make_pre_tokenizer(pattern)
        self._byte_decoder = decoders.ByteLevel()
        self._encoder = PickyEncoder(self._event_lines, self.encoder)
        # special tokens (pad/unk/bos/eos/...) are regular HF added tokens, handled by the base class
        super().__init__(
            pattern=pattern, add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token, add_eos_token=add_eos_token, **kwargs,
        )

    # ------------------------------------------------------------------ vocab
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> dict[str, int]:
        return {**self.encoder, **self.added_tokens_encoder}

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    # ------------------------------------------------------------------ encode / decode
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if self.add_prefix_space and not text[:1].isspace():
            text = ' ' + text
        return text, kwargs

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        tokens: list[str] = []
        decoder = self.decoder
        for pretoken in pre_tokenize(self._pretokenizer, text):
            tokens.extend(decoder[i] for i in self._encoder.encode_pretoken(pretoken))
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._byte_decoder.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        out = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            out = out + bos + token_ids_1 + eos
        return out

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        out = bos + [0] * len(token_ids_0) + eos
        if token_ids_1 is not None:
            out += bos + [0] * len(token_ids_1) + eos
        return out

    # ------------------------------------------------------------------ persistence
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str, str]:
        prefix = f'{filename_prefix}-' if filename_prefix else ''
        vocab_path = os.path.join(save_directory, prefix + VOCAB_FILES_NAMES['vocab_file'])
        merges_path = os.path.join(save_directory, prefix + VOCAB_FILES_NAMES['merges_file'])
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=1)
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self._event_lines).rstrip('\n') + '\n')
        return vocab_path, merges_path

    # ------------------------------------------------------------------ training
    @classmethod
    def train_new_from_iterator(
        cls,
        text_iterator: Iterable[str],
        vocab_size: int,
        threshold: float = 0.9999,
        pattern: Optional[str] = None,
        min_frequency: int = 1,
        queue: str = 'lazy',
        logging_step: int = 200,
        check: bool = False,
        **tokenizer_kwargs,
    ) -> 'PickyBPETokenizer':
        """Train a new tokenizer. ``vocab_size`` includes the special tokens and the 256 bytes.
        ``threshold`` is the IoS threshold; 1.0 gives vanilla BPE. Special tokens are passed like to any
        HF tokenizer, e.g. ``eos_token='<|endoftext|>'``, ``pad_token='<pad>'`` (``additional_special_tokens=[...]``)."""
        word_counts = count_pretokens(make_pre_tokenizer(pattern), text_iterator)
        logger.info(f'Counted {len(word_counts)} unique pretokens.')
        n_special = len({t for t in cls._special_token_strings(tokenizer_kwargs)})
        trainer = PickyTrainer(
            vocab_size, threshold, num_special_tokens=n_special, queue=queue,
            min_frequency=min_frequency, check=check,
        )
        result = trainer.train(word_counts, logging_step)
        return cls._from_result(result, pattern, **tokenizer_kwargs)

    @classmethod
    def train_from_files(cls, files: Union[str, os.PathLike, Iterable], vocab_size: int, **kwargs):
        """Like :meth:`train_new_from_iterator` on one or more text files (one text per line)."""
        if isinstance(files, (str, os.PathLike)):
            files = [files]

        def lines():
            for file in files:
                with open(file, encoding='utf-8') as f:
                    for line in f:
                        yield line.rstrip('\n')

        return cls.train_new_from_iterator(lines(), vocab_size, **kwargs)

    @staticmethod
    def _special_token_strings(kwargs: dict) -> list[str]:
        out = []
        for key, value in kwargs.items():
            if key == 'additional_special_tokens':
                out.extend(str(t) for t in value)
            elif key.endswith('_token') and value is not None:
                out.append(str(value))
        return out

    @classmethod
    def _from_result(cls, result: TrainResult, pattern, **tokenizer_kwargs):
        vocab: dict[str, int] = {}
        for tok, present in zip(result.token_str, result.present):
            if present:
                vocab[tok] = len(vocab)
        s = result.token_str.__getitem__
        lines = [HEADER]
        for ev in result.events:
            if ev[0] == 'M':
                lines.append(f'M {s(ev[1])} {s(ev[2])}')
            else:
                lines.append(f'S {s(ev[1])} {" ".join(s(p) for p in ev[2])}')
        with tempfile.TemporaryDirectory() as tmp:
            vocab_file = os.path.join(tmp, VOCAB_FILES_NAMES['vocab_file'])
            merges_file = os.path.join(tmp, VOCAB_FILES_NAMES['merges_file'])
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False)
            with open(merges_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            return cls(vocab_file, merges_file, pattern=pattern, **tokenizer_kwargs)

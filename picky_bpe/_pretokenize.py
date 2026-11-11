"""Pre-tokenization is delegated to Hugging Face ``tokenizers`` (ByteLevel: GPT-2 regex +
byte->unicode mapping). A custom regex is supported via ``Split`` + ByteLevel."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from tokenizers import Regex
from tokenizers.pre_tokenizers import ByteLevel, PreTokenizer, Sequence, Split


def make_pre_tokenizer(pattern: Optional[str] = None) -> PreTokenizer:
    if pattern is None:
        return ByteLevel(add_prefix_space=False, use_regex=True)
    return Sequence([Split(Regex(pattern), 'isolated'), ByteLevel(add_prefix_space=False, use_regex=False)])


def pre_tokenize(pre_tokenizer: PreTokenizer, text: str) -> list[str]:
    return [piece for piece, _ in pre_tokenizer.pre_tokenize_str(text)]


def count_pretokens(pre_tokenizer: PreTokenizer, texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        counter.update(pre_tokenize(pre_tokenizer, text))
    return counter

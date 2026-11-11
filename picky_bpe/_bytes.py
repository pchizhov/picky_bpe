"""The byte alphabet, taken from Hugging Face's ByteLevel pre-tokenizer.

Internal token ids 0..255 are the indices into the *sorted* alphabet (the order of
``ByteLevel.alphabet()`` itself is not stable).
"""
from tokenizers.pre_tokenizers import ByteLevel

ALPHABET: list[str] = sorted(ByteLevel.alphabet())
assert len(ALPHABET) == 256
CHAR2ID: dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}

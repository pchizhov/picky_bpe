from __future__ import annotations
from typing import Optional
from utils import MCounter


class Token:

    def __init__(
        self,
        id: int,
        str: str,
        freq: int = 0,
        special: bool = False,
        present: bool = True,
        left: Optional[Token] = None,
        right: Optional[Token] = None,
        split: Optional[list[Token]] = None
    ):
        self.id = id
        self.str = str
        self.freq = freq
        self.special = special
        self.present = present
        self.atomic = len(str) == 1 or special
        self.words = set()
        self.left = left
        self.right = right
        self.split = split

    def __repr__(self):
        return f'{self.str} ({self.freq})'

    def walk(self) -> list[Token]:
        if self.atomic or self.present:
            return [self]
        return self.left.walk() + self.right.walk()

    def remove(self) -> None:
        if self.atomic:
            raise ValueError(f'Cannot remove an atomic token {self.str}.')
        self.present = False
        self.freq = 0
        self.words = set()

    def restore(self) -> None:
        if self.present:
            raise ValueError(f'Cannot revoke already present token {self.str}.')
        self.present = True

    def split_if_possible(self) -> Optional[list[Token]]:
        if self.atomic:
            return None
        self.present = False
        return self.walk()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'str': self.str,
            'freq': self.freq,
            'special': self.special,
            'present': self.present,
            'left': self.left.id if self.left is not None else None,
            'right': self.right.id if self.right is not None else None,
            'split': [t.id for t in self.walk()]
        }


class Word:

    def __init__(self, id: int, word: str, freq: int = 0):
        self.id = id
        self.str = word
        self.freq = freq
        self.tokens = None
        self.pairs = None

    def __repr__(self) -> str:
        return f'{self.str} ({self.freq})'

    def encode(self, str2token: dict[str, Token]) -> None:
        self.tokens = [str2token[c] for c in self.str]
        self._recalculate()

    def _recalculate(self, update_tokens: bool = True) -> None:
        self.pairs = MCounter(zip(self.tokens[:-1], self.tokens[1:])) * self.freq
        if update_tokens:
            for token in self.tokens:
                token.words.add(self)

    def merge_pair(self, pair: tuple[Token], new_token: Token, update_tokens: bool = True) -> int:
        new_tokens = []
        i = 0
        while i < len(self.tokens):
            if i < len(self.tokens) - 1 and (self.tokens[i], self.tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1
        new_token_frequency = len(self.tokens) - len(new_tokens)
        if update_tokens:
            pair[0].words.discard(self)
            pair[1].words.discard(self)
        self.tokens = new_tokens
        self._recalculate(update_tokens=update_tokens)
        return new_token_frequency * self.freq

    def split_token(self, token: Token, split: list[Token]):
        new_tokens = []
        for t in self.tokens:
            if t == token:
                new_tokens.extend(split)
            else:
                new_tokens.append(t)
        for token in new_tokens:
            token.words.add(self)
        self.tokens = new_tokens
        self._recalculate()

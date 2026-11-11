"""Event-replay encoder for Picky BPE (internal).

A model is an ordered log of events:
    M <left> <right>          merge two tokens (new token = left + right)
    S <token> <p1> <p2> ...   remove ``token``, replacing it by its parts
Encoding a pretoken is equivalent to replaying the whole log on its bytes, which is
done here by repeatedly jumping to the next applicable event.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import Iterable, Optional

from ._bytes import ALPHABET, CHAR2ID

HEADER = '#version: picky-bpe 1'


class PickyEncoder:

    def __init__(self, event_lines: Iterable[str], vocab: dict[str, int], cache_size: int = 500_000):
        self.vocab = vocab
        s2i: dict[str, int] = dict(CHAR2ID)
        i2s: list[str] = list(ALPHABET)
        self.merge_ranks: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.split_ranks: dict[int, list[int]] = defaultdict(list)
        self.ev_new: list[int] = []  # merge: new token; split: the removed token
        self.ev_pair: list[Optional[tuple[int, int]]] = []  # merge: (left, right); split: None
        self.ev_parts: list[Optional[list[int]]] = []  # split: replacement; merge: None
        k = 0
        for line in event_lines:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            kind, *toks = line.split(' ')
            if kind == 'M':
                left, right = toks
                merged = left + right
                if merged not in s2i:
                    s2i[merged] = len(i2s)
                    i2s.append(merged)
                pair = (s2i[left], s2i[right])
                self.merge_ranks[pair].append(k)
                self.ev_new.append(s2i[merged])
                self.ev_pair.append(pair)
                self.ev_parts.append(None)
            elif kind == 'S':
                self.split_ranks[s2i[toks[0]]].append(k)
                self.ev_new.append(s2i[toks[0]])
                self.ev_pair.append(None)
                self.ev_parts.append([s2i[p] for p in toks[1:]])
            else:
                raise ValueError(f'Unknown event: {line!r}')
            k += 1
        self.num_events = k
        # internal id -> final vocabulary id (-1: not in the final vocabulary)
        self.final_id: list[int] = [vocab.get(s, -1) for s in i2s]
        self.internal_str: list[str] = i2s
        self._cache: dict[str, tuple[int, ...]] = {}
        self._cache_size = cache_size

    def encode_internal(self, word: str) -> list[int]:
        w = [CHAR2ID[c] for c in word]
        merge_ranks, split_ranks = self.merge_ranks, self.split_ranks
        prev = 0
        while True:
            best = None
            if len(w) > 1:
                for pr in set(zip(w, w[1:])):
                    ranks = merge_ranks.get(pr)
                    if ranks:
                        j = bisect_left(ranks, prev)
                        if j < len(ranks) and (best is None or ranks[j] < best):
                            best = ranks[j]
            for t in set(w):
                ranks = split_ranks.get(t)
                if ranks:
                    j = bisect_left(ranks, prev)
                    if j < len(ranks) and (best is None or ranks[j] < best):
                        best = ranks[j]
            if best is None:
                return w
            parts = self.ev_parts[best]
            out = []
            if parts is None:  # merge
                new = self.ev_new[best]
                a, b = self.ev_pair[best]
                i, n = 0, len(w)
                while i < n:
                    if i + 1 < n and w[i] == a and w[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
            else:  # split
                t = self.ev_new[best]
                for x in w:
                    if x == t:
                        out.extend(parts)
                    else:
                        out.append(x)
            w = out
            prev = best + 1

    def encode_pretoken(self, word: str) -> tuple[int, ...]:
        """Encode a pretoken (byte-level alphabet string) into final vocabulary ids."""
        res = self._cache.get(word)
        if res is None:
            res = tuple(self.final_id[x] for x in self.encode_internal(word))
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[word] = res
        return res

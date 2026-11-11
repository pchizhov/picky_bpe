"""Fast byte-level Picky BPE trainer (internal).

Token ids are ints; ids 0..255 are the raw bytes. Words are lists of ints.
Pair counts are maintained incrementally from per-word local diffs, an inverted
index ``pair -> words`` limits the work per merge, and the argmax is served by a
priority queue instead of ``Counter.most_common``.
"""
from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Optional

from heapdict import heapdict

from ._bytes import ALPHABET, CHAR2ID

logger = logging.getLogger(__name__)

NUM_BYTES = 256
_SHIFT = 40
_MASK = (1 << _SHIFT) - 1
_ID_LIMIT = 1 << 20


def _prio(pair: tuple[int, int], count: int) -> int:
    """Smaller is better: higher count first, then smaller (left, right) ids."""
    return -(count << _SHIFT) + (pair[0] << 20) + pair[1]


class HeapDictQueue:
    """Indexed priority queue backed by ``heapdict`` (exact, supports updates)."""

    def __init__(self, counts: Mapping[tuple[int, int], int]):
        self._h = heapdict({p: _prio(p, c) for p, c in counts.items()})

    def update(self, pair, count: int, old: int) -> None:
        self._h[pair] = _prio(pair, count)

    def discard(self, pair) -> None:
        try:
            del self._h[pair]
        except KeyError:
            pass

    def pop(self) -> Optional[tuple[int, int]]:
        try:
            return self._h.popitem()[0]
        except IndexError:
            return None


class LazyHeapQueue:
    """``heapq`` with lazy invalidation. Pushes only on increases; stale entries
    are fixed up when popped. ``counts`` is the shared source of truth."""

    def __init__(self, counts: dict[tuple[int, int], int]):
        self._counts = counts
        self._h = [_prio(p, c) for p, c in counts.items()]
        heapq.heapify(self._h)

    def update(self, pair, count: int, old: int) -> None:
        if count > old:
            heapq.heappush(self._h, _prio(pair, count))

    def discard(self, pair) -> None:
        pass

    def pop(self) -> Optional[tuple[int, int]]:
        h, counts = self._h, self._counts
        while h:
            entry = heapq.heappop(h)
            pair = ((entry & _MASK) >> 20, entry & (_ID_LIMIT - 1))
            count = counts.get(pair)
            if count is None:
                continue
            cur = _prio(pair, count)
            if cur == entry:
                return pair
            if cur > entry:  # stale entry is too optimistic: re-queue with the true count
                heapq.heappush(h, cur)
            # cur < entry: a better entry for this pair exists elsewhere in the heap
        return None


QUEUES = {'heapdict': HeapDictQueue, 'lazy': LazyHeapQueue}


@dataclass
class TrainResult:
    token_str: list[str]  # token text in the byte-level alphabet
    freqs: list[int]
    present: list[bool]
    left: list[int]
    right: list[int]
    # ('M', left, right, new) or ('S', token, [parts])
    events: list[tuple] = field(default_factory=list)


class PickyTrainer:

    def __init__(
        self,
        vocab_size: int,
        threshold: float = 0.9999,
        num_special_tokens: int = 0,
        queue: str = 'lazy',
        min_frequency: int = 1,
        check: bool = False,
    ):
        self.vocab_size = vocab_size
        self.threshold = threshold
        self.num_special_tokens = num_special_tokens
        self.queue_cls = QUEUES[queue]
        self.min_frequency = min_frequency
        self.check = check

    # ------------------------------------------------------------------ setup
    def train(self, word_counts: Mapping[str, int], logging_step: int = 200) -> TrainResult:
        t0 = time.time()
        self.words: list[list[int]] = [[CHAR2ID[c] for c in w] for w in word_counts]
        self.freqs: list[int] = list(word_counts.values())
        self.tbytes: list[str] = list(ALPHABET)
        self.tfreq: list[int] = [0] * NUM_BYTES
        self.present: list[bool] = [True] * NUM_BYTES
        self.left: list[int] = [-1] * NUM_BYTES
        self.right: list[int] = [-1] * NUM_BYTES
        self.str2id: dict[str, int] = {b: i for i, b in enumerate(self.tbytes)}
        self.tok_words: dict[int, set[int]] = defaultdict(set)  # non-atomic token -> words (superset)
        self.index: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> words (superset)
        self.counts: dict[tuple[int, int], int] = {}
        self.events: list[tuple] = []
        self.size = self.num_special_tokens + NUM_BYTES

        counts, index = self.counts, self.index
        for wi, (w, f) in enumerate(zip(self.words, self.freqs)):
            for x in w:
                self.tfreq[x] += f
            for pair in zip(w, w[1:]):
                counts[pair] = counts.get(pair, 0) + f
                index[pair].add(wi)
        logger.info(f'Initialized {len(self.words)} words, {len(counts)} pairs in {time.time() - t0:.2f}s.')

        self.queue = self.queue_cls(counts)
        n_merges, step_time = 0, 0.0
        while self.size < self.vocab_size:
            start = time.time()
            pair = self.queue.pop()
            if pair is None or counts[pair] < self.min_frequency:
                logger.info(f'No more pairs to merge. Stopping with vocab size of {self.size}.')
                break
            freq = self._merge(pair)
            step_time += time.time() - start
            n_merges += 1
            if self.check:
                self._check()
            if n_merges % logging_step == 0:
                logger.info(
                    f'VOCABULARY SIZE: {self.size}. Merged {self.tbytes[pair[0]]!r} + {self.tbytes[pair[1]]!r} '
                    f'with frequency {freq}. Average merge time {step_time / logging_step:.4f}s.'
                )
                step_time = 0.0
        logger.info(f'Training finished in {time.time() - t0:.2f}s, vocabulary size {self.size}.')
        return TrainResult(self.tbytes, self.tfreq, self.present, self.left, self.right, self.events)

    # ------------------------------------------------------------------ helpers
    def _apply(self, delta: dict[tuple[int, int], int]) -> None:
        counts, queue = self.counts, self.queue
        for pair, d in delta.items():
            if not d:
                continue
            old = counts.get(pair, 0)
            new = old + d
            if new > 0:
                counts[pair] = new
                queue.update(pair, new, old)
            else:
                counts.pop(pair, None)
                queue.discard(pair)

    def _walk(self, t: int) -> list[int]:
        if t < NUM_BYTES or self.present[t]:
            return [t]
        return self._walk(self.left[t]) + self._walk(self.right[t])

    # ------------------------------------------------------------------ merge
    def _merge(self, pair: tuple[int, int]) -> int:
        a, b = pair
        merged = self.tbytes[a] + self.tbytes[b]
        c = self.str2id.get(merged)
        if c is None:
            c = len(self.tbytes)
            if c >= _ID_LIMIT:
                raise ValueError('Too many tokens for the priority encoding.')
            self.tbytes.append(merged)
            self.tfreq.append(0)
            self.present.append(True)
            self.left.append(a)
            self.right.append(b)
            self.str2id[merged] = c
            self.size += 1
        elif not self.present[c]:
            self.present[c] = True
            self.size += 1
            logger.debug(f'Restored previously removed token {merged!r}.')
        self.events.append(('M', a, b, c))

        words, freqs, index = self.words, self.freqs, self.index
        delta: dict[tuple[int, int], int] = {}
        total = 0
        c_words = self.tok_words[c]
        for wi in index.pop(pair, ()):
            w = words[wi]
            n = len(w)
            # locate non-overlapping occurrences of (a, b) left to right
            ot = []
            try:
                i = w.index(a)
                while True:
                    if i + 1 < n and w[i + 1] == b:
                        ot.append(i)
                        i = w.index(a, i + 2)
                    else:
                        i = w.index(a, i + 1)
            except ValueError:
                pass
            if not ot:
                continue  # stale index entry
            f = freqs[wi]
            total += len(ot) * f
            if len(ot) == 1:
                i = ot[0]
                new = w[:i]
                new.append(c)
                new.extend(w[i + 2:])
                m = n - 1
                old_ps = range(i - 1 if i else 0, min(i + 1, n - 2) + 1)
                new_ps = range(i - 1 if i else 0, min(i, m - 2) + 1)
            else:
                new, prev = [], 0
                nt = []
                for i in ot:
                    new.extend(w[prev:i])
                    nt.append(len(new))
                    new.append(c)
                    prev = i + 2
                new.extend(w[prev:])
                m = len(new)
                old_ps = {p for i in ot for p in (i - 1, i, i + 1) if 0 <= p < n - 1}
                new_ps = {p for j in nt for p in (j - 1, j) if 0 <= p < m - 1}
            for p in old_ps:
                pr = (w[p], w[p + 1])
                delta[pr] = delta.get(pr, 0) - f
            for p in new_ps:
                pr = (new[p], new[p + 1])
                delta[pr] = delta.get(pr, 0) + f
                index[pr].add(wi)
            words[wi] = new
            c_words.add(wi)
        delta.pop(pair, None)
        self.counts.pop(pair, None)
        self._apply(delta)

        self.tfreq[c] += total
        self.tfreq[a] -= total
        self.tfreq[b] -= total
        for t in ((a,) if a == b else (a, b)):
            if not self.present[t]:
                raise ValueError(f'Token {self.tbytes[t]!r} is not present in the vocabulary.')
            freq_left = self.tfreq[t]
            if self._maybe_remove(t, total):
                logger.debug(
                    f'Removed token {self.tbytes[t]!r} with frequency {freq_left} '
                    f'after merging into {merged!r} with frequency {self.tfreq[c]}.'
                )
        return total

    # ------------------------------------------------------------------ removal
    def _maybe_remove(self, t: int, merged_freq: int) -> bool:
        if t < NUM_BYTES:
            return False
        if merged_freq / (self.tfreq[t] + merged_freq) <= self.threshold:
            return False
        self.present[t] = False
        parts = self._walk(t)
        self.size -= 1

        words, freqs, index, tok_words = self.words, self.freqs, self.index, self.tok_words
        delta: dict[tuple[int, int], int] = {}
        nparts = len(parts)
        for wi in tok_words.pop(t, ()):
            w = words[wi]
            if t not in w:
                continue  # stale
            n = len(w)
            new, ot, nt = [], [], []
            for i, x in enumerate(w):
                if x == t:
                    ot.append(i)
                    nt.append(len(new))
                    new.extend(parts)
                else:
                    new.append(x)
            f = freqs[wi]
            m = len(new)
            for p in {p for i in ot for p in (i - 1, i) if 0 <= p < n - 1}:
                pr = (w[p], w[p + 1])
                delta[pr] = delta.get(pr, 0) - f
            for p in {p for j in nt for p in range(j - 1, j + nparts) if 0 <= p < m - 1}:
                pr = (new[p], new[p + 1])
                delta[pr] = delta.get(pr, 0) + f
                index[pr].add(wi)
            for x in parts:
                self.tfreq[x] += f * len(ot)
                if x >= NUM_BYTES:
                    tok_words[x].add(wi)
            words[wi] = new
        self._apply(delta)
        self.tfreq[t] = 0
        self.events.append(('S', t, parts))
        return True

    # ------------------------------------------------------------------ debug
    def _check(self) -> None:
        counts: dict[tuple[int, int], int] = {}
        tfreq = [0] * len(self.tbytes)
        for w, f in zip(self.words, self.freqs):
            for x in w:
                tfreq[x] += f
                assert self.present[x], f'absent token {self.tbytes[x]!r} in a word'
            for pair in zip(w, w[1:]):
                counts[pair] = counts.get(pair, 0) + f
        assert counts == self.counts, 'pair counts diverged from a full recount'
        assert tfreq == self.tfreq, 'token frequencies diverged from a full recount'

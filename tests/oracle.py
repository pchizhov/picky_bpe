"""The original character-level implementation, with the new deterministic tie-break
(count desc, smaller left id, smaller right id) and exact vocab-size accounting."""
from bpe_trainer import BPE, EventType
from language import Word
from utils import MCounter


class Oracle(BPE):

    @staticmethod
    def _get_characters(words):
        counter = MCounter()
        for word in words:
            counter.update(MCounter(word.str) * word.freq)
        return MCounter(dict(sorted(counter.items())))

    def run(self, word_counts):
        words = [Word(i, w, f) for i, (w, f) in enumerate(word_counts.items())]
        self._initialize_vocab(words)
        self._encode_words(words)
        pairs = self._initialize_pairs(words)
        while self.actual_vocab_size < self.desired_vocab_size and pairs:
            pair, count = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0].id, -kv[0][1].id))
            merged = pair[0].str + pair[1].str
            existing = self.str2token.get(merged) if merged in self.str2token else None
            grows = existing is None or not existing.present
            self._merge_pair(pair, pairs)
            self.actual_vocab_size += 1 if grows else 0
        return self.events

    def event_log(self):
        out = []
        for ev in self.events:
            if ev[0] == EventType.MERGE:
                out.append(('M', ev[1][0].str, ev[1][1].str))
            else:
                out.append(('S', ev[1].str, tuple(t.str for t in ev[2])))
        return out

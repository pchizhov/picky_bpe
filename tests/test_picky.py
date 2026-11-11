import json
import random

import pytest

from picky_bpe import PickyBPETokenizer
from picky_bpe._bytes import CHAR2ID
from picky_bpe._trainer import PickyTrainer
from oracle import Oracle


def synthetic_word_counts(seed=0, n=1500):
    rng = random.Random(seed)
    syl = ['ab', 'ba', 'a', 'b', 'ca', 'ac', 'aa', 'bb', 'cab', 'abc', 'th', 'er', 'in', 'ing', 'qu', 'x']
    counts = {}
    for _ in range(n):
        w = ''.join(rng.choice(syl) for _ in range(rng.randint(1, 6)))
        counts[w] = counts.get(w, 0) + int(rng.paretovariate(1.2))
    return counts


def new_events(result):
    s = lambda i: result.token_str[i]
    return [('M', s(e[1]), s(e[2])) if e[0] == 'M' else ('S', s(e[1]), tuple(s(p) for p in e[2]))
            for e in result.events]


@pytest.mark.parametrize('threshold', [1.0, 0.9, 0.6, 0.3])
@pytest.mark.parametrize('queue', ['lazy', 'heapdict'])
def test_matches_original_algorithm(threshold, queue):
    wc = synthetic_word_counts()
    n_chars = len({c for w in wc for c in w})
    extra = 150  # vocabulary growth beyond the initial alphabet
    oracle = Oracle(vocab_size=4 + n_chars + extra, coverage=1.0, threshold=threshold)
    oracle.run(wc)
    expected = oracle.event_log()
    trainer = PickyTrainer(4 + 256 + extra, threshold, num_special_tokens=4, queue=queue, check=True)
    result = trainer.train(dict(wc))
    got = new_events(result)
    assert got == expected
    if threshold < 1.0:
        assert any(e[0] == 'S' for e in got), 'test corpus should trigger removals'
    # same final frequencies / presence
    for tok in oracle.id2token.values():
        if tok.special:
            continue
        tid = trainer.str2id[tok.str]
        assert result.present[tid] == tok.present
        assert result.freqs[tid] == tok.freq


def corpus(seed=1):
    wc = synthetic_word_counts(seed)
    return [(' ' + w + ' ') * f for w, f in list(wc.items())[:300] for f in [1 + f % 3]]


@pytest.fixture(scope='module')
def tok():
    return PickyBPETokenizer.train_new_from_iterator(corpus(), vocab_size=380, threshold=0.7, check=True,
                                                  pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')


def test_lossless_roundtrip(tok):
    for s in ['hello world', 'ab ba cab  ', 'Привет, мир! 你好 😀', '\n\t x\x00\xff', '', ' ', 'a' * 50,
              '\ud83d'.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore') + 'é']:
        ids = tok(s)['input_ids']
        assert tok.decode(ids) == s
        assert tok.unk_token_id not in ids


def test_fast_encoder_matches_event_sequence_replay(tok):
    """Ground truth: apply every event in order to the word (old `_encode_word_by_event_sequence`)."""
    enc = tok._encoder
    kinds = [('M', enc.ev_pair[k], enc.ev_new[k]) if enc.ev_parts[k] is None else ('S', enc.ev_new[k], enc.ev_parts[k])
             for k in range(enc.num_events)]

    def replay(word: str):
        w = [CHAR2ID[c] for c in word]
        for kind, x, y in kinds:
            if kind == 'M':
                a, b = x
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        out.append(y); i += 2
                    else:
                        out.append(w[i]); i += 1
                w = out
            else:
                w = [p for t in w for p in (y if t == x else [t])]
        return w

    rng = random.Random(3)
    words = list(synthetic_word_counts(7, 400)) + \
            [''.join(rng.choice('abcqthxeringÃ©') for _ in range(rng.randint(1, 12))) for _ in range(300)]
    for w in words:
        assert enc.encode_internal(w) == replay(w), w


def test_final_tokens_are_in_vocab(tok):
    enc = tok._encoder
    for w in synthetic_word_counts(11, 300):
        assert all(i >= 0 for i in enc.encode_pretoken(w))


def test_hf_save_load_roundtrip(tok, tmp_path):
    tok.save_pretrained(tmp_path)
    s = 'ab ba cab thing 你好 aaa'
    for loaded in (PickyBPETokenizer.from_pretrained(tmp_path),):
        assert loaded.get_vocab() == tok.get_vocab()
        assert loaded(s)['input_ids'] == tok(s)['input_ids']
    import picky_bpe  # noqa: F401  (registers with AutoTokenizer)
    from transformers import AutoTokenizer
    auto = AutoTokenizer.from_pretrained(tmp_path)
    assert type(auto).__name__ == 'PickyBPETokenizer'
    assert auto(s)['input_ids'] == tok(s)['input_ids']


def test_hf_interface(tok):
    batch = tok(['ab ba', 'cab cab cab cab'], padding=True, truncation=True, max_length=8, return_tensors='pt')
    assert batch['input_ids'].shape[0] == 2 and batch['input_ids'].shape[1] <= 8
    assert batch['attention_mask'].shape == batch['input_ids'].shape
    s = f'ab {tok.eos_token} ba'
    ids = tok(s)['input_ids']
    assert tok.eos_token_id in ids
    assert tok.decode(ids, skip_special_tokens=True) == 'ab  ba'
    assert min(tok.pad_token_id, tok.unk_token_id, tok.eos_token_id) >= tok.vocab_size  # HF added tokens
    assert len(tok) == tok.vocab_size + 3
    assert tok.batch_decode([tok('ab ba')['input_ids']]) == ['ab ba']
    assert json.loads(json.dumps(tok.get_vocab()))

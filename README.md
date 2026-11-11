# Picky BPE

This repository contains a prototype code for the paper "BPE Gets Picky: Efficient Vocabulary Refinement
During Tokenizer Training", which was presented at EMNLP 2024.

[[`ACL Anthology`](https://aclanthology.org/2024.emnlp-main.925)] [[`arXiv`](https://arxiv.org/abs/2409.04599)] [[`BibTeX`](#referencing)] 

## Installation

```
$ pip install -e .          # deps: transformers (incl. tokenizers), heapdict
```

## Usage

`PickyBPETokenizer` is a byte-level (lossless, no `<UNK>` needed) Hugging Face `PreTrainedTokenizer`.
Training, encoding and persistence all go through this one class:

```python
from picky_bpe import PickyBPETokenizer

tok = PickyBPETokenizer.train_from_files("train.txt", vocab_size=8192, threshold=0.9)
# or: PickyBPETokenizer.train_new_from_iterator(texts, vocab_size=8192, threshold=0.9)
tok.save_pretrained("picky-8k")

tok = PickyBPETokenizer.from_pretrained("picky-8k")   # AutoTokenizer.from_pretrained works after `import picky_bpe`
tok("Hello world!", return_tensors="pt")
tok.decode(tok("Hello world!")["input_ids"])
```

Training options (`train_new_from_iterator` / `train_from_files`):

```
vocab_size     Total vocabulary size (special tokens + 256 bytes + learned tokens)
threshold      IoS threshold; a token is removed when IoS > threshold. 1.0 = vanilla BPE (default: 0.9999)
pattern        Custom pretokenization regex (default: HF ByteLevel, i.e. the GPT-2 pattern)
<name>_token   Standard HF special tokens (eos_token=..., pad_token=..., additional_special_tokens=[...]);
               added by HF after the base vocabulary, they count towards vocab_size
queue          "lazy" (heapq with lazy invalidation, default) or "heapdict"
min_frequency  Stop when the best pair is rarer than this
check          Debug: re-count all pairs after every merge and assert they match
```

Pre-tokenization (`tokenizers` ByteLevel) and special-token handling are the stock Hugging Face ones.
Tokenizer options (`__init__` / `from_pretrained(..., **kwargs)`): `add_prefix_space`, `add_bos_token`, `add_eos_token`.

A command-line interface is also available:

```
$ python -m picky_bpe train --input_file train.txt --output_dir picky-8k --vocab_size 8192 --threshold 0.9
$ python -m picky_bpe encode --tokenizer picky-8k --input_file train.txt --output_file train.tok.txt --return_type str
```

Saved files: `vocab.json`, `picky_merges.txt` (ordered merge `M` and removal `S` events; encoding replays them),
`tokenizer_config.json`, `special_tokens_map.json`.

The tokenizer is implemented in Python (exact Picky semantics: merges and token removals are interleaved, which the
Rust `tokenizers` BPE cannot express). Encoding caches results per pretoken.

The original character-level prototype used in the paper is kept in `tests/reference/` and serves as the oracle
for the test-suite (`pytest`).

## Referencing

To cite PickyBPE:

```
@inproceedings{chizhov-etal-2024-bpe,
    title = "{BPE} Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training",
    author = "Chizhov, Pavel  and
      Arnett, Catherine  and
      Korotkova, Elizaveta  and
      Yamshchikov, Ivan P.",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.925",
    pages = "16587--16604",
    abstract = "Language models can greatly benefit from efficient tokenization. However, they still mostly utilize the classical Byte-Pair Encoding (BPE) algorithm, a simple and reliable method. BPE has been shown to cause such issues as under-trained tokens and sub-optimal compression that may affect the downstream performance. We introduce PickyBPE, a modified BPE algorithm that carries out vocabulary refinement during tokenizer training by removing merges that leave intermediate {``}junk{''} tokens. Our method improves vocabulary efficiency, eliminates under-trained tokens, and does not compromise text compression. Our experiments show that this method either improves downstream performance or does not harm it.",
}
```

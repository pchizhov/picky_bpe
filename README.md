# Picky BPE

This repository contains a prototype code for the paper "BPE Gets Picky: Efficient Vocabulary Refinement
During Tokenizer Training", which was presented at EMNLP 2024.

[[`ACL Anthology`](https://aclanthology.org/2024.emnlp-main.925)] [[`arXiv`](https://arxiv.org/abs/2409.04599)] [[`BibTeX`](#referencing)] 

## Training

For training you should use `bpe_trainer.py` script. For example, the following command trains a 
Picky BPE tokenizer with vocabulary size 8192 and IoS threshold of 0.9.

```
$ python bpe_trainer.py --input_file train.txt --model_file model.json --vocab_size 8192 --threshold 0.9
```

The complete list of options is:

```
Args:
  --input_file     Path to the training corpus
  --model_file     Path to save the model
  --vocab_size     Desired vocabulary size
  --threshold      Desired IoS threshold
  --coverage       Relative symbol coverage for the initial vocabulary (default: 0.9999)
  --pad_id         PAD token id (default: 0)
  --unk_id         UNK token id (default: 1)
  --bos_id         BOS token id (default: 2)
  --eos_id         EOS token id (default: 3)
  --logging_step   Frequency of merges logging (default: 200)
```

## Tokenization

To apply the trained Picky BPE model, use the `picky_tokenize.py` script. For example:

```
$ python picky_tokenize.py --bpe_model model.json --input_file train.txt --output_file train.tok.txt
```

The complete list of options is:

```
Args:
  --model_file    Path to the trained model
  --input_file    Path to the raw corpus
  --output_file   Path to save the tokenized corpus
  --return_type   Whether to output tokens ("str") or ids ("int") (default: "str")
```

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

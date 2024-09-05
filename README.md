# Picky BPE

This repository contains a prototype code for the paper "BPE Gets Picky: Efficient Vocabulary Refinement
During Tokenizer Training".

[arxiv](https://arxiv.org)

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

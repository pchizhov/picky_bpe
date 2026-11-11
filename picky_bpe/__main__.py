"""CLI: ``python -m picky_bpe train|encode ...``"""
import argparse
import logging

from . import PickyBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(prog='picky_bpe')
    sub = parser.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Train a byte-level Picky BPE tokenizer.')
    t.add_argument('--input_file', nargs='+', required=True, help='Training text file(s), one text per line.')
    t.add_argument('--output_dir', required=True, help='Directory to save the tokenizer to.')
    t.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size (specials + 256 bytes + merges).')
    t.add_argument('--threshold', type=float, default=0.9999, help='IoS threshold (1.0 = vanilla BPE).')
    t.add_argument('--pattern', default=None, help='Pretokenization regex (default: GPT-2).')
    for name in ('pad', 'unk', 'bos', 'eos'):
        t.add_argument(f'--{name}_token', default=None, help=f'{name.upper()} token string (optional).')
    t.add_argument('--queue', choices=['lazy', 'heapdict'], default='lazy')
    t.add_argument('--logging_step', type=int, default=200)

    e = sub.add_parser('encode', help='Tokenize a text file (one text per line).')
    e.add_argument('--tokenizer', required=True, help='Directory of a trained tokenizer.')
    e.add_argument('--input_file', required=True)
    e.add_argument('--output_file', required=True)
    e.add_argument('--return_type', choices=['str', 'int'], default='str')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.cmd == 'train':
        tok = PickyBPETokenizer.train_from_files(
            args.input_file, args.vocab_size, threshold=args.threshold, pattern=args.pattern,
            queue=args.queue, logging_step=args.logging_step,
            **{k: getattr(args, k) for k in ('pad_token', 'unk_token', 'bos_token', 'eos_token') if getattr(args, k)},
        )
        tok.save_pretrained(args.output_dir)
    else:
        tok = PickyBPETokenizer.from_pretrained(args.tokenizer)
        with open(args.input_file, encoding='utf-8') as fin, open(args.output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                toks = tok.tokenize(line.rstrip('\n'))
                out = toks if args.return_type == 'str' else tok.convert_tokens_to_ids(toks)
                fout.write(' '.join(map(str, out)) + '\n')


if __name__ == '__main__':
    main()

import argparse
import json


def trim_vanilla_model(input_file: str, output_file: str, vocab_size: int) -> None:
    with open(input_file, 'r') as f:
        bpe_model = json.load(f)
    tokens = [token for token in bpe_model['tokens'] if token['id'] < vocab_size]
    id2int = {int(id_): int_ for id_, int_ in bpe_model['id2int'].items() if int(id_) < vocab_size}
    int2id = {int_: id_ for id_, int_ in id2int.items()}
    with open(output_file, 'w') as f:
        json.dump({
            'tokens': tokens,
            'id2int': id2int,
            'int2id': int2id
        }, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to the input model file.')
    parser.add_argument('--output_file', type=str, help='Path to the output model file.')
    parser.add_argument('--vocab_size', type=int, help='Desired vocabulary size.')
    args = parser.parse_args()
    trim_vanilla_model(args.input_file, args.output_file, args.vocab_size)

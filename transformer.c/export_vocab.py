import json


def export_vocab(vocab, path):
    total_vocab_size_in_bytes = sum([len(token.encode('utf-8')) for token in vocab.keys()])
    with open(path, 'wb') as f:
        f.write(total_vocab_size_in_bytes.to_bytes(4, byteorder='little'))
        for token, idx in vocab.items():
            size_in_bytes = len(token.encode('utf-8'))
            f.write(size_in_bytes.to_bytes(4, byteorder='little'))
            f.write(idx.to_bytes(4, byteorder='little'))
            f.write(token.encode('utf-8'))


if __name__ == '__main__':
    with open('vocab.json', 'r') as f:
        v = json.load(f)

    export_vocab(v, "vocab.bin")
import pickle
import tiktoken

class EncoderDecoder:

    def __init__(self, meta_path=None):
        if meta_path and self._is_char_encoding(meta_path):
            self.impl = CharEncoderDecoder(meta_path)
        else:
            self.impl = BPEEncoderDecoder()

    def encode(self, text):
        return self.impl.encode(text)

    def decode(self, tokens):
        return self.impl.decode(tokens)

    def _is_char_encoding(self, meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            return 'itos' in meta

class CharEncoderDecoder:

    def __init__(self, meta_path):
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)

        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])


class BPEEncoderDecoder:

    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, tokens):
        return self.encoder.decode(tokens)
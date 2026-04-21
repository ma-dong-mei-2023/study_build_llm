import tiktoken


class Tokenizer:
    def __init__(self, encoding_name="gpt2"):
        self._enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self):
        return self._enc.n_vocab

    def encode(self, text: str) -> list:
        return self._enc.encode(text)

    def decode(self, ids: list) -> str:
        return self._enc.decode(ids)

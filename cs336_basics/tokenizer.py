from typing import Iterable, Iterator
import regex as re
from collections import defaultdict
import json
from functools import lru_cache


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_re = re.compile(PAT)

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens is not None else None
        self.split_re = re.compile("(" + "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True))) + ")") if special_tokens is not None else None
        self.vocab2idx = {v: k for k, v in vocab.items()}
        self.idx2vocab = vocab
        
        if special_tokens is not None:
            for special_token in special_tokens:
                special_token_encoding = special_token.encode()
                if special_token_encoding not in self.vocab2idx:
                    self.vocab2idx[special_token_encoding] = len(self.vocab2idx)
                    self.idx2vocab[len(self.idx2vocab)] = special_token_encoding

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        output = list()
        if self.split_re is not None:
            chunks = self.split_re.split(text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if self.special_tokens is not None and chunk in self.special_tokens:
                output.extend([self.vocab2idx[chunk.encode()]])
                continue
            for token in self.token_re.finditer(chunk):
                token_bytes = token.group().encode()
                if token_bytes in self.vocab2idx:
                    output.extend([self.vocab2idx[token_bytes]])
                else:
                    encoding = self.encode_one(token_bytes)
                    output.extend(encoding)

        return output

    def encode_one(self, word: bytes):
        n = len(word)
        sym = [bytes([b]) for b in word]
        nxt = list(range(1, n)) + [-1]
        alive = [True] * n

        byte2node = defaultdict(set)
        for i, s in enumerate(sym):
            byte2node[s].add(i)

        for merge in self.merges:
            if merge[0] in byte2node and merge[1] in byte2node:
                node_idxs = list(byte2node[merge[0]])
                for node_idx in node_idxs:
                    next_node_idx = nxt[node_idx]
                    if alive[node_idx] and next_node_idx != -1 and alive[next_node_idx] and sym[next_node_idx] == merge[1]:
                        alive[next_node_idx] = False
                        sym[node_idx] = merge[0] + merge[1]
                        nxt[node_idx] = nxt[next_node_idx]
                        byte2node[sym[node_idx]].add(node_idx)
                        byte2node[merge[0]].discard(node_idx)
                        if len(byte2node[merge[0]]) == 0:
                            del byte2node[merge[0]]
                    
        output = list()
        i = 0
        while i != -1:
            if alive[i]:
                output.append(self.vocab2idx[sym[i]])
            i = nxt[i]

        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        string_bytes = b''.join(self.idx2vocab[id] for id in ids)
        string = string_bytes.decode(errors='replace')

        return string


if __name__ == '__main__':
    # string = 'the cat ate'
    # vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    # special_tokens = ["<|endoftext|>"]
    # tokenizer = Tokenizer(vocab, merges, special_tokens)

    vocab_dir = "/work/users/o/w/owenjf/stanford_cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
    merge_dir = "/work/users/o/w/owenjf/stanford_cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt"
    string = "🙃"
    tokenizer = Tokenizer.from_files(vocab_dir, merge_dir)

    encoding = tokenizer.encode(string)
    decoding = tokenizer.decode(encoding)
    
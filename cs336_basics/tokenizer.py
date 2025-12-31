from typing import Iterable, Iterator, Optional
import pickle
import regex as re
from collections import defaultdict
import itertools
from dataclasses import dataclass


@dataclass
class Node:
    sym: bytes
    next: Optional["Node"]


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    split_re = re.compile(PAT)

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.merges = merges
        self.token_re = re.compile("|".join(map(re.escape, special_tokens)))
        self.idx2vocab = vocab
        self.vocab2idx = {v: k for k, v in vocab.items()}

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        output = list()
        token_encoding_lookup = dict()
        for chunk in self.split_re.split(text):
            for token in self.token_re.finditer(chunk):
                token_bytes = token.group().encode()
                if token_bytes in token_encoding_lookup:
                    output.append(token_encoding_lookup[token_bytes])
                else:
                    encoding = self.encode_one(token_bytes)
                    token_encoding_lookup[token_bytes] = encoding
                    output.append(encoding)

        output = list(itertools.chain.from_iterable(output))
        return output

    def encode_one(self, word: bytes):
        byte2node = defaultdict(set)
        prev = None
        head = None
        for token_byte in word:
            node = Node(sym=token_byte, next=None)
            if prev is not None:
                prev.next = node
            else:
                head = node
            prev = node
            byte2node[token_byte].add(node)

        for merge in self.merges:
            if merge[0] in byte2node and merge[1] in byte2node:
                nodes = list(byte2node.get(merge[0], ()))
                for node in nodes:
                    if node.next and node.next.sym == merge[1]:
                        node.sym = merge[0] + merge[1]
                        byte2node[merge[0]].discard(node)
                        byte2node[merge[1]].discard(node.next)
                        if node.next.next:
                            node.next = node.next.next
                        else:
                            node.next = None
                        byte2node[node.sym].add(node)
                if len(byte2node[merge[0]]) == 0:
                    del byte2node[merge[0]]
                if len(byte2node[merge[1]]) == 0:
                    del byte2node[merge[1]]
                    
        output = list()
        curr = head
        while curr:
            output.append(self.vocab2idx[curr.sym])
            curr = curr.next

        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        string = ''
        for id in ids:
            string += self.idx2vocab[id].decode()

        return string
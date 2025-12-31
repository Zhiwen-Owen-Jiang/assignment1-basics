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
    alive: bool = True


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_re = re.compile(PAT)

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.merges = merges
        self.split_re = re.compile("|".join(map(re.escape, special_tokens)))
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
        all_nodes = list()
        prev = None
        head = None
        for i, x in enumerate(word):
            token_byte = chr(x).encode()
            node = Node(sym=token_byte, next=None)
            if prev is not None:
                prev.next = node
            else:
                head = node
            prev = node
            byte2node[token_byte].add(i)
            all_nodes.append(node)

        for merge in self.merges:
            if merge[0] in byte2node and merge[1] in byte2node:
                node_idxs = list(byte2node.get(merge[0], ()))
                for node_idx in node_idxs:
                    node = all_nodes[node_idx]
                    next_node = node.next
                    if node.alive and next_node and next_node.alive and next_node.sym == merge[1]:
                        next_node.alive = False
                        node.sym = merge[0] + merge[1]
                        if next_node.next:
                            node.next = next_node.next
                        else:
                            node.next = None
                        byte2node[node.sym].add(node_idx)
                        byte2node[merge[0]].discard(node_idx)
                        if len(byte2node[merge[0]]) == 0:
                            del byte2node[merge[0]]
                    
        output = list()
        curr = head
        while curr and curr.alive:
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


if __name__ == '__main__':
    string = 'the cat ate'
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    special_tokens = ["<|endoftext|>"]
    
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    encoding = tokenizer.encode(string)
    decoding = tokenizer.decode(encoding)
    
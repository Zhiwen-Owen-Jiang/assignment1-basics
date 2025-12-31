import regex as re
import os
from typing import BinaryIO, Optional
from dataclasses import dataclass
from collections import defaultdict


Pair = tuple[bytes, bytes]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@dataclass
class Node:
    sym: bytes
    prev: Optional[int]
    next: Optional[int]
    alive: bool = True


@dataclass
class Word:
    head: Optional[int]
    freq: int


class BPE:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, input_path, vocab_size, special_tokens, **kwargs):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.word2freq = defaultdict(int)
        self.pair2count = defaultdict(int)
        self.count2pairs = defaultdict(set)
        self.pair2occ = defaultdict(set)
        self.nodes = list()
        self.words = list()

    def train_bpe(self, desired_num_chunks=100, num_processes=4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        if "<|endoftext|>" not in self.special_tokens:
            self.special_tokens.append("<|endoftext|>")

        # does special_tokens include `b"<|endoftext|>"`?
        assert self.vocab_size > len(self.special_tokens) + 256

        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pattern = "|".join(map(re.escape, self.special_tokens))
                splited_chunk = re.split(pattern, chunk)
                # splited_chunk = re.split("|".join(special_tokens), chunk)

                for chunk in splited_chunk:
                    # pre-tokenization
                    for token in re.finditer(self.PAT, chunk):
                        token_bytes = token.group().encode()
                        self.word2freq[token_bytes] += 1
            
            # build the linked list for each word token as bytes
            for word, freq in self.word2freq.items():
                head = self.build_word(word)
                self.words.append(Word(head=head, freq=freq))

            # pair occurrence index and pair2count
            for word_idx, (word, freq) in enumerate(self.word2freq.items()):
                start_idx = self.words[word_idx].head
                for i in range(len(word) - 1):
                    node_idx = start_idx + i
                    self.pair2occ[(word[i:i+1], word[i+1:i+2])].add((word_idx, node_idx))
                    self.pair2count[(word[i:i+1], word[i+1:i+2])] += freq

            # count2pairs
            for pair, count in self.pair2count.items():
                self.count2pairs[count].add(pair)
                
        merges, merged = self.bpe()
        vocab_list = [x.encode() for x in self.special_tokens] + [bytes([i]) for i in range(256)] + merged
        vocab = {i: bytes(x) for i, x in enumerate(vocab_list)}
        
        return vocab, merges

    def new_node(self, sym: bytes, prev: Optional[int]) -> int:
        idx = len(self.nodes)
        self.nodes.append(Node(sym=sym, prev=prev, next=None))
        if prev is not None:
            self.nodes[prev].next = idx
        return idx

    def build_word(self, tok: bytes) -> Optional[int]:
        """
        Build linked-list nodes for this token.
        Returns head node index.

        """
        if not tok:
            return None
        
        bts = list(tok)
        head = None
        prev = None

        for b in bts:
            sym = bytes([b])
            cur = self.new_node(sym, prev)
            if head is None:
                head = cur
            prev = cur

        return head

    def bucket_move_pair(self, p: Pair, newc: int) -> None:
        """
        If pair exists, remove the pair from count2pairs and update
        If pair does not exist, directly update
        
        """
        oldc = self.pair2count.get(p)

        # If it existed before, remove it from the old bucket
        if oldc is not None:
            s = self.count2pairs.get(oldc)
            if s is not None:
                s.discard(p)
                if not s:
                    del self.count2pairs[oldc]

        # If new count <= 0, delete it completely
        if newc <= 0:
            if oldc is not None:
                del self.pair2count[p]
            return
        
        self.pair2count[p] = newc
        self.count2pairs[newc].add(p)

    def add_to_pair(self, p: Pair, delta: int):
        """
        Add a new pair or existing pair with change
        
        """
        if delta == 0:
            return
        oldc = self.pair2count.get(p, 0)
        self.bucket_move_pair(p, oldc + delta)

    def merge_once(self, A: bytes, B: bytes) -> None:
        key = (A, B)
        occs = list(self.pair2occ.get(key, ()))

        for word_idx, node_idx in occs:
            # extract all nodes
            node1 = self.nodes[node_idx] # left node
            if not node1 or not node1.alive or node1.sym != A:
                continue
            node2 = self.nodes[node1.next] if node1.next is not None else None # right node
            if not node2 or not node2.alive or node2.sym != B:
                continue    
            pre_node = self.nodes[node1.prev] if node1.prev is not None else None # pre of left node
            next_node = self.nodes[node2.next] if (node2 and node2.next is not None) else None # next of right node
            
            # update right side
            if node2.sym == B:
                node1.sym = A + B # merge (A, B) to one symbol
                node2.alive = False
                if next_node and next_node.alive:
                    node2_idx = node1.next
                    self.pair2occ[(A + B, next_node.sym)].add((word_idx, node_idx)) # if the next of right exists, create a new occ
                    self.pair2occ[(B, next_node.sym)].discard((word_idx, node2_idx)) # remove the original occ
                    self.add_to_pair((A + B, next_node.sym), self.words[word_idx].freq)
                    self.add_to_pair((B, next_node.sym), -self.words[word_idx].freq)
                    next_node.prev = node_idx # connect the next node to node1
                node1.next = node2.next # connect the node1 to the next node

            # update left side
            if pre_node and pre_node.alive:
                self.pair2occ[(pre_node.sym, A + B)].add((word_idx, node1.prev)) # if the pre of left exists, create a new occ
                self.pair2occ[(pre_node.sym, A)].discard((word_idx, node1.prev)) # remove the original occ
                self.add_to_pair((pre_node.sym, A + B), self.words[word_idx].freq)
                self.add_to_pair((pre_node.sym, A), -self.words[word_idx].freq)

            # update pair (A, B)
            self.add_to_pair(key, -self.words[word_idx].freq)
            self.pair2occ[key].discard((word_idx, node_idx))
            s = self.pair2occ.get(key)
            if s is not None and not s:
                del self.pair2occ[key]

    def bpe(self,):
        merges = list()
        merged = list()
        vocab_size = self.vocab_size - len(self.special_tokens) - 256

        while len(merges) < vocab_size:
            if not self.count2pairs:
                break
            max_count = max(self.count2pairs)
            bucket = self.count2pairs[max_count]
            if not bucket:
                del self.count2pairs[max_count]
                continue

            pair = max(bucket)
            if pair not in self.pair2occ:   # no occurrences left
                bucket.discard(pair)
                if not bucket: del self.count2pairs[max_count]
                continue

            merges.append(pair)
            merged_token = pair[0] + pair[1]
            merged.append(merged_token)
            self.merge_once(*pair)

        return merges, merged


if __name__ == '__main__':
    # bpe1 = BPE("/work/users/o/w/owenjf/stanford_cs336/data/simple_case1.txt", 256+1+3, ["<|endoftext|>"])
    bpe1 = BPE("/work/users/o/w/owenjf/stanford_cs336/assignment1-basics/tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    vocab, merges = bpe1.train_bpe()
    vocab, merges
    # train_bpe("/work/users/o/w/owenjf/stanford_cs336/data/toy_example.txt", 256+1+6, ["<|endoftext|>"])

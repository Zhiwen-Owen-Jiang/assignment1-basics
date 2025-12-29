import regex as re
import os
from typing import BinaryIO
from collections import defaultdict


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


class BPE:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, input_path, vocab_size, special_tokens, **kwargs):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pre_token_count = defaultdict(int)
        self.adjcent_bytes_count = defaultdict(int)
        self.count_adjcent_bytes = defaultdict(set)

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
                        self.pre_token_count[token_bytes] += 1
                
            for vocab, count in self.pre_token_count.items():
                for i in range(len(vocab) - 1):
                    self.adjcent_bytes_count[(vocab[i:i+1], vocab[i+1:i+2])] += count

            for token, count in self.adjcent_bytes_count.items():
                self.count_adjcent_bytes[count].add(token)
                
        merges, merged = self.bpe()
        vocab_list = [x.encode() for x in self.special_tokens] + [bytes(x) for x in range(256)] + merged
        vocab = {i: bytes(x) for i, x in enumerate(vocab_list)}
        
        return vocab, merges

    def bpe(self,):
        merges = list()
        merged = list()
        vocab_size = self.vocab_size - len(self.special_tokens) - 256

        while len(merges) < vocab_size:
            max_count = max(self.count_adjcent_bytes.keys())
            bytes_to_merge = max(self.count_adjcent_bytes[max_count])
            merges.append(bytes_to_merge)
            merged_token = bytes_to_merge[0] + bytes_to_merge[1]
            merged.append(merged_token)

            for pre_token, count in self.pre_token_count.items():
                for i in range(len(pre_token) - len(merged_token) + 1):
                    if merged_token == pre_token[i:i+len(merged_token)]:
                        if i > 0:
                            self.add((pre_token[i-1:i], merged_token), count)
                            for j in range(1, len(merged_token)):
                                # for k in range(i):
                                    self.remove((pre_token[i-1:i], merged_token[:j]), count)      

                        if i + len(merged_token) < len(pre_token):
                            self.add((merged_token, pre_token[i+len(merged_token):i+len(merged_token)+1]), count)
                            for j in range(1, len(merged_token)):
                                # for k in range(i+len(merged_token)+1, len(pre_token)+1):
                                    self.remove((merged_token[j:], pre_token[i+len(merged_token):i+len(merged_token)+1]), count)
                        
                        self.remove(bytes_to_merge, count)

        return merges, merged

    def add(self, token, count):
        self.adjcent_bytes_count[token] += count 
        self.count_adjcent_bytes[self.adjcent_bytes_count[token]].add(token)

    def remove(self, token, count):
        if token in self.adjcent_bytes_count:
            self.count_adjcent_bytes[self.adjcent_bytes_count[token]].remove(token)
            if len(self.count_adjcent_bytes[self.adjcent_bytes_count[token]]) == 0:
                del self.count_adjcent_bytes[self.adjcent_bytes_count[token]]
            self.adjcent_bytes_count[token] -= count
            if self.adjcent_bytes_count[token] == 0:
                del self.adjcent_bytes_count[token]
            else:
                self.count_adjcent_bytes[self.adjcent_bytes_count[token]].add(token)


if __name__ == '__main__':
    # bpe1 = BPE("/work/users/o/w/owenjf/stanford_cs336/data/toy_example.txt", 256+1+6, ["<|endoftext|>"])
    bpe1 = BPE("/work/users/o/w/owenjf/stanford_cs336/assignment1-basics/tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    vocab, merges = bpe1.train_bpe()
    vocab, merges
    # train_bpe("/work/users/o/w/owenjf/stanford_cs336/data/toy_example.txt", 256+1+6, ["<|endoftext|>"])

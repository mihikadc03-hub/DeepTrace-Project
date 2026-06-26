"""
Task 1 & 2: Load TinyStories, tokenize with tiktoken, save to binary files.

What goes in / what comes out / what is the model trying to learn (Task 1 answer):
- Input: a sequence of token IDs representing part of a story.
- Output: a probability distribution over the vocabulary for "what token comes next".
- Goal: learn the statistical patterns of language well enough to predict the next
  token in a sequence, which in aggregate lets it generate coherent short stories.
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
enc = tiktoken.get_encoding("gpt2")  # 50257 vocab, well-tested BPE tokenizer


def quick_tiktoken_demo():
    text = "Once upon a time there was a little girl."
    ids = enc.encode(text)
    decoded = enc.decode(ids)
    print("Demo sentence:", text)
    print("Token IDs:", ids)
    print("Decoded back:", decoded)
    assert decoded == text
    print()


def tokenize_example(example):
    ids = enc.encode_ordinary(example["text"])  # no special tokens
    ids.append(enc.eot_token)  # mark end of each story
    return {"ids": ids, "len": len(ids)}


def main():
    quick_tiktoken_demo()

    print("Loading TinyStories dataset from HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories")

    # TinyStories ships with 'train' and 'validation' splits already
    print(dataset)
    print("\nSample story:\n", dataset["train"][0]["text"][:300])

    for split_name, split_data in dataset.items():
        print(f"\nTokenizing split: {split_name} ({len(split_data)} stories)")
        tokenized = split_data.map(
            tokenize_example,
            remove_columns=["text"],
            desc=f"tokenizing {split_name}",
            num_proc=4,
        )

        total_len = np.sum(tokenized["len"], dtype=np.uint64)
        filename = os.path.join(OUT_DIR, f"{'train' if split_name == 'train' else 'val'}.bin")
        dtype = np.uint16  # gpt2 vocab fits in uint16 (max 50256)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(int(total_len),))

        idx = 0
        for batch in tokenized.iter(batch_size=1000):
            for ids in batch["ids"]:
                arr[idx: idx + len(ids)] = ids
                idx += len(ids)
        arr.flush()
        print(f"Saved {filename}  ({total_len:,} tokens)")

    # Verification step
    print("\nVerifying train.bin...")
    train_path = os.path.join(OUT_DIR, "train.bin")
    arr = np.memmap(train_path, dtype=np.uint16, mode="r")
    print("Total tokens in train.bin:", len(arr))
    first_50 = arr[:50].tolist()
    print("First 50 token IDs:", first_50)
    print("Decoded back to text:\n", enc.decode(first_50))


if __name__ == "__main__":
    main()

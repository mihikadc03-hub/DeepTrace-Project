"""
Task 3: Data loader — produces (x, y) batches where y is x shifted right by one.
"""

import os
import numpy as np
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def get_batch(split, block_size=128, batch_size=32, device="cpu"):
    """
    split: 'train' or 'val'
    Returns x, y tensors of shape (batch_size, block_size).
    y[i] = x[i] shifted one position to the right (the next-token targets).
    """
    filename = os.path.join(DATA_DIR, f"{split}.bin")
    # re-open memmap every call (cheap) to avoid memory leaks from large files
    data = np.memmap(filename, dtype=np.uint16, mode="r")

    # pick random starting indices, leaving room for block_size+1 tokens
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    if device == "cuda" and torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = get_batch("train", block_size=8, batch_size=4, device=device)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x[0]:", x[0].tolist())
    print("y[0]:", y[0].tolist())
    print("y is x shifted right by one:", torch.equal(x[0][1:], y[0][:-1]))

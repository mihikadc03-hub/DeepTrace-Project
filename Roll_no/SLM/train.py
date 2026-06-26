"""
Task 6: Pre-training loop for the SLM.

Run `python prepare_data.py` first to generate train.bin / val.bin.
"""

import os
import time
import torch
import matplotlib.pyplot as plt

from model import GPT, GPTConfig
from data_loader import get_batch
from training_setup import build_optimizer_and_scheduler

# ----------------------------- config -----------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLOCK_SIZE = 128
BATCH_SIZE = 32
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1

LR = 3e-4
WEIGHT_DECAY = 0.1
MAX_ITERS = 2000          # bump this up for a stronger model if you have the compute
EVAL_INTERVAL = 200
EVAL_ITERS = 50           # number of batches averaged for estimate_loss

CKPT_PATH = os.path.join(OUT_DIR, "best_model.pt")
PLOT_PATH = os.path.join(OUT_DIR, "loss_curve.png")


@torch.no_grad()
def estimate_loss(model):
    """Average loss over EVAL_ITERS batches for both train and val splits."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, device=DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Using device: {DEVICE}")

    config = GPTConfig(
        vocab_size=50257,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )
    model = GPT(config).to(DEVICE)
    print(f"Model parameter count: {model.num_params():,}")

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=LR, weight_decay=WEIGHT_DECAY, max_iters=MAX_ITERS
    )

    train_losses, val_losses, eval_steps = [], [], []
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(1, MAX_ITERS + 1):
        x, y = get_batch("train", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, device=DEVICE)

        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if it % EVAL_INTERVAL == 0 or it == 1:
            losses = estimate_loss(model)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            eval_steps.append(it)
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                f"elapsed {dt:.1f}s"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "iter": it,
                        "val_loss": best_val_loss,
                    },
                    CKPT_PATH,
                )
                print(f"  -> saved new best checkpoint (val loss {best_val_loss:.4f})")

    # plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(eval_steps, train_losses, label="train loss")
    plt.plot(eval_steps, val_losses, label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title("Training & validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"\nSaved loss curve to {PLOT_PATH}")
    print(f"Best val loss: {best_val_loss:.4f}  (checkpoint at {CKPT_PATH})")


if __name__ == "__main__":
    main()

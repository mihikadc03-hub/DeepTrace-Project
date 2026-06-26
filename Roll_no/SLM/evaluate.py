"""
Task 7: Evaluate the trained model — perplexity, overfitting check, qualitative samples.

Perplexity = exp(val_loss). It measures how "surprised" the model is by the real
text on average; lower is better. A perplexity of 1 would mean the model perfectly
predicts every token; random guessing over a 50257-word vocab gives a much larger
number.
"""

import os
import math
import torch
import tiktoken

from model import GPT, GPTConfig
from data_loader import get_batch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(OUT_DIR, "best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 100

enc = tiktoken.get_encoding("gpt2")


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt


@torch.no_grad()
def compute_val_loss(model, config):
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        x, y = get_batch("val", block_size=config.block_size, batch_size=32, device=DEVICE)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def generate_sample(model, config, prompt, max_new_tokens=80, temperature=0.8):
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
    return enc.decode(out[0].tolist())


def main():
    model, config, ckpt = load_model()
    print(f"Loaded checkpoint from iter {ckpt['iter']}, recorded val loss {ckpt['val_loss']:.4f}")

    val_loss = compute_val_loss(model, config)
    perplexity = math.exp(val_loss)
    print(f"\nVal loss (eval mode, {EVAL_ITERS} batches): {val_loss:.4f}")
    print(f"Val perplexity: {perplexity:.2f}")

    print(
        "\nOverfitting check: compare this val loss to the train loss logged at the "
        "end of training in train.py's printed output / loss_curve.png. If train loss "
        "keeps dropping while val loss flattens or rises, that's overfitting."
    )

    prompts = [
        "Once upon a time",
        "Tom and his dog went to",
        "The little girl found a",
        "One sunny morning, the rabbit",
    ]
    print("\nQualitative samples:")
    for p in prompts:
        text = generate_sample(model, config, p)
        print(f"\nPrompt: {p!r}\nGenerated: {text}")


if __name__ == "__main__":
    main()

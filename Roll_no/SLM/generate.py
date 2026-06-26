"""
Task 8: Inference — load the best checkpoint and generate short stories from a prompt.

Temperature controls randomness in sampling:
- Low temperature (e.g. 0.3): the model becomes more conservative/deterministic,
  favoring its highest-probability tokens. Output is more repetitive but coherent.
- High temperature (e.g. 1.2): the probability distribution flattens, so less-likely
  tokens get picked more often. Output is more "creative" but can become incoherent
  or nonsensical at extreme values.
"""

import os
import argparse
import torch
import tiktoken

from model import GPT, GPTConfig

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(OUT_DIR, "best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def generate(model, config, prompt, max_new_tokens=120, temperature=0.8, top_k=50):
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    args = parser.parse_args()

    model, config = load_model()

    for temp in (0.5, 1.0):
        text = generate(model, config, args.prompt, max_new_tokens=args.max_new_tokens, temperature=temp)
        print(f"\n=== Temperature {temp} ===")
        print(text)


if __name__ == "__main__":
    main()

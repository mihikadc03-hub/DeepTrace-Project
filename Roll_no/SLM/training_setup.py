"""
Task 5: Set up AdamW optimizer + CosineAnnealingLR scheduler.

AdamW vs Adam: AdamW decouples weight decay from the gradient-based update
(standard Adam folds L2 regularization into the gradient itself, which interacts
oddly with Adam's adaptive learning rates). Weight decay shrinks the weights a
little on every step, which helps prevent overfitting.

Cosine annealing: the learning rate starts high and follows a cosine curve down
to (near) zero over training. Early on, a higher LR lets the model take big steps
and explore; lowering it later lets it settle into a good minimum more precisely.
"""

import torch
from model import GPT, GPTConfig


def build_optimizer_and_scheduler(model, lr=3e-4, weight_decay=0.1, max_iters=2000):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    return optimizer, scheduler


if __name__ == "__main__":
    config = GPTConfig(vocab_size=50257, block_size=64, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=3e-4, max_iters=2000)

    print("Initial learning rate:", optimizer.param_groups[0]["lr"])

    # dummy step
    dummy = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(dummy, targets=dummy)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    print("Learning rate after one scheduler step:", optimizer.param_groups[0]["lr"])

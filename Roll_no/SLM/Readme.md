# Small Language Model (SLM)
Small Language Model (SLM) on TinyStories — full project
This implements all 8 tasks from the project doc: a small GPT-style transformer
trained from scratch on the TinyStories dataset, including data prep, model,
training, evaluation, and inference.
Important: network access
This sandbox's network is locked to a short allow-list (pypi, npm, github, etc.)
and cannot reach huggingface.co or openaipublic.blob.core.windows.net, which
`datasets` and `tiktoken` need to download TinyStories and the GPT-2 BPE files.
So `prepare_data.py` could not actually be run here — but every other script
(`model.py`, `training_setup.py`, `train.py`, `evaluate.py`, `generate.py`) was
tested end-to-end with synthetic token data and confirmed to work correctly.
Run this project on your own machine (or Colab) where those domains are reachable.
Setup
```bash
pip install torch tiktoken datasets numpy matplotlib
```
Run order (matches the 8 tasks)
Task 1–2 — data prep (downloads TinyStories, tokenizes, saves train.bin/val.bin):
```bash
   python prepare_data.py
   ```
Task 3 — data loader check:
```bash
   python data_loader.py
   ```
Task 4 — model architecture check:
```bash
   python model.py
   ```
Task 5 — optimizer/scheduler check:
```bash
   python training_setup.py
   ```
Task 6 — pre-train:
```bash
   python train.py
   ```
This trains the model, prints train/val loss every `EVAL_INTERVAL` iterations,
saves the best checkpoint to `best_model.pt`, and writes `loss_curve.png`.
Edit the config constants at the top of `train.py` (model size, `MAX_ITERS`,
batch size) to match your hardware — the defaults (4 layers, 128 dim, 2000
iters) are CPU-friendly but will give a weak model; bump them up if you have
a GPU.
Task 7 — evaluate:
```bash
   python evaluate.py
   ```
Prints val loss, perplexity (`exp(val_loss)`), and a few generated samples
for a quick coherence check.
Task 8 — generate text:
```bash
   python generate.py --prompt "Once upon a time"
   ```
Generates text at two different temperatures so you can compare how
conservative vs. creative the sampling looks.
Files
File	Purpose
`prepare_data.py`	Load TinyStories, tokenize with tiktoken, save `train.bin`/`val.bin`
`data_loader.py`	`get_batch()` — random (x, y) batches, y = x shifted right by 1
`model.py`	`GPTConfig` + `GPT` — embeddings, causal self-attention, feed-forward, generate()
`training_setup.py`	AdamW optimizer + cosine annealing LR scheduler
`train.py`	Main training loop, `estimate_loss()`, checkpointing, loss curve plot
`evaluate.py`	Perplexity, overfitting check, qualitative sample review
`generate.py`	Load checkpoint, generate text at different temperatures
Notes on the design choices
Tokenizer: GPT-2 BPE via `tiktoken` (vocab size 50257), matching nanoGPT.
Token storage: `uint16` memmap files, since 50257 fits comfortably in 16 bits.
Model: standard decoder-only transformer (token emb + learned positional emb,
pre-LN transformer blocks, causal masked self-attention, GELU feed-forward).
Defaults are deliberately small (4 layers, 4 heads, 128-dim, 128 context) so a
full run is feasible on a laptop CPU in a reasonable time; scale up for better
output quality if you have GPU access.

Build a **Small Language Model (SLM)** from scratch and gain a practical understanding of how modern language models function internally. This project is designed to help participants move beyond simply using pretrained models and instead understand the fundamental components that power text generation systems.

Participants will explore the complete process of building a lightweight transformer-based language model — beginning with text preprocessing and tokenization, progressing through embeddings and attention mechanisms, and eventually training a model capable of generating coherent text.

The goal of this project is not to create a production-scale LLM, but to develop an intuitive and implementation-level understanding of:
- How language models learn patterns from text
- Tokenization and vocabulary construction
- Embeddings and positional encodings
- Self-attention and transformer architecture
- Training loops, optimization, and loss computation
- Text generation and inference strategies

Through experimentation and guided implementation, participants will understand how small architectural decisions influence model behavior and performance. By the end of the project, participants should have a working small language model along with a solid conceptual foundation for exploring larger transformer systems and advanced NLP research.

## What You’ll Learn
- Text preprocessing and tokenization
- Embeddings and positional encoding
- Attention mechanisms and transformers
- Training language models on text data
- Loss functions and optimization
- Text generation and inference

## Project Flow

### Task 1 — Understand Language Models
Learn how language models work and explore the basics of autoregressive text generation.

### Task 2 — Prepare the Dataset
Clean, preprocess, and tokenize text data for training.

### Task 3 — Build the Model
Implement the architecture components required for a small transformer-based language model.

### Task 4 — Train the Model
Write the training pipeline, define the loss function, and optimize model parameters.

### Task 5 — Evaluate & Generate Text
Test the model and generate text samples to observe learned behavior.

### Task 6 — Improve & Experiment
Experiment with architecture changes, hyperparameters, and dataset improvements.

## Tech Stack
- Python
- PyTorch
- Tokenizers / Text Processing Libraries

## Goal

By the end of this project, participants should have both a working **Small Language Model (SLM)** and a strong conceptual understanding of the building blocks behind modern language models. The focus is not on training a massive production-scale model, but on understanding the complete pipeline involved in language modeling — including preprocessing text, building transformer components, training neural networks, and generating text outputs.

Participants should leave the project with enough practical and theoretical understanding to confidently explore larger transformer architectures, advanced NLP concepts, and future work involving language models, fine-tuning, or generative AI systems.

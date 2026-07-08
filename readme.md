# Transformers From Scratch

Transformer architecture implemented from scratch in Python, without using high-level abstractions like `nn.Transformer`. Weights are randomly initialized and **not trained** — this is an inference-only walkthrough of the architecture, built with NumPy for the core math and PyTorch only for the feed-forward layers and top-k sampling.

---

## What I built

Each component is implemented independently so you can see exactly how the pieces connect:

- **Token embeddings** — maps input token IDs to dense vectors of dimension `d_model`, via a randomly initialized embedding matrix
- **Positional encoding** — sinusoidal encoding added to embeddings to inject sequence position
- **Scaled dot-product attention** — computes `softmax(QKᵀ / √d_k) · V` with optional causal masking
- **Multi-head self-attention** — splits Q/K/V across `h` heads, attends in parallel, then concatenates and projects with `WO`
- **Position-wise feed-forward network** — two linear layers with ReLU, applied identically to each position (implemented in PyTorch)
- **Layer normalization** — custom implementation with learnable `gamma`/`beta`
- **Encoder block** — self-attention → add & norm → FFN → add & norm, stacked `6` times
- **Full encoder stack** — embeddings + positional encoding fed through 6 encoder blocks
- **Decoder block** — masked self-attention → cross-attention over encoder output → FFN, stacked `6` times
- **Autoregressive inference** — greedy/top-k token-by-token generation using the BPE tokenizer from `tiktoken`

---

## Design decisions & what I learned

The trickiest part was keeping the NumPy and PyTorch boundaries consistent: attention and layer norm are plain NumPy, while the feed-forward network and final output projection are PyTorch `nn.Module`s, so tensors get converted back and forth at each block. That mix makes the math easy to read step-by-step but isn't the most efficient way to structure things — a good candidate for cleanup if this gets extended into a trainable model.

---

## Known limitations

- **No batching** — the implementation works on a single sequence at a time (2D arrays), not `(batch, seq_len, d_model)` tensors.
- **No training** — all weights are randomly initialized with `np.random.randn`, so generated text is expected to be incoherent. This project demonstrates the architecture's data flow, not a working language model.
- **Add & norm in the decoder is currently disabled** — the three layer-norm calls in the decoder block are present but commented out, so the decoder currently runs without normalization between sub-layers.
- Masking is applied with an explicit nested loop rather than a vectorized/broadcast mask, which is simple to read but slower than it needs to be.

---

## How to run

```bash
git clone https://github.com/code-cerebrum0/Transformers-From-Scratch.git
cd Transformers-From-Scratch
pip install numpy torch tiktoken
jupyter notebook transformerarchitecture.ipynb
```

---

## Reference

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
# Transformers From Scratch

Transformer architecture implemented from scratch in Python, without using high-level abstractions like `nn.Transformer`.

---

## What I built

Each component is implemented independently so you can see exactly how the pieces connect:

- **Token embeddings** — maps input token IDs to dense vectors of dimension `d_model`
- **Positional encoding** — sinusoidal encoding added to embeddings to inject sequence position
- **Scaled dot-product attention** — computes `softmax(QKᵀ / √d_k) · V` with optional masking
- **Multi-head self-attention** — splits Q/K/V across `h` heads, attends in parallel, then concatenates
- **Position-wise feed-forward network** — two linear layers with ReLU, applied identically to each position
- **Encoder block** — self-attention → add & norm → FFN → add & norm, stacked `6` times
- **Full encoder stack** — embeddings + positional encoding fed through 6 encoder blocks
- **Decoder block** — masked self-attention → cross-attention over encoder output → FFN → add & norm, stacked `6` times

## Design decisions & what I learned


Getting attention mask shapes right for batched inputs was the trickiest part — the mask has to broadcast correctly across `(batch, heads, seq_len, seq_len)` without accidentally masking the wrong positions. Positional encoding uses fixed sinusoidal functions rather than learned embeddings, following the original paper — the motivation is that sinusoidal encoding generalizes to sequence lengths longer than those seen during training.

---

## How to run

```bash
git clone https://github.com/code-cerebrum0/Transformers-From-Scratch.git
cd Transformers-From-Scratch
pip install -r requirements.txt
jupyter notebook transformerarchitecture.ipynb
```


---

## Results

Verified encoder output shapes match expected `(batch_size, seq_len, d_model)` on synthetic input data. Attention weights sum to 1.0 across the key dimension as expected, confirmed by inspecting the softmax output before the value projection.

---

## Reference

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

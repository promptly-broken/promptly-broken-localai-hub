# Same FLOPs, Different Speeds

A benchmark showing that two neural network layers with **identical FLOP counts** can have wildly different wall-clock times — and why.

Inspired by [Barba & Cruz (2026), "FLOPs vs Real Work: The Importance of Replication in AI Efficiency Assessment"](https://arxiv.org/abs/2608.14550), which demonstrated this effect on an RTX 4090 for convolutional layers. This repo reproduces a simplified version of their experiment on Apple Silicon (MPS).

## The Experiment

Two PyTorch layers, each performing exactly **4.83 billion FLOPs** on a `1×3×128×128` input:

| Layer | Architecture | Weight Memory | Time (100 trials) |
|---|---|---|---|
| Dense | Linear 49,152 → 49,152 | 9.00 GB | 105 ms ± 0.6 ms |
| Convolution | Depthwise 3×3 + Pointwise 3 → 49,152 | 0.56 MB | 13 ms ± 0.2 ms |

**Same FLOPs. 8× difference.**

## Why

The dense layer is **memory-bound**. At batch size 1, its 9 GB weight matrix gets used once and discarded — the forward pass is entirely bottlenecked by memory bandwidth.

The convolution is **compute-bound**. It achieves the same FLOP count by reusing 0.56 MB of weights across 16,384 spatial positions. Those weights fit in cache, so the GPU stays busy doing math instead of waiting for data.

The paper finds a complementary effect on NVIDIA hardware: spatial dimensions are more easily parallelized than kernel dimensions, and newer GPUs actually introduce *more* instability (discrete jumps, oscillations) in the FLOPs-to-time relationship — making FLOPs an even less reliable proxy than before.

## Run It

```bash
pip install -r requirements.txt
python demo.py
```

Requires an Apple Silicon Mac with MPS support (falls back to CPU otherwise).

## Hardware

Tested on a MacBook Pro with Apple M-series chip and 48 GB unified memory.

## References

- Barba, E. & Cruz, L. (2026). *FLOPs vs Real Work: The Importance of Replication in AI Efficiency Assessment*. [arXiv:2608.14550](https://arxiv.org/abs/2608.14550)
- Asperti, A., Evangelista, D. & Marzolla, M. (2022). *Dissecting FLOPs along input dimensions for GreenAI cost estimations*. (Original study replicated by Barba & Cruz)

## Files

- `demo.py` — Self-contained benchmark script. No external dependencies beyond PyTorch.
- `requirements.txt` — Just `torch>=2.0`.

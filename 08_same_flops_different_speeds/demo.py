#!/usr/bin/env python3
"""
FLOPs vs Real Work: The Importance of Replication in AI Efficiency Assessment

This demo compares execution time of two neural network layers with identical FLOPs but different operation types.
It demonstrates that FLOPs alone don't tell the whole story about computational efficiency.

Usage:
    python demo.py

Requirements:
    pip install torch

Hardware: Apple M-series with 48GB unified memory (MPS)
"""

import torch
import torch.nn as nn
import time
from typing import Tuple, List

# --- Configuration ---
INPUT_SIZE = (1, 3, 128, 128)  # Batch size, channels, height, width
NUM_TRIALS = 100               # Number of timed inference trials
WARMUP_TRIALS = 10             # Warm-up iterations (discarded from stats)

# --- Core Logic ---

class DenseLayer(nn.Module):
    """Dense (fully connected) layer for benchmarking."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DepthwiseConvLayer(nn.Module):
    """Depthwise separable convolution layer for benchmarking."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            groups=in_channels, padding=1, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate FLOPs for a given model with specified input shape.

    Uses the input_shape to derive spatial dimensions rather than hardcoding.
    With bias=False on all layers, FLOPs = 2 * MACs exactly.

    Args:
        model: PyTorch model to analyze
        input_shape: Input tensor shape (batch, channels, height, width)

    Returns:
        Total number of FLOPs
    """
    _, _, H, W = input_shape
    spatial = H * W

    flops = 0
    if isinstance(model, DenseLayer):
        in_features = model.fc.in_features
        out_features = model.fc.out_features
        # 1 MAC = 2 FLOPs (multiply + accumulate)
        flops = 2 * in_features * out_features
    elif isinstance(model, DepthwiseConvLayer):
        in_channels = model.depthwise.in_channels
        out_channels = model.pointwise.out_channels
        k = model.depthwise.kernel_size[0]
        # Depthwise: k*k kernel per channel over H*W spatial positions
        flops_depthwise = 2 * in_channels * (k * k) * spatial
        # Pointwise: 1x1 conv = in_channels multiplies per output pixel
        flops_pointwise = 2 * in_channels * out_channels * spatial
        flops = flops_depthwise + flops_pointwise

    return flops


def benchmark_layer(
    layer: nn.Module,
    input_shape: Tuple[int, ...],
    num_trials: int,
    warmup_trials: int,
) -> Tuple[List[float], List[float]]:
    """
    Benchmark a layer's execution time and memory usage.

    Runs warmup_trials iterations with a full MPS sync before collecting
    num_trials timed measurements. Each trial syncs MPS before stopping
    the timer to ensure GPU work is complete.

    Args:
        layer: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_trials: Number of timed trials to run
        warmup_trials: Number of warm-up trials (discarded)

    Returns:
        Tuple of (execution_times_ms, memory_usages_mb)
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    layer = layer.to(device)

    execution_times = []
    memory_usages = []

    # Warm up: run several passes and sync so GPU caches/pipelines are primed
    for _ in range(warmup_trials):
        dummy_input = torch.randn(input_shape, device=device)
        with torch.no_grad():
            _ = layer(dummy_input)
    if device.type == 'mps':
        torch.mps.synchronize()

    # Timed trials
    for _ in range(num_trials):
        dummy_input = torch.randn(input_shape, device=device)

        start_time = time.perf_counter()
        with torch.no_grad():
            _ = layer(dummy_input)
            if device.type == 'mps':
                torch.mps.synchronize()
        end_time = time.perf_counter()

        execution_times.append((end_time - start_time) * 1000)  # ms

        if device.type == 'mps':
            memory_usages.append(torch.mps.current_allocated_memory() / (1024 * 1024))  # MB
        else:
            memory_usages.append(0)

    return execution_times, memory_usages


def main():
    """Main benchmarking function."""
    print("FLOPs vs Real Work: The Importance of Replication in AI Efficiency Assessment")
    print("=============================================================================")

    # Check for MPS availability
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type != 'mps':
        print("Warning: MPS not available. Using CPU instead.")

    # Create layers with matched FLOP counts (bias=False for exact math)
    print("\nCreating benchmark layers...")

    in_features = 128 * 128 * 3  # 49,152

    # Dense layer: 49,152 → 49,152 = 2 * 49152² ≈ 4.83 GFLOPs
    dense_layer = DenseLayer(in_features, in_features)

    # Depthwise conv: 3-channel depthwise 3×3 + pointwise 3 → 49,152
    # Depthwise FLOPs  = 2 * 3 * 9 * 16384 =       884,736
    # Pointwise FLOPs  = 2 * 3 * 49152 * 16384 = 4,831,838,208
    # Total                                     ≈ 4.833 GFLOPs
    conv_layer = DepthwiseConvLayer(3, in_features, kernel_size=3)

    print("\nCalculating FLOPs for both layers...")

    dense_flops = calculate_flops(dense_layer, INPUT_SIZE)
    conv_flops = calculate_flops(conv_layer, INPUT_SIZE)

    print(f"Dense layer FLOPs: {dense_flops:,}")
    print(f"Conv layer FLOPs:  {conv_flops:,}")

    if dense_flops > 0 and abs(dense_flops - conv_flops) / dense_flops < 0.1:
        print("FLOP counts are approximately equal (within 10%)")
    else:
        print("Warning: FLOP counts differ significantly")

    # Weight memory footprint
    dense_weight_bytes = in_features * in_features * 4  # float32
    conv_dw_params = 3 * 9         # depthwise: 3 channels × 3×3 kernel
    conv_pw_params = 3 * in_features  # pointwise: 3 × 49,152
    conv_weight_bytes = (conv_dw_params + conv_pw_params) * 4

    print(f"\nDense weight memory:  {dense_weight_bytes / (1024**3):.2f} GB")
    print(f"Conv weight memory:   {conv_weight_bytes / (1024**2):.2f} MB")

    # Output activation size for conv (both layers produce same FLOP count,
    # but conv writes a large spatial output tensor)
    _, C_in, H, W = INPUT_SIZE
    conv_output_bytes = in_features * H * W * 4  # 49,152 channels × 128 × 128 × 4 bytes
    print(f"Conv output activation: {conv_output_bytes / (1024**3):.2f} GB")

    # Benchmark both layers
    print(f"\nRunning benchmarks ({WARMUP_TRIALS} warm-up + {NUM_TRIALS} timed trials)...")

    dense_times, dense_memory = benchmark_layer(dense_layer, INPUT_SIZE, NUM_TRIALS, WARMUP_TRIALS)
    conv_times, conv_memory = benchmark_layer(conv_layer, INPUT_SIZE, NUM_TRIALS, WARMUP_TRIALS)

    # Statistics
    def mean(lst): return sum(lst) / len(lst) if lst else 0
    def std(lst):
        m = mean(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5 if lst else 0

    dense_mean = mean(dense_times)
    dense_std = std(dense_times)
    conv_mean = mean(conv_times)
    conv_std = std(conv_times)

    print(f"\nDense layer - Mean time: {dense_mean:.3f} ms ± {dense_std:.3f} ms")
    print(f"Conv layer  - Mean time: {conv_mean:.3f} ms ± {conv_std:.3f} ms")

    if conv_mean > 0:
        speedup = dense_mean / conv_mean
        print(f"Dense / Conv ratio: {speedup:.2f}x")
    else:
        print("Cannot calculate ratio — conv time is zero")




if __name__ == "__main__":
    main()
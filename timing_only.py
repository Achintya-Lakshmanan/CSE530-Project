#!/usr/bin/env python3
"""
Simple timing comparison: MLP vs KAN
"""

import torch
import torch.nn as nn
import time
from kan import KAN

INPUT_DIM = 64
OUTPUT_DIM = 64
BATCH = 16
DEVICE = 'cpu'

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

print("Creating models...")
mlp = SimpleMLP(INPUT_DIM, 128, OUTPUT_DIM).to(DEVICE)
kan = KAN([INPUT_DIM, 6, OUTPUT_DIM]).to(DEVICE)

mlp_params = sum(p.numel() for p in mlp.parameters())
kan_params = sum(p.numel() for p in kan.parameters())

print(f"\nMLP Parameters: {mlp_params:,}")
print(f"KAN Parameters: {kan_params:,}")

x = torch.randn(BATCH, INPUT_DIM, device=DEVICE)
y = torch.randn(BATCH, OUTPUT_DIM, device=DEVICE)
criterion = nn.MSELoss()

# MLP Timing
print("\nRunning MLP timing...")
mlp.train()
# Warmup
for _ in range(3):
    mlp.zero_grad()
    out = mlp(x)
    loss = criterion(out, y)
    loss.backward()

# Actual timing
times = []
for _ in range(10):
    start = time.perf_counter()
    mlp.zero_grad()
    out = mlp(x)
    loss = criterion(out, y)
    loss.backward()
    end = time.perf_counter()
    times.append((end - start) * 1000)

mlp_time = sum(times) / len(times)

# KAN Timing
print("Running KAN timing...")
kan.train()
# Warmup
for _ in range(3):
    kan.zero_grad()
    out = kan(x)
    loss = criterion(out, y)
    loss.backward()

# Actual timing
times = []
for _ in range(10):
    start = time.perf_counter()
    kan.zero_grad()
    out = kan(x)
    loss = criterion(out, y)
    loss.backward()
    end = time.perf_counter()
    times.append((end - start) * 1000)

kan_time = sum(times) / len(times)

# Results
print("\n" + "="*60)
print("TIMING RESULTS")
print("="*60)
print(f"MLP (Forward + Backward): {mlp_time:.3f} ms per iteration")
print(f"KAN (Forward + Backward): {kan_time:.3f} ms per iteration")
print(f"\nKAN / MLP Ratio: {kan_time/mlp_time:.2f}x")
print(f"KAN is {kan_time/mlp_time:.2f}x SLOWER than MLP")
print("="*60)

#!/usr/bin/env python3
"""
MLP vs KAN Performance Analysis
- Timing: Forward+Backward pass
- Cache Simulation: Using your Simulator
- Trace Generation: Synthetic memory access pattern
"""

import torch
import torch.nn as nn
import time
import subprocess
import sys
import os
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

def generate_trace(model, model_name, num_iterations=100):
    """
    TRACE GENERATION LOGIC:
    =======================
    
    Goal: Simulate memory accesses that happen during model execution
    
    Step 1: Iterate through each parameter tensor
        - MLP has: fc1.weight, fc1.bias, fc2.weight, fc2.bias (4 tensors)
        - KAN has: many small spline/basis tensors (22+ tensors)
    
    Step 2: For each parameter, simulate multiple access iterations
        - num_iterations=100 simulates ~100 forward/backward passes
        - Each iteration represents one epoch or batch of work
    
    Step 3: Generate memory addresses within each parameter
        - Stride by 16 elements = 64 bytes (standard cache line size for float32)
        - This groups accesses into cache-line-sized chunks
        - Simulates how CPU actually loads memory in blocks
        - Example: Parameter with 1000 elements -> 1000/16 = ~63 cache line accesses
    
    Step 4: Create trace line for each access
        Format: "<line_number> R <address>"
        - line_number: Sequential identifier (0, 1, 2, ...)
        - R: Read operation (all accesses are reads in forward/backward)
        - address: Memory address of the access
    
    Why this matters:
        - MLP: Few large tensors -> addresses are nearby -> cache locality
        - KAN: Many small tensors -> addresses are scattered -> poor cache locality
    
    Output:
        - File: "{model_name}_trace.txt"
        - Size: ~(sum of parameter sizes) / 16 * num_iterations lines
        - Example: 100K params, 100 iterations -> ~625K trace lines
    """
    trace_file = f"{model_name.lower()}_trace.txt"
    
    with open(trace_file, 'w') as f:
        line_num = 0
        
        for param_idx, param in enumerate(model.parameters()):
            param_base = id(param)      # Memory address of this parameter
            param_size = param.numel()  # Number of elements
            
            # For each iteration (simulating multiple forward/backward passes)
            for iteration in range(num_iterations):
                # Stride through parameter in 16-element chunks
                # (16 float32 elements = 64 bytes = one cache line)
                for elem_offset in range(0, param_size, 16):
                    # Compute the memory address for this access
                    addr = param_base + (elem_offset * 4) % (param_size * 4)
                    
                    # Write trace line
                    f.write(f"{line_num} R {addr}\n")
                    line_num += 1
    
    return trace_file

def run_cache_simulation(trace_file, model_name):
    """Run your Simulator on the generated trace"""
    config_path = os.path.abspath("Simulator/config/config_simple_multilevel")
    simulator_path = os.path.abspath("Simulator/src/cache_simulator.py")
    trace_path = os.path.abspath(trace_file)
    
    try:
        result = subprocess.run(
            [sys.executable, simulator_path, "-c", config_path, "-t", trace_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd()
        )
        
        # Extract only summary statistics - skip all trace detail lines
        output = result.stdout
        lines = output.split('\n')
        
        summary = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines, trace entries, and debug messages
            if not stripped:
                continue
            if 'hit_list' in line or 'Reading ' in line:
                continue
            if 'Memory hierarchy' in line or 'Loading' in line or 'Loaded' in line or 'Begin simulation' in line:
                continue
            if 'ERROR:' in line:
                continue
            # Keep only lines with actual stats
            if any(x in line for x in ['Number of instructions', 'Total cycles', 'cache_', 'Number of accesses', 'Number of hits', 'Number of misses', 'AMATs']):
                summary.append(line)
        
        return '\n'.join(summary), result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT: Cache simulation exceeded 120 seconds"
    except Exception as e:
        return "", f"ERROR: {str(e)}"

# ========================================================================
# MAIN
# ========================================================================

output = []

output.append("=" * 80)
output.append("MLP vs KAN: PERFORMANCE & CACHE ANALYSIS")
output.append("=" * 80)
output.append("")

# Create models
print("Creating models...")
mlp = SimpleMLP(INPUT_DIM, 128, OUTPUT_DIM).to(DEVICE)  # 64*128 + 128 + 128*64 + 64 = 16,576 params
kan = KAN([INPUT_DIM, 6, OUTPUT_DIM]).to(DEVICE)  # Adjusted to get closer to MLP params

mlp_params = sum(p.numel() for p in mlp.parameters())
kan_params = sum(p.numel() for p in kan.parameters())

output.append(f"Model Configuration:")
output.append(f"  Input Dimension:  {INPUT_DIM}")
output.append(f"  Output Dimension: {OUTPUT_DIM}")
output.append(f"  Batch Size:       {BATCH}")
output.append("")

output.append(f"MLP Parameters: {mlp_params:,}")
output.append(f"KAN Parameters: {kan_params:,}")
output.append("")

x = torch.randn(BATCH, INPUT_DIM, device=DEVICE)
y = torch.randn(BATCH, OUTPUT_DIM, device=DEVICE)
criterion = nn.MSELoss()

# ========================================================================
# TIMING: Complete Forward + Backward Pass
# ========================================================================
output.append("=" * 80)
output.append("TIMING RESULTS (Forward + Backward)")
output.append("=" * 80)
output.append("")

print("Running MLP timing...")
mlp.train()
# Warmup
for _ in range(3):
    mlp.zero_grad()
    out = mlp(x)
    loss = criterion(out, y)
    loss.backward()

# Timed runs
start = time.perf_counter()
for _ in range(10):
    mlp.zero_grad()
    out = mlp(x)
    loss = criterion(out, y)
    loss.backward()
mlp_time = (time.perf_counter() - start) / 10

output.append(f"MLP (Forward + Backward): {mlp_time*1000:.3f} ms per iteration")

print("Running KAN timing...")
kan.train()
# Warmup
for _ in range(3):
    kan.zero_grad()
    out = kan(x)
    loss = criterion(out, y)
    loss.backward()

# Timed runs
start = time.perf_counter()
for _ in range(10):
    kan.zero_grad()
    out = kan(x)
    loss = criterion(out, y)
    loss.backward()
kan_time = (time.perf_counter() - start) / 10

output.append(f"KAN (Forward + Backward): {kan_time*1000:.3f} ms per iteration")
output.append("")

output.append(f"Performance Comparison:")
output.append(f"  KAN / MLP Ratio: {kan_time/mlp_time:.2f}x")
output.append(f"  KAN is {kan_time/mlp_time:.2f}x SLOWER than MLP")
output.append("")

# ========================================================================
# CACHE SIMULATION
# ========================================================================
output.append("=" * 80)
output.append("CACHE SIMULATION")
output.append("=" * 80)
output.append("")

output.append("TRACE GENERATION EXPLANATION:")
output.append("-" * 80)
output.append("1. For each parameter tensor (weights, biases)")
output.append("2. Generate 100 iterations of memory accesses")
output.append("3. Stride by 16 elements (64-byte cache lines for float32)")
output.append("4. Output format: '<line_number> R <memory_address>'")
output.append("")
output.append("Why this approach:")
output.append("  - Mimics how CPUs load parameters into cache")
output.append("  - MLP: Few large tensors -> contiguous addresses -> good locality")
output.append("  - KAN: Many small tensors -> scattered addresses -> poor locality")
output.append("")

print("Generating memory traces...")
mlp_trace = generate_trace(mlp, "MLP", num_iterations=100)
kan_trace = generate_trace(kan, "KAN", num_iterations=100)

output.append(f"Generated Traces:")
output.append(f"  MLP: {mlp_trace}")
output.append(f"  KAN: {kan_trace}")
output.append("")

print("Running MLP cache simulation...")
mlp_out, mlp_err = run_cache_simulation(mlp_trace, "MLP")
output.append("MLP CACHE SIMULATION RESULTS:")
output.append("=" * 80)
if mlp_out:
    output.append(mlp_out)
else:
    output.append(f"ERROR: {mlp_err}")
output.append("")

print("Running KAN cache simulation...")
kan_out, kan_err = run_cache_simulation(kan_trace, "KAN")
output.append("KAN CACHE SIMULATION RESULTS:")
output.append("=" * 80)
if kan_out:
    output.append(kan_out)
else:
    output.append(f"ERROR: {kan_err}")
output.append("")

# ========================================================================
# SUMMARY
# ========================================================================
output.append("=" * 80)
output.append("SUMMARY & ANALYSIS")
output.append("=" * 80)
output.append("")

output.append(f"TIMING: KAN is {kan_time/mlp_time:.2f}x slower than MLP")
output.append("")
output.append("CACHE ANALYSIS:")
output.append("  Check the simulation results above for L1/L2/L3 hit/miss rates")
output.append("  Expected: KAN should have lower cache hit rates due to fragmentation")
output.append("")
output.append("INTERPRETATION:")
output.append("  - Higher hit rates = better cache locality = faster execution")
output.append("  - Lower hit rates = poor cache locality = more main memory accesses")
output.append("")

# ========================================================================
# WRITE RESULTS
# ========================================================================
results_text = "\n".join(output)
print("\n" + results_text)

with open("results.txt", 'w', encoding='utf-8') as f:
    f.write(results_text)

print("\n" + "=" * 80)
print("RESULTS SAVED TO: results.txt")
print("=" * 80)

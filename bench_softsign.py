#!/usr/bin/env python3
"""
bench_softsign_cpu.py

Benchmarks PyTorch softsign and (if available) an ONNX softsign model
using onnxruntime (CPU only).
Outputs times in microseconds (us).
"""

import os
import time
import numpy as np
import torch

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# Parameters: (C_i, H, W, k, s, padding, C_o) -- we only use C_i,H,W here
params = [
    (16, 48, 48, 1, 1, "PADDING_F", 16),
    (32, 24, 24, 1, 1, "PADDING_F", 32),

    (32, 48, 48, 1, 1, "PADDING_F", 32),
    (64, 24, 24, 1, 1, "PADDING_F", 64),
    (128, 12, 12, 1, 1, "PADDING_F", 128),

    (16, 48, 48, 1, 1, "PADDING_F", 32),
    (32, 24, 24, 1, 1, "PADDING_F", 64),
    (64, 12, 12, 1, 1, "PADDING_F", 128),
    (128, 6, 6, 1, 1, "PADDING_F", 256),

    (128, 24, 24, 1, 1, "PADDING_F", 128),
    (256, 12, 12, 1, 1, "PADDING_F", 256),

    (512, 12, 12, 1, 1, "PADDING_F", 512),
    (1024, 6, 6, 1, 1, "PADDING_F", 1024),

    (32, 208, 208, 1, 1, "PADDING_F", 64),
    (64, 104, 104, 1, 1, "PADDING_F", 128),
    (128, 52, 52, 1, 1, "PADDING_F", 256),
    (256, 26, 26, 1, 1, "PADDING_F", 512),
    (512, 13, 13, 1, 1, "PADDING_F", 1024),

    (16, 12, 12, 1, 1, "PADDING_F", 16),
    (32, 6, 6, 1, 1, "PADDING_F", 32),
    (16, 24, 24, 1, 1, "PADDING_F", 16),
]

num_threads_list = [1]
num_runs = 100

# utility timers in microseconds
def now_us():
    return time.perf_counter() * 1e9

# Torch model
class SoftSignModule(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.softsign(x)

# Export ONNX
def export_onnx_model(path="softsign_dynamic.onnx", opset=14):
    dummy = torch.randn(1, 3, 8, 8, dtype=torch.float32)
    model = SoftSignModule().eval()
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N", 1: "C", 2: "H", 3: "W"},
                      "output": {0: "N", 1: "C", 2: "H", 3: "W"}},
        opset_version=opset,
        do_constant_folding=True,
    )

# ONNXRuntime session on CPU
def make_ort_session(onnx_path, num_threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    sess = ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess

# PyTorch benchmark
print("\nPyTorch softsign (CPU, float32)")
print("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min(us)\tt_max(us)\tt_avg(us)")

device = torch.device("cpu")
torch.set_num_threads(1)
torch.set_grad_enabled(False)

for (C_i, H, W, k, s, pad, C_o) in params:
    rng = np.random.RandomState(12345 + C_i + H + W)
    input_np = rng.randn(1, C_i, H, W).astype(np.float32)
    input_t = torch.from_numpy(input_np).to(device)

    # warmup
    with torch.no_grad():
        _ = torch.nn.functional.softsign(input_t)

    tx, min_t, max_t = 0.0, float("inf"), 0.0
    for _ in range(num_runs):
        t0 = now_us()
        with torch.no_grad():
            out = torch.nn.functional.softsign(input_t)
            _ = out.cpu().numpy()  # ensure materialization
        t1 = now_us()
        dt = t1 - t0
        tx += dt
        min_t, max_t = min(min_t, dt), max(max_t, dt)

    avg = tx / num_runs
    print("\t{:<4}\t{:<3}\t{:<3}\t{:<1}\t{:<1}\t{:<4}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(
        C_i, H, W, k, s, 1, num_runs, min_t, max_t, avg
    ))

# ONNXRuntime benchmark
if ORT_AVAILABLE:
    print("\nONNXRuntime softsign (CPU, float32)")
    print("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min(us)\tt_max(us)\tt_avg(us)")

    onnx_path = "softsign_dynamic.onnx"
    export_onnx_model(onnx_path)

    for nth in num_threads_list:
        sess = make_ort_session(onnx_path, nth)
        inp_name = sess.get_inputs()[0].name

        for (C_i, H, W, k, s, pad, C_o) in params:
            rng = np.random.RandomState(54321 + C_i + H + W)
            input_np = rng.randn(1, C_i, H, W).astype(np.float32)

            # warmup
            _ = sess.run(None, {inp_name: input_np})

            tx, min_t, max_t = 0.0, float("inf"), 0.0
            for _ in range(num_runs):
                t0 = now_us()
                _ = sess.run(None, {inp_name: input_np})
                t1 = now_us()
                dt = t1 - t0
                tx += dt
                min_t, max_t = min(min_t, dt), max(max_t, dt)

            avg = tx / num_runs
            print("\t{:<4}\t{:<3}\t{:<3}\t{:<1}\t{:<1}\t{:<4}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(
                C_i, H, W, k, s, nth, num_runs, min_t, max_t, avg
            ))
else:
    print("\nonnxruntime not available -> skipped ONNX benchmark. Install with `pip install onnxruntime` to enable it.")

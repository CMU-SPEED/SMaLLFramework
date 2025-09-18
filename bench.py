#!/usr/bin/env python3
"""
bench_activations_convtranspose_cpu_to_csv.py

Benchmarks PyTorch and (if available) ONNX PReLU, CELU, Softmax, and ConvTranspose2d
(using ONNXRuntime CPU provider). Outputs times in nanoseconds (ns) and writes a CSV.

CSV columns:
  op,backend,C_i,H,W,k,s,pad,C_o,nthd,runs,t_min_ns,t_max_ns,t_avg_ns
"""

import os
import time
import csv
import numpy as np
import torch

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# Parameters: (C_i, H, W, k, s, padding, C_o)
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

num_runs = 100

# utility timers in nanoseconds
def now_ns():
    return time.perf_counter() * 1e9

device = torch.device("cpu")
torch.set_num_threads(1)
torch.set_grad_enabled(False)

#
# ONNX export helpers
#

def export_prelu_onnx(path, C, opset=14):
    class PReLUModule(torch.nn.Module):
        def __init__(self, num_parameters):
            super().__init__()
            self.prelu = torch.nn.PReLU(num_parameters=num_parameters)
        def forward(self, x):
            return self.prelu(x)
    model = PReLUModule(C).eval()
    dummy = torch.randn(1, C, 8, 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"},
                      "output": {0: "N", 2: "H", 3: "W"}},
        opset_version=opset,
        do_constant_folding=True,
    )

def export_celu_onnx(path, opset=14):
    class CELUModule(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.celu(x, alpha=1.0)
    model = CELUModule().eval()
    dummy = torch.randn(1, 3, 8, 8, dtype=torch.float32)
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

def export_softmax_onnx(path, dim=1, opset=14):
    class SoftmaxModule(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        def forward(self, x):
            return torch.nn.functional.softmax(x, dim=self.dim)
    model = SoftmaxModule(dim).eval()
    dummy = torch.randn(1, 3, 8, 8, dtype=torch.float32)
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

def export_convtranspose_onnx(path, C_i, C_o, kernel_size=3, stride=1, padding=0, opset=14):
    class ConvTransposeModule(torch.nn.Module):
        def __init__(self, C_i, C_o, k, s, p):
            super().__init__()
            self.weight = torch.randn(C_i, C_o, k, k)
            self.bias = None
            self.s = s
            self.p = p
        def forward(self, x):
            return torch.nn.functional.conv_transpose2d(x, self.weight, bias=self.bias, stride=self.s, padding=self.p)
    model = ConvTransposeModule(C_i, C_o, kernel_size, stride, padding).eval()
    dummy = torch.randn(1, C_i, 8, 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"},
                      "output": {0: "N", 2: "H", 3: "W"}},
        opset_version=opset,
        do_constant_folding=True,
    )

def make_ort_session(onnx_path):
    # default SessionOptions (no manual thread tuning)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess

#
# CSV writer setup
#
csv_path = "bench_results.csv"
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "op", "backend",
    "C_i", "H", "W", "k", "s", "pad", "C_o",
    "nthd", "runs", "t_min_ns", "t_max_ns", "t_avg_ns"
])

def write_csv_row(op, backend, C_i, H, W, k, s, pad, C_o, nthd, runs, tmin, tmax, tavg):
    csv_writer.writerow([op, backend, C_i, H, W, k, s, pad, C_o, nthd, runs, f"{tmin:.0f}", f"{tmax:.0f}", f"{tavg:.0f}"])
    csv_file.flush()

#
# Benchmarks
#

def bench_pytorch_prelu(params, num_runs):
    print("\nPyTorch PReLU (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(12345 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        m = torch.nn.PReLU(num_parameters=C_i).eval()
        with torch.no_grad():
            _ = m(input_t)
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            with torch.no_grad():
                out = m(input_t)
                _ = out.cpu().numpy()
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("PReLU", "PyTorch", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_prelu(params, num_runs):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX PReLU benchmark.")
        return
    print("\nONNXRuntime PReLU (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        onnx_path = f"prelu_C{C_i}_dyn.onnx"
        export_prelu_onnx(onnx_path, C_i)
        sess = make_ort_session(onnx_path)
        inp_name = sess.get_inputs()[0].name
        rng = np.random.RandomState(54321 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        _ = sess.run(None, {inp_name: input_np})
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            _ = sess.run(None, {inp_name: input_np})
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("PReLU", "ONNX", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_pytorch_celu(params, num_runs):
    print("\nPyTorch CELU (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(11111 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        with torch.no_grad():
            _ = torch.nn.functional.celu(input_t, alpha=1.0)
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            with torch.no_grad():
                out = torch.nn.functional.celu(input_t, alpha=1.0)
                _ = out.cpu().numpy()
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("CELU", "PyTorch", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_celu(params, num_runs):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX CELU benchmark.")
        return
    print("\nONNXRuntime CELU (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    onnx_path = "celu_dyn.onnx"
    export_celu_onnx(onnx_path)
    sess = make_ort_session(onnx_path)
    inp_name = sess.get_inputs()[0].name
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(22222 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        _ = sess.run(None, {inp_name: input_np})
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            _ = sess.run(None, {inp_name: input_np})
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("CELU", "ONNX", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_pytorch_softmax(params, num_runs, dim=1):
    print(f"\nPyTorch Softmax(dim={dim}) (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(33333 + C_i + H + W)
        input_np = rng.randn(1, C_i * H * W, 1, 1).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        with torch.no_grad():
            _ = torch.nn.functional.softmax(input_t, dim=dim)
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            with torch.no_grad():
                out = torch.nn.functional.softmax(input_t, dim=dim)
                _ = out.cpu().numpy()
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row(f"Softmax_dim{dim}", "PyTorch", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_softmax(params, num_runs, dim=1):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX Softmax benchmark.")
        return
    print(f"\nONNXRuntime Softmax(dim={dim}) (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    onnx_path = f"softmax_dim{dim}_dyn.onnx"
    export_softmax_onnx(onnx_path, dim=dim)
    sess = make_ort_session(onnx_path)
    inp_name = sess.get_inputs()[0].name
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(44444 + C_i + H + W)
        input_np = rng.randn(1, C_i * H * W, 1, 1).astype(np.float32)
        _ = sess.run(None, {inp_name: input_np})
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            _ = sess.run(None, {inp_name: input_np})
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row(f"Softmax_dim{dim}", "ONNX", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_pytorch_convtranspose(params, num_runs):
    print("\nPyTorch ConvTranspose2d (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\tk\ts\tC_o\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        kernel_size = k
        stride = s
        padding = 0
        rng = np.random.RandomState(55555 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        weight = torch.randn(C_i, C_o, kernel_size, kernel_size, dtype=torch.float32)
        with torch.no_grad():
            out = torch.nn.functional.conv_transpose2d(input_t, weight, bias=None, stride=stride, padding=padding)
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            with torch.no_grad():
                out = torch.nn.functional.conv_transpose2d(input_t, weight, bias=None, stride=stride, padding=padding)
                _ = out.cpu().numpy()
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<1}\t{:<1}\t{:<4}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(
            C_i, H, W, kernel_size, stride, C_o, num_runs, min_t, max_t, avg
        ))
        write_csv_row("ConvTranspose2d", "PyTorch", C_i, H, W, kernel_size, stride, padding, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_convtranspose(params, num_runs):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX ConvTranspose2d benchmark.")
        return
    print("\nONNXRuntime ConvTranspose2d (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\tk\ts\tC_o\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        kernel_size = k
        stride = s
        padding = 0
        onnx_path = f"convtranspose_Ci{C_i}_Co{C_o}_k{kernel_size}_dyn.onnx"
        export_convtranspose_onnx(onnx_path, C_i, C_o, kernel_size=kernel_size, stride=stride, padding=padding)
        sess = make_ort_session(onnx_path)
        inp_name = sess.get_inputs()[0].name
        rng = np.random.RandomState(66666 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        _ = sess.run(None, {inp_name: input_np})
        tx, min_t, max_t = 0.0, float("inf"), 0.0
        for _ in range(num_runs):
            t0 = now_ns()
            _ = sess.run(None, {inp_name: input_np})
            t1 = now_ns()
            dt = t1 - t0
            tx += dt
            min_t, max_t = min(min_t, dt), max(max_t, dt)
        avg = tx / num_runs
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<1}\t{:<1}\t{:<4}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(
            C_i, H, W, kernel_size, stride, C_o, num_runs, min_t, max_t, avg
        ))
        write_csv_row("ConvTranspose2d", "ONNX", C_i, H, W, kernel_size, stride, padding, C_o, 1, num_runs, min_t, max_t, avg)

if __name__ == "__main__":
    try:
        # PyTorch benches
        # bench_pytorch_prelu(params, num_runs)
        # bench_pytorch_celu(params, num_runs)
        bench_pytorch_softmax(params, num_runs, dim=1)
        # bench_pytorch_convtranspose(params, num_runs)

        # ONNX benches
        if ORT_AVAILABLE:
            # bench_onnx_prelu(params, num_runs)
            # bench_onnx_celu(params, num_runs)
            bench_onnx_softmax(params, num_runs, dim=1)
            # bench_onnx_convtranspose(params, num_runs)
        else:
            print("\nonnxruntime not available -> skipped ONNX benchmarks. Install with `pip install onnxruntime` to enable them.")
    finally:
        csv_file.close()
        print(f"\nWrote CSV results to: {csv_path}")

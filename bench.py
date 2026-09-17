#!/usr/bin/env python3
"""
bench_activations_convtranspose_cpu_to_csv.py

Benchmarks PyTorch and (if available) ONNX PReLU, CELU, EXP, whole-tensor Softmax, global Add, and
ConvTranspose2d (using ONNXRuntime CPU provider). Outputs times in nanoseconds
(ns) and writes a CSV.

CSV columns:
  op,backend,C_i,H,W,k,s,pad,C_o,nthd,runs,t_min_ns,t_max_ns,t_avg_ns
"""

import os
import time
import csv
import numpy as np
import torch
import sys


try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# Parameters: (C_i, H, W, k, s, padding, C_o)
params = [
    (16, 48, 48, 1, 1, "PADDING_V", 16),
    (32, 24, 24, 1, 1, "PADDING_V", 32),

    (32, 48, 48, 1, 1, "PADDING_V", 32),
    (64, 24, 24, 1, 1, "PADDING_V", 64),
    (128, 12, 12, 1, 1, "PADDING_V", 128),

    (16, 48, 48, 1, 1, "PADDING_V", 32),
    (32, 24, 24, 1, 1, "PADDING_V", 64),
    (64, 12, 12, 1, 1, "PADDING_V", 128),
    (128, 6, 6, 1, 1, "PADDING_V", 256),

    (128, 24, 24, 1, 1, "PADDING_V", 128),
    (256, 12, 12, 1, 1, "PADDING_V", 256),

    (512, 12, 12, 1, 1, "PADDING_V", 512),
    (1024, 6, 6, 1, 1, "PADDING_V", 1024),

    # (32, 208, 208, 1, 1, "PADDING_V", 64),
    # (64, 104, 104, 1, 1, "PADDING_V", 128),
    # (128, 52, 52, 1, 1, "PADDING_V", 256),
    # (256, 26, 26, 1, 1, "PADDING_V", 512),
    # (512, 13, 13, 1, 1, "PADDING_V", 1024),

    # (16, 12, 12, 1, 1, "PADDING_V", 16),
    # (32, 6, 6, 1, 1, "PADDING_V", 32),
    # (16, 24, 24, 1, 1, "PADDING_V", 16),
]

num_runs = 100

# utility timers in nanoseconds
def now_ns():
    return time.perf_counter() * 1e9

device = torch.device("cpu")
torch.set_num_threads(12)
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

def export_exp_onnx(path, opset=14):
    class ExpModule(torch.nn.Module):
        def forward(self, x):
            return torch.exp(x)
    model = ExpModule().eval()
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

def export_softmax_onnx(path, opset=14):
    class SoftmaxModule(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.softmax(x, dim=0)
    model = SoftmaxModule().eval()
    dummy = torch.randn(3 * 8 * 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "elements"},
                      "output": {0: "elements"}},
        opset_version=opset,
        do_constant_folding=True,
    )

def export_global_add_onnx(path, opset=14):
    class GlobalAddModule(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x)
    model = GlobalAddModule().eval()
    dummy = torch.randn(3 * 8 * 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "elements"}},
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
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 12
    session_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"], sess_options=session_options)
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

def time_pytorch_op(run_op, num_runs, num_trials=1000):
    with torch.no_grad():
        out = run_op()
        _ = out.cpu().numpy()

    times = []
    for _ in range(num_runs):
        t0 = now_ns()
        with torch.no_grad():
            for _ in range(num_trials):
                out = run_op()
                _ = out.cpu().numpy()
        t1 = now_ns()
        times.append((t1 - t0) / num_trials)

    min_t = min(times)
    max_t = max(times)
    if len(times) > 2:
        avg = (sum(times) - min_t - max_t) / (len(times) - 2)
    else:
        avg = sum(times) / len(times)
    return min_t, max_t, avg


def bench_pytorch_prelu(params, num_runs, num_trials=1000):
    print("\nPyTorch PReLU (CPU, float32) -- times in ns")
    print("	C_i	H	W	runs	t_min(ns)	t_max(ns)	t_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(12345 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        m = torch.nn.PReLU(num_parameters=C_i).eval()
        min_t, max_t, avg = time_pytorch_op(lambda: m(input_t), num_runs, num_trials)
        print("	{:<4}	{:<3}	{:<3}	{:<4}	{:.0f}		{:.0f}		{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
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

def bench_pytorch_celu(params, num_runs, num_trials=1000):
    print("\nPyTorch CELU (CPU, float32) -- times in ns")
    print("	C_i	H	W	runs	t_min(ns)	t_max(ns)	t_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(11111 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        min_t, max_t, avg = time_pytorch_op(
            lambda: torch.nn.functional.celu(input_t, alpha=1.0),
            num_runs,
            num_trials,
        )
        print("	{:<4}	{:<3}	{:<3}	{:<4}	{:.0f}		{:.0f}		{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
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

def bench_pytorch_exp(params, num_runs, num_trials=1000):
    print("\nPyTorch EXP (CPU, float32) -- times in ns")
    print("	C_i	H	W	runs	t_min(ns)	t_max(ns)	t_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(77777 + C_i + H + W)
        input_np = rng.uniform(2.0, 10.0, (1, C_i, H, W)).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        min_t, max_t, avg = time_pytorch_op(lambda: torch.exp(input_t), num_runs, num_trials)
        print("	{:<4}	{:<3}	{:<3}	{:<4}	{:.0f}		{:.0f}		{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("EXP", "PyTorch", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_exp(params, num_runs):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX EXP benchmark.")
        return
    print("\nONNXRuntime EXP (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    onnx_path = "exp_dyn.onnx"
    export_exp_onnx(onnx_path)
    sess = make_ort_session(onnx_path)
    inp_name = sess.get_inputs()[0].name
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(88888 + C_i + H + W)
        input_np = rng.uniform(2.0, 10.0, (1, C_i, H, W)).astype(np.float32)
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
        write_csv_row("EXP", "ONNX", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_pytorch_softmax(params, num_runs, num_trials=1000):
    print("\nPyTorch Softmax (whole tensor) (CPU, float32) -- times in ns")
    print("	C_i	H	W	runs	t_min(ns)	t_max(ns)	t_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(33333 + C_i + H + W)
        input_np = rng.randn(C_i * H * W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        min_t, max_t, avg = time_pytorch_op(
            lambda: torch.nn.functional.softmax(input_t, dim=0),
            num_runs,
            num_trials,
        )
        print("	{:<4}	{:<3}	{:<3}	{:<4}	{:.0f}		{:.0f}		{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("Softmax", "PyTorch", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_softmax(params, num_runs):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX Softmax benchmark.")
        return
    print("\nONNXRuntime Softmax (whole tensor) (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    onnx_path = "softmax_whole_tensor_dyn.onnx"
    export_softmax_onnx(onnx_path)
    sess = make_ort_session(onnx_path)
    inp_name = sess.get_inputs()[0].name
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(44444 + C_i + H + W)
        input_np = rng.randn(C_i * H * W).astype(np.float32)
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
        write_csv_row("Softmax", "ONNX", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_pytorch_global_add(params, num_runs, num_trials=1000):
    print("\nPyTorch Global Add (whole tensor to scalar) (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(55555 + C_i + H + W)
        input_np = rng.randn(C_i * H * W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        min_t, max_t, avg = time_pytorch_op(
            lambda: torch.sum(input_t), num_runs, num_trials
        )
        print("\t{:<4}\t{:<3}\t{:<3}\t{:<4}\t{:.0f}\t\t{:.0f}\t\t{:.0f}".format(C_i, H, W, num_runs, min_t, max_t, avg))
        write_csv_row("GlobalAdd", "PyTorch", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_onnx_global_add(params, num_runs):
    if not ORT_AVAILABLE:
        print("\nonnxruntime not available -> skipped ONNX Global Add benchmark.")
        return
    print("\nONNXRuntime Global Add (whole tensor to scalar) (CPU, float32) -- times in ns")
    print("\tC_i\tH\tW\truns\tt_min(ns)\tt_max(ns)\tt_avg(ns)")
    onnx_path = "global_add_whole_tensor_dyn.onnx"
    export_global_add_onnx(onnx_path)
    sess = make_ort_session(onnx_path)
    inp_name = sess.get_inputs()[0].name
    for (C_i, H, W, k, s, pad, C_o) in params:
        rng = np.random.RandomState(66666 + C_i + H + W)
        input_np = rng.randn(C_i * H * W).astype(np.float32)
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
        write_csv_row("GlobalAdd", "ONNX", C_i, H, W, k, s, pad, C_o, 1, num_runs, min_t, max_t, avg)

def bench_pytorch_convtranspose(params, num_runs, num_trials=1000):
    print("\nPyTorch ConvTranspose2d (CPU, float32) -- times in ns")
    print("	C_i	H	W	k	s	C_o	runs	t_min(ns)	t_max(ns)	t_avg(ns)")
    for (C_i, H, W, k, s, pad, C_o) in params:
        kernel_size = k
        stride = s
        padding = 0
        rng = np.random.RandomState(55555 + C_i + H + W)
        input_np = rng.randn(1, C_i, H, W).astype(np.float32)
        input_t = torch.from_numpy(input_np).to(device)
        weight = torch.randn(C_i, C_o, kernel_size, kernel_size, dtype=torch.float32)
        min_t, max_t, avg = time_pytorch_op(
            lambda: torch.nn.functional.conv_transpose2d(
                input_t,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
            ),
            num_runs,
            num_trials,
        )
        print("	{:<4}	{:<3}	{:<3}	{:<1}	{:<1}	{:<4}	{:<4}	{:.0f}		{:.0f}		{:.0f}".format(
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
    test_to_run = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    valid_tests = {"all", "prelu", "celu", "exp", "softmax", "global_add", "convtranspose", "convtranspose2d"}
    try:
        if test_to_run not in valid_tests:
            raise ValueError(f"Unknown test '{test_to_run}'. Expected one of: {', '.join(sorted(valid_tests))}")

        if test_to_run in ("all", "prelu"):
            bench_pytorch_prelu(params, num_runs)
        if test_to_run in ("all", "celu"):
            bench_pytorch_celu(params, num_runs)
        if test_to_run in ("all", "exp"):
            bench_pytorch_exp(params, num_runs)
        if test_to_run in ("all", "softmax"):
            bench_pytorch_softmax(params, num_runs)
        if test_to_run in ("all", "global_add"):
            bench_pytorch_global_add(params, num_runs)
        if test_to_run in ("all", "convtranspose", "convtranspose2d"):
            bench_pytorch_convtranspose(params, num_runs)

        # ONNX benches
        if ORT_AVAILABLE:
            if test_to_run in ("all", "prelu"):
                bench_onnx_prelu(params, num_runs)
            if test_to_run in ("all", "celu"):
                bench_onnx_celu(params, num_runs)
            if test_to_run in ("all", "exp"):
                bench_onnx_exp(params, num_runs)
            if test_to_run in ("all", "softmax"):
                bench_onnx_softmax(params, num_runs)
            if test_to_run in ("all", "global_add"):
                bench_onnx_global_add(params, num_runs)
            if test_to_run in ("all", "convtranspose", "convtranspose2d"):
                bench_onnx_convtranspose(params, num_runs)
        else:
            print("\nonnxruntime not available -> skipped ONNX benchmarks. Install with `pip install onnxruntime` to enable them.")
    finally:
        csv_file.close()
        print(f"\nWrote CSV results to: {csv_path}")

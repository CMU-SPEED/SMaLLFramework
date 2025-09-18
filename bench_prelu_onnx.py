#!/usr/bin/env python3
"""
Benchmark PReLU performance in ONNXRuntime (and optional PyTorch baseline).

Usage examples:
  # CPU ONNX
  python bench_prelu_onnx.py --batch 32 --C 64 --H 224 --W 224 --provider cpu --iters 200

  # GPU ONNX (if onnxruntime-gpu installed and CUDA available)
  python bench_prelu_onnx.py --batch 32 --C 64 --H 224 --W 224 --provider cuda --iters 200

  # Channel-wise PReLU (one alpha per channel)
  python bench_prelu_onnx.py --num_parameters channels

  # Shared PReLU (single alpha)
  python bench_prelu_onnx.py --num_parameters 1
"""
import argparse
import time
import statistics
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise SystemExit("Please install torch (pip install torch) to run this script.") from e

try:
    import onnx
    import onnxruntime as ort
except Exception as e:
    raise SystemExit("Please install onnx and onnxruntime (pip install onnx onnxruntime) to run this script.") from e


def build_torch_prelu(num_parameters: int, channels: int):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.prelu = nn.PReLU(num_parameters=num_parameters)
        def forward(self, x):
            return self.prelu(x)
    m = Model()
    # initialize alpha with something non-zero to exercise real compute
    with torch.no_grad():
        if num_parameters == 1:
            m.prelu.weight.fill_(0.25)
        else:
            # channel-wise: weight length should match channel dim
            with torch.no_grad():
                m.prelu.weight.copy_(torch.linspace(0.01, 0.5, num=channels))
    return m


def export_onnx(model, example_input, onnx_path, opset=13):
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=False,
        dynamic_axes=None,
    )
    # basic sanity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def make_session(onnx_path, provider_name="cpu"):
    providers = None
    provider_name = provider_name.lower()
    if provider_name == "cuda":
        # try GPU provider first, fallback to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    # tweak: enable sequential/parallel? leave default
    sess = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
    return sess


def time_session(session, input_name, input_array, warmup=20, iters=200):
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_array})

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_array})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return times


def time_torch(model, input_tensor, warmup=20, iters=200, device="cpu"):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    # warmup
    for _ in range(warmup):
        _ = model(input_tensor)
    times = []
    torch.cuda.synchronize() if device.startswith("cuda") and torch.cuda.is_available() else None
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(input_tensor)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def stats_from_times(times_ms, batch):
    avg = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    p90 = np.percentile(times_ms, 90)
    p99 = np.percentile(times_ms, 99)
    minimum = min(times_ms)
    maximum = max(times_ms)
    throughput_avg = 1000.0 * batch / avg  # images/sec
    throughput_med = 1000.0 * batch / med
    return {
        "mean_ms": avg, "median_ms": med, "p90_ms": p90, "p99_ms": p99,
        "min_ms": minimum, "max_ms": maximum,
        "throughput_mean_imgs_s": throughput_avg,
        "throughput_median_imgs_s": throughput_med,
    }


def human_print_stats(name, stats):
    print(f"--- {name} ---")
    print(f"mean latency     : {stats['mean_ms']:.3f} ms")
    print(f"median latency   : {stats['median_ms']:.3f} ms")
    print(f"p90 latency      : {stats['p90_ms']:.3f} ms")
    print(f"p99 latency      : {stats['p99_ms']:.3f} ms")
    print(f"min/max latency  : {stats['min_ms']:.3f}/{stats['max_ms']:.3f} ms")
    print(f"throughput (mean): {stats['throughput_mean_imgs_s']:.1f} images/sec")
    print(f"throughput (med) : {stats['throughput_median_imgs_s']:.1f} images/sec")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark PReLU in ONNXRuntime")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--C", type=int, default=64, help="number of channels")
    parser.add_argument("--H", type=int, default=224)
    parser.add_argument("--W", type=int, default=224)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu", help="onnxruntime provider")
    parser.add_argument("--num_parameters", choices=["1", "channels"], default="channels",
                        help="PReLU alpha shape: '1' (shared) or 'channels' (per-channel)")
    parser.add_argument("--iters", type=int, default=200, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--export_onnx", action="store_true", help="force re-export of ONNX even if exists")
    parser.add_argument("--pybench", action="store_true", help="also run PyTorch baseline")
    args = parser.parse_args()

    batch = args.batch
    C = args.C
    H = args.H
    W = args.W
    dtype = args.dtype
    provider = args.provider
    iters = args.iters
    warmup = args.warmup
    num_params = 1 if args.num_parameters == "1" else C

    model = build_torch_prelu(num_parameters=num_params, channels=C)
    model_name = f"prelu_B{batch}_C{C}_{'shared' if num_params==1 else 'channel'}.onnx"
    if not os.path.exists(model_name) or args.export_onnx:
        print("Exporting ONNX model:", model_name)
        example = torch.randn(batch, C, H, W, dtype=torch.float32)
        export_onnx(model, example, model_name, opset=13)
    else:
        print("Using cached ONNX model:", model_name)

    # Prepare input
    np_dtype = np.float16 if dtype == "fp16" else np.float32
    input_np = np.random.randn(batch, C, H, W).astype(np_dtype)

    # ONNX session
    print("Creating ONNX Runtime session (provider = %s)..." % provider)
    sess = make_session(model_name, provider_name=provider)
    input_name = sess.get_inputs()[0].name
    # convert if needed to float32 because some ORT GPU builds don't support fp16 inputs to CPU-bound ops
    if dtype == "fp16":
        # ONNXRuntime supports fp16 on GPU provider for many builds; try to use the dtype provided
        pass

    print("Warming up and timing ONNXRuntime...")
    onnx_times = time_session(sess, input_name, input_np, warmup=warmup, iters=iters)
    onnx_stats = stats_from_times(onnx_times, batch)
    human_print_stats("ONNXRuntime", onnx_stats)

    if args.pybench:
        dev = "cuda" if (provider == "cuda" and torch.cuda.is_available()) else "cpu"
        torch_input = torch.from_numpy(input_np.astype(np.float32))
        print(f"Running PyTorch baseline on device: {dev}")
        torch_times = time_torch(model, torch_input, warmup=warmup, iters=iters, device=dev)
        torch_stats = stats_from_times(torch_times, batch)
        human_print_stats("PyTorch", torch_stats)

    # print brief summary numbers
    print("Summary (mean latency ms):")
    print(f"ONNX mean: {onnx_stats['mean_ms']:.3f} ms, throughput: {onnx_stats['throughput_mean_imgs_s']:.1f} imgs/s")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare SMaLL global SoftMax with PyTorch and collect timing CSV data."""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOFTMAX_INTERFACE = REPO_ROOT / "test" / "run_softmax_interface.py"
DEFAULT_OUTPUT = REPO_ROOT / "test" / "softmax_pytorch_global_results.csv"

CASES = (
    (16, 1, 6),
    (32, 1, 18),
    (16, 30, 30),
    (96, 1, 6),
    (96, 30, 30),
    (16, 48, 48),
    (32, 24, 24),
    (32, 48, 48),
    (64, 12, 12),
    (64, 24, 24),
    (128, 6, 6),
    (128, 12, 12),
    (128, 24, 24),
    (256, 12, 12),
    (512, 12, 12),
    (1024, 6, 6),
    (16, 12, 12),
    (32, 6, 6),
    (16, 24, 24),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--runs-per-trial", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--init", choices=("linspace", "random", "zeros", "ones"),
                        default="linspace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-min", type=float, default=-4.0)
    parser.add_argument("--value-max", type=float, default=4.0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.trials < 3:
        raise ValueError("--trials must be at least 3 for trimmed statistics")
    if args.runs_per_trial <= 0 or args.warmup_runs < 0:
        raise ValueError("--runs-per-trial must be positive and warmups nonnegative")
    return args


def load_softmax_interface():
    spec = importlib.util.spec_from_file_location(
        "run_softmax_interface", SOFTMAX_INTERFACE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_interface_args(module, sweep_args: argparse.Namespace,
                        channels: int, height: int, width: int):
    class InterfaceArgs:
        pass

    args = InterfaceArgs()
    args.height = height
    args.width = width
    args.logical_channels = channels
    args.input_values = None
    args.init = sweep_args.init
    args.seed = sweep_args.seed
    args.value_min = sweep_args.value_min
    args.value_max = sweep_args.value_max
    args.rtol = sweep_args.rtol
    args.atol = sweep_args.atol
    args.platform_dir = None
    args.build_dir = module.DEFAULT_BUILD_DIR
    args.skip_pytorch = False
    args.rebuild = False
    args.cache_dir = module.DEFAULT_CACHE_DIR
    args.verbose = False
    return args


def bind_small_softmax(module, interface_args):
    library = ctypes.CDLL(str(module.ensure_small_ctypes_library(interface_args)))
    function = library.small_softmax_float
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    return function


def run_bound_small(function, args, input_array, output_array, error_buffer):
    status = function(
        args.logical_channels,
        args.height,
        args.width,
        input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(error_buffer, ctypes.c_char_p),
        len(error_buffer),
    )
    if status != 0:
        error_text = error_buffer.value.decode("utf-8", errors="replace")
        raise RuntimeError(error_text or "unknown SMaLL SoftMax error")


def trimmed_stats(times: list[float]) -> tuple[float, float, float, float]:
    min_time = min(times)
    max_time = max(times)
    min_index = times.index(min_time)
    max_index = times.index(max_time)
    trimmed = [
        trial_time
        for index, trial_time in enumerate(times)
        if index != min_index and index != max_index
    ]
    avg = sum(trimmed) / len(trimmed)
    std_dev = math.sqrt(sum((t - avg) * (t - avg) for t in trimmed)
                        / len(trimmed))
    return min_time, max_time, avg, std_dev


def timed_trials(callback, trials: int, runs_per_trial: int) -> list[float]:
    trial_times = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(runs_per_trial):
            callback()
        trial_times.append(time.perf_counter() - start)
    return trial_times


def main() -> int:
    sweep_args = parse_args()
    module = load_softmax_interface()
    rows = []

    for channels, height, width in CASES:
        interface_args = make_interface_args(
            module, sweep_args, channels, height, width)
        logical_input = module.make_input(interface_args)
        input_array = np.ascontiguousarray(
            logical_input.reshape(-1), dtype=np.float32)
        output_array = np.zeros_like(input_array)
        error_buffer = ctypes.create_string_buffer(4096)
        small_softmax = bind_small_softmax(module, interface_args)

        torch_input = torch.from_numpy(input_array).to(torch.float32)
        torch_exp_input = torch.exp(torch_input)

        def run_small():
            run_bound_small(small_softmax, interface_args, input_array,
                            output_array, error_buffer)

        def run_torch_softmax():
            nonlocal torch_reference_flat
            torch_reference_flat = torch.softmax(torch_input, dim=0).numpy()

        def run_torch_global_add():
            nonlocal torch_global_sum
            torch_global_sum = torch.sum(torch_exp_input)

        torch_reference_flat = None
        torch_global_sum = None
        for _ in range(sweep_args.warmup_runs):
            run_small()
            run_torch_softmax()
            run_torch_global_add()

        small_times = timed_trials(
            run_small, sweep_args.trials, sweep_args.runs_per_trial)
        torch_softmax_times = timed_trials(
            run_torch_softmax, sweep_args.trials, sweep_args.runs_per_trial)
        torch_global_add_times = timed_trials(
            run_torch_global_add, sweep_args.trials, sweep_args.runs_per_trial)

        actual = output_array.reshape(channels, height, width).copy()
        reference = torch_reference_flat.reshape(channels, height, width).copy()
        max_abs = float(np.max(np.abs(actual - reference)))
        allclose = bool(np.allclose(actual, reference,
                                    rtol=sweep_args.rtol,
                                    atol=sweep_args.atol))

        small_min, small_max, small_avg, small_std = trimmed_stats(small_times)
        torch_min, torch_max, torch_avg, torch_std = trimmed_stats(
            torch_softmax_times)
        add_min, add_max, add_avg, add_std = trimmed_stats(
            torch_global_add_times)

        row = {
            "channels": channels,
            "height": height,
            "width": width,
            "reference": "pytorch_global",
            "allclose": str(allclose).lower(),
            "max_abs_diff": f"{max_abs:.9g}",
            "warmup_runs": sweep_args.warmup_runs,
            "trials": sweep_args.trials,
            "runs_per_trial": sweep_args.runs_per_trial,
            "small_trial_min_s": f"{small_min:.9g}",
            "small_trial_max_s": f"{small_max:.9g}",
            "small_trial_avg_trim_s": f"{small_avg:.9g}",
            "small_trial_std_dev_trim_s": f"{small_std:.9g}",
            "pytorch_trial_min_s": f"{torch_min:.9g}",
            "pytorch_trial_max_s": f"{torch_max:.9g}",
            "pytorch_trial_avg_trim_s": f"{torch_avg:.9g}",
            "pytorch_trial_std_dev_trim_s": f"{torch_std:.9g}",
            "pytorch_global_add_trial_min_s": f"{add_min:.9g}",
            "pytorch_global_add_trial_max_s": f"{add_max:.9g}",
            "pytorch_global_add_trial_avg_trim_s": f"{add_avg:.9g}",
            "pytorch_global_add_trial_std_dev_trim_s": f"{add_std:.9g}",
        }
        print(row, flush=True)
        rows.append(row)

    sweep_args.output.parent.mkdir(parents=True, exist_ok=True)
    with sweep_args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {sweep_args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

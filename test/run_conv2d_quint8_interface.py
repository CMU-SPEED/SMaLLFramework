#!/usr/bin/env python3
"""Generate and run a parameterized QUInt8 Conv2D interface check.

The generated C++ harness calls small::Conv2D. Python computes the expected
uint8 output using the same signed-centered inputs/filters and quantization
steps used by the current QUInt8 microkernels.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLATFORM = REPO_ROOT / "include" / "small" / "platforms" / "quantized_arm7E"
TEST_CONV2D_CASES = [
    # C_i, H, W, k, s, padding, C_o. Mirrors test_conv2d.cpp regression data.
    (3, 3, 3, 1, 1, "valid", 16),
    (3, 3, 3, 3, 1, "valid", 16),
    (3, 1, 1, 1, 1, "full", 16),
    (3, 3, 3, 1, 1, "full", 16),
    (16, 1, 1, 1, 1, "valid", 16),
    (16, 1, 6, 1, 1, "valid", 16),
    (16, 3, 3, 3, 1, "valid", 16),
    (16, 3, 8, 3, 1, "valid", 16),
    (16, 30, 30, 3, 1, "valid", 16),
    (16, 1, 6, 1, 1, "valid", 96),
    (16, 3, 8, 3, 1, "valid", 96),
    (96, 1, 6, 1, 1, "valid", 16),
    (96, 3, 8, 3, 1, "valid", 16),
    (96, 30, 30, 1, 1, "valid", 96),
    (96, 30, 30, 3, 1, "valid", 96),
    (16, 3, 3, 3, 1, "full", 16),
    (16, 3, 3, 3, 2, "full", 16),
    (16, 3, 8, 3, 1, "full", 16),
    (16, 3, 8, 3, 1, "full", 96),
    (16, 3, 13, 3, 2, "full", 16),
    (16, 3, 13, 3, 2, "full", 96),
    (96, 3, 8, 3, 1, "full", 16),
    (96, 3, 13, 3, 2, "full", 16),
    (96, 30, 30, 3, 1, "full", 96),
    # (96, 30, 30, 3, 2, "full", 96), '''removed because quantized pytorch does not support asymmetric padding'''
]

PYTORCH_INPUT_SCALE = 1.0
PYTORCH_INPUT_ZERO_POINT = 128
PYTORCH_WEIGHT_SCALE = 1.0
PYTORCH_WEIGHT_ZERO_POINT = 0
PYTORCH_OUTPUT_SCALE = 255.0 / 2.0
PYTORCH_OUTPUT_ZERO_POINT = 0
DEFAULT_PYTORCH_OUTPUT_ATOL = 2.0 / PYTORCH_OUTPUT_SCALE
PYTORCH_COMPARISON_EPSILON = 1e-12


def vqrdmulh(a: int, b: int) -> int:
    prod = a * b
    nudge = (1 << 30) if prod >= 0 else (1 - (1 << 30))
    return int((prod + nudge) / (1 << 31))


def rndrshift(a: int, b: int) -> int:
    mask = (1 << b) - 1
    mod = a & mask
    one = ~0
    threshold = (mask >> 1) + (one & (~0 if a < 0 else 0))
    return (a >> b) + (one & (~0 if mod > threshold else 0))


def quantize(acc: int) -> int:
    # Defaults from QUInt8Buffer::quantized_init() for [-1, 1] uint8.
    val = rndrshift(vqrdmulh(acc, 1077952576), 6)
    return max(0, min(255, val))


def logical_input(index: int) -> int:
    # Deterministic mix of negative and positive values in [-100, 100].
    return ((index * 37 + 13) % 201) - 100


def logical_filter(index: int) -> int:
    # Deterministic mix of negative and positive values in [-100, 100].
    return ((index * 171 + 28) % 201) - 100


def to_uint8(logical: int) -> int:
    if not -128 <= logical <= 127:
        raise ValueError(f"logical value {logical} cannot be encoded as raw uint8 - 128")
    return logical + 128


def calc_padding(input_dim: int, kernel: int, stride: int) -> tuple[int, int]:
    if input_dim % stride == 0:
        padding = kernel - stride if kernel > stride else 0
    else:
        padding = kernel - (input_dim % stride) if kernel > (input_dim % stride) else 0
    front = padding // 2
    return front, padding - front


def suite_case_args(
    case: tuple[int, int, int, int, int, str, int],
    base_args: argparse.Namespace,
) -> argparse.Namespace:
    input_channels, height, width, kernel, stride, padding, output_channels = case
    if padding == "valid":
        pad_top = pad_bottom = pad_left = pad_right = 0
    elif padding == "full":
        pad_top, pad_bottom = calc_padding(height, kernel, stride)
        pad_left, pad_right = calc_padding(width, kernel, stride)
    else:
        raise ValueError(f"unsupported padding mode: {padding}")

    return argparse.Namespace(
        height=height,
        width=width,
        input_channels=input_channels,
        output_channels=output_channels,
        kernel=kernel,
        stride=stride,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        platform_dir=base_args.platform_dir,
        arena_size=base_args.arena_size,
        reference_source=base_args.reference_source,
        check_pytorch=base_args.check_pytorch,
        python_only=base_args.python_only,
        pytorch_atol=base_args.pytorch_atol,
        keep=base_args.keep,
    )


def output_dim(input_dim: int, kernel: int, stride: int, front_pad: int, back_pad: int) -> int:
    padded = input_dim + front_pad + back_pad
    if padded < kernel:
        raise ValueError(f"incompatible size: padded dim {padded} is smaller than kernel {kernel}")
    return 1 + (padded - kernel) // stride


def reference_accumulations(
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[int], int, int]:
    h_o = output_dim(args.height, args.kernel, args.stride, args.pad_top, args.pad_bottom)
    w_o = output_dim(args.width, args.kernel, args.stride, args.pad_left, args.pad_right)

    inputs = [logical_input(i) for i in range(args.input_channels * args.height * args.width)]
    filters = [
        logical_filter(i)
        for i in range(args.output_channels * args.input_channels * args.kernel * args.kernel)
    ]
    expected: list[int] = []

    for co in range(args.output_channels):
        for oh in range(h_o):
            for ow in range(w_o):
                acc = 0
                for ci in range(args.input_channels):
                    for kh in range(args.kernel):
                        ih = oh * args.stride + kh - args.pad_top
                        if ih < 0 or ih >= args.height:
                            continue
                        for kw in range(args.kernel):
                            iw = ow * args.stride + kw - args.pad_left
                            if iw < 0 or iw >= args.width:
                                continue
                            input_ix = ci * args.height * args.width + ih * args.width + iw
                            filter_ix = (
                                co * args.input_channels * args.kernel * args.kernel
                                + ci * args.kernel * args.kernel
                                + kh * args.kernel
                                + kw
                            )
                            acc += inputs[input_ix] * filters[filter_ix]
                expected.append(acc)

    return inputs, filters, expected, h_o, w_o


def reference(args: argparse.Namespace) -> tuple[list[int], list[int], list[int], int, int]:
    inputs, filters, accumulations, h_o, w_o = reference_accumulations(args)
    uint8_inputs = [to_uint8(v) for v in inputs]
    uint8_filters = [to_uint8(v) for v in filters]
    expected = [quantize(acc) for acc in accumulations]
    return uint8_inputs, uint8_filters, expected, h_o, w_o


def pytorch_quantized_outputs(
    args: argparse.Namespace,
    inputs: list[int],
    filters: list[int],
) -> list[int]:
    try:
        import torch
        import torch.ao.nn.quantized.functional as qF
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyTorch is required for --reference-source pytorch or --check-pytorch") from exc

    if args.pad_top != args.pad_bottom or args.pad_left != args.pad_right:
        raise ValueError(
            "PyTorch quantized conv2d reference currently requires symmetric padding"
        )

    input_tensor = torch.tensor(inputs, dtype=torch.float32).reshape(
        1, args.input_channels, args.height, args.width
    )
    q_input = torch.quantize_per_tensor(
        input_tensor,
        scale=PYTORCH_INPUT_SCALE,
        zero_point=PYTORCH_INPUT_ZERO_POINT,
        dtype=torch.quint8,
    )

    weight_tensor = torch.tensor(filters, dtype=torch.int8).reshape(
        args.output_channels, args.input_channels, args.kernel, args.kernel
    )
    q_weight = torch.quantize_per_tensor(
        weight_tensor.to(torch.float32),
        scale=PYTORCH_WEIGHT_SCALE,
        zero_point=PYTORCH_WEIGHT_ZERO_POINT,
        dtype=torch.qint8,
    )

    output = qF.conv2d(
        q_input,
        q_weight,
        bias=None,
        stride=(args.stride, args.stride),
        padding=(args.pad_top, args.pad_left),
        dilation=(1, 1),
        groups=1,
        scale=PYTORCH_OUTPUT_SCALE,
        zero_point=PYTORCH_OUTPUT_ZERO_POINT,
    )
    return [int(v) for v in output.int_repr().reshape(-1).tolist()]


def dequantize_outputs(values: list[int]) -> list[float]:
    return [(value - PYTORCH_OUTPUT_ZERO_POINT) / PYTORCH_OUTPUT_SCALE for value in values]


def check_pytorch_matches_manual(
    manual_quantized: list[int],
    pytorch_quantized: list[int],
    *,
    atol: float,
) -> None:
    if len(manual_quantized) != len(pytorch_quantized):
        raise AssertionError(
            f"reference length mismatch: manual={len(manual_quantized)} pytorch={len(pytorch_quantized)}"
        )

    manual_float = dequantize_outputs(manual_quantized)
    pytorch_float = dequantize_outputs(pytorch_quantized)

    max_abs_error = 0.0
    worst_index = -1
    for index, (manual, pytorch_value) in enumerate(zip(manual_float, pytorch_float)):
        abs_error = abs(manual - pytorch_value)
        if abs_error > max_abs_error:
            max_abs_error = abs_error
            worst_index = index
        if not math.isclose(manual, pytorch_value, rel_tol=0.0, abs_tol=atol + PYTORCH_COMPARISON_EPSILON):
            raise AssertionError(
                "PyTorch quantized Conv2D mismatch at index "
                f"{index}: manual_q={manual_quantized[index]} pytorch_q={pytorch_quantized[index]} "
                f"manual={manual} pytorch={pytorch_value} abs_error={abs_error} atol={atol}"
            )

    print(
        f"PyTorch quantized check passed: max_abs_error={max_abs_error} at output index {worst_index}",
        flush=True,
    )


def summarize_quantized_difference(
    manual_quantized: list[int],
    pytorch_quantized: list[int],
) -> str:
    if len(manual_quantized) != len(pytorch_quantized):
        return (
            "reference length mismatch: "
            f"manual={len(manual_quantized)} pytorch={len(pytorch_quantized)}"
        )

    worst_q_error = -1
    worst_float_error = -1.0
    worst_index = -1
    manual_float = dequantize_outputs(manual_quantized)
    pytorch_float = dequantize_outputs(pytorch_quantized)

    for index, (manual_q, pytorch_q, manual_f, pytorch_f) in enumerate(
        zip(manual_quantized, pytorch_quantized, manual_float, pytorch_float)
    ):
        q_error = abs(manual_q - pytorch_q)
        float_error = abs(manual_f - pytorch_f)
        if q_error > worst_q_error or (q_error == worst_q_error and float_error > worst_float_error):
            worst_q_error = q_error
            worst_float_error = float_error
            worst_index = index

    return (
        "Python SMaLL vs PyTorch quantized summary: "
        f"worst_q_error={worst_q_error} "
        f"worst_float_error={worst_float_error} "
        f"at output index {worst_index}"
    )


def cpp_list(values: list[int]) -> str:
    return ", ".join(str(v) for v in values)


def render_harness(args: argparse.Namespace, inputs: list[int], filters: list[int], expected: list[int], h_o: int, w_o: int) -> str:
    output_capacity = len(expected) * 4
    return f"""
#define PARALLEL 1

#include <cstdint>
#include <iostream>

#include <params.h>
#include <Buffer.hpp>
#include <intrinsics.h>

#include <small/buffers.hpp>
#include <small/interface_abstract.hpp>

int main()
{{
    using BufferT = small::QUInt8Buffer;

    constexpr uint32_t C_i = {args.input_channels};
    constexpr uint32_t C_o = {args.output_channels};
    constexpr uint32_t H = {args.height};
    constexpr uint32_t W = {args.width};
    constexpr uint32_t K = {args.kernel};
    constexpr uint32_t stride = {args.stride};
    constexpr uint8_t t_pad = {args.pad_top};
    constexpr uint8_t b_pad = {args.pad_bottom};
    constexpr uint8_t l_pad = {args.pad_left};
    constexpr uint8_t r_pad = {args.pad_right};
    constexpr uint32_t H_o = {h_o};
    constexpr uint32_t W_o = {w_o};
    constexpr uint32_t output_size = C_o * H_o * W_o;

    uint8_t const input_raw[] = {{{cpp_list(inputs)}}};
    uint8_t const filter_raw[] = {{{cpp_list(filters)}}};
    uint8_t const expected[] = {{{cpp_list(expected)}}};

    BufferT input(sizeof(input_raw) / sizeof(input_raw[0]));
    input.m_zero = -128;
    for (uint32_t ix = 0; ix < input.size(); ++ix) input[ix] = input_raw[ix];

    BufferT filter(sizeof(filter_raw) / sizeof(filter_raw[0]));
    filter.m_zero = -128;
    for (uint32_t ix = 0; ix < filter.size(); ++ix) filter[ix] = filter_raw[ix];

    BufferT packed_input(input.size());
    small::pack_buffer(input, small::INPUT, 1U, C_i, H, W,
                       BufferT::C_ib, BufferT::C_ob, packed_input);
    packed_input.m_zero = -128;

    BufferT packed_filter(filter.size());
    small::pack_buffer(filter, small::FILTER_CONV, C_o, C_i, K, K,
                       BufferT::C_ib, BufferT::C_ob, packed_filter);
    packed_filter.m_zero = -128;

    BufferT packed_output({output_capacity});
    small::Conv2D(K, K, stride, t_pad, b_pad, l_pad, r_pad,
                  C_o, C_i, H, W, packed_input, packed_filter, packed_output);

    BufferT output(output_size);
    small::unpack_buffer(packed_output, small::OUTPUT, 1U, C_o, H_o, W_o,
                         BufferT::C_ib, BufferT::C_ob, output);

    bool passing = true;
    for (uint32_t ix = 0; ix < output_size; ++ix)
    {{
        if (output[ix] != expected[ix])
        {{
            passing = false;
            std::cerr << "Mismatch at " << ix
                      << ": computed=" << static_cast<int>(output[ix])
                      << ", expected=" << static_cast<int>(expected[ix])
                      << std::endl;
        }}
    }}
    if (!passing) return 1;

    std::cout << "PASS Conv2D QUInt8 "
              << "Ci=" << C_i << " Co=" << C_o
              << " H=" << H << " W=" << W
              << " K=" << K << " stride=" << stride
              << " output=" << H_o << "x" << W_o << std::endl;
    return 0;
}}
"""


def validate_args(args: argparse.Namespace) -> None:
    for name in ("height", "width", "input_channels", "output_channels", "kernel"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("pad_top", "pad_bottom", "pad_left", "pad_right"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if not args.platform_dir.exists():
        raise ValueError(f"platform dir not found: {args.platform_dir}")
    if args.arena_size <= 0:
        raise ValueError("arena-size must be positive")


def run_case(args: argparse.Namespace, label: str | None = None) -> None:
    validate_args(args)
    if label:
        print(label, flush=True)

    logical_inputs, logical_filters, manual_accumulations, h_o, w_o = reference_accumulations(args)

    manual_quantized = [quantize(acc) for acc in manual_accumulations]

    pytorch_values: list[int] | None = None
    if args.reference_source == "pytorch" or args.check_pytorch:
        pytorch_values = pytorch_quantized_outputs(args, logical_inputs, logical_filters)

    if args.check_pytorch:
        assert pytorch_values is not None
        check_pytorch_matches_manual(
            manual_quantized,
            pytorch_values,
            atol=args.pytorch_atol,
        )
        if args.python_only:
            print(summarize_quantized_difference(manual_quantized, pytorch_values), flush=True)

    inputs = [to_uint8(v) for v in logical_inputs]
    filters = [to_uint8(v) for v in logical_filters]
    if args.reference_source == "pytorch":
        assert pytorch_values is not None
        expected = pytorch_values
    else:
        expected = manual_quantized

    if args.python_only:
        if not args.check_pytorch:
            print("Python-only mode completed using the SMaLL Python reference.", flush=True)
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="small_conv2d_quint8_"))
    try:
        source = temp_dir / "conv2d_quint8_harness.cpp"
        exe = temp_dir / "conv2d_quint8_harness.exe"
        source.write_text(render_harness(args, inputs, filters, expected, h_o, w_o))

        compile_cmd = [
            "g++",
            "-std=c++17",
            "-Wall",
            "-fopenmp",
            "-O0",
            "-g",
            "-fpermissive",
            "-DQUANTIZED",
            f"-DMAX_BUFF_SIZE={args.arena_size}",
            f"-I{REPO_ROOT / 'include'}",
            f"-I{args.platform_dir}",
            str(source),
            "-o",
            str(exe),
        ]
        subprocess.run(compile_cmd, check=True)
        subprocess.run([str(exe)], check=True)
        if args.keep:
            print(f"Kept generated harness in {temp_dir}")
    finally:
        if not args.keep:
            shutil.rmtree(temp_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("single", "test-conv2d"),
        default="single",
        help="run one custom case or the regression case list from test_conv2d.exe",
    )
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--output-channels", type=int, default=2)
    parser.add_argument("--kernel", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1, choices=(1, 2))
    parser.add_argument("--pad-top", type=int, default=0)
    parser.add_argument("--pad-bottom", type=int, default=0)
    parser.add_argument("--pad-left", type=int, default=0)
    parser.add_argument("--pad-right", type=int, default=0)
    parser.add_argument("--platform-dir", type=Path, default=DEFAULT_PLATFORM)
    parser.add_argument(
        "--arena-size",
        type=int,
        default=8 * 1024 * 1024,
        help="override the quantized_arm7E static buffer arena size for host-side generated tests",
    )
    parser.add_argument(
        "--reference-source",
        choices=("small", "pytorch"),
        default="small",
        help="choose whether the quantized expected output comes from SMaLL's hand reference or PyTorch Conv2D",
    )
    parser.add_argument(
        "--check-pytorch",
        action="store_true",
        help="verify that the Python SMaLL quantized reference matches PyTorch quantized conv2d",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="run only the Python reference checks and skip compiling/running the C++ harness",
    )
    parser.add_argument(
        "--pytorch-atol",
        type=float,
        default=DEFAULT_PYTORCH_OUTPUT_ATOL,
        help="absolute tolerance for dequantized output comparison against PyTorch quantized conv2d"    )
    parser.add_argument("--keep", action="store_true", help="keep the generated C++ harness directory")
    args = parser.parse_args()

    if args.suite == "test-conv2d":
        for case_num, case in enumerate(TEST_CONV2D_CASES, start=1):
            case_args = suite_case_args(case, args)
            padding = case[5]
            label = (
                f"CASE {case_num}/{len(TEST_CONV2D_CASES)} "
                f"Ci={case_args.input_channels} Co={case_args.output_channels} "
                f"H={case_args.height} W={case_args.width} "
                f"K={case_args.kernel} stride={case_args.stride} padding={padding}"
            )
            run_case(case_args, label)
        print(f"PASS {len(TEST_CONV2D_CASES)} Conv2D QUInt8 cases from test_conv2d.exe")
    else:
        run_case(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

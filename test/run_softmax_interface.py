#!/usr/bin/env python3
"""Run SMaLL SoftMax through a cached ctypes shim and compare with PyTorch fp32."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLATFORM = REPO_ROOT / "include" / "small" / "platforms" / "zen2"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_CACHE_DIR = REPO_ROOT / "test" / ".softmax_debug_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--logical-channels", type=int, default=31)
    parser.add_argument(
        "--input-values",
        type=str,
        default=None,
        help=(
            "comma-separated logical input values in NCHW order with length "
            "logical_channels*height*width"
        ),
    )
    parser.add_argument(
        "--init",
        choices=("random", "linspace", "zeros", "ones"),
        default="random",
        help="how to initialize the logical fp32 input when --input-values is not given",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-min", type=float, default=-4.0)
    parser.add_argument("--value-max", type=float, default=4.0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--platform-dir",
        type=Path,
        default=None,
        help="directory containing params.h, Buffer.hpp, and intrinsics.h",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="existing CMake build directory to mine for uarch/platform flags",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="skip the PyTorch reference check and only print SMaLL output",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force recompiling the cached ctypes shim",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="directory used to cache the compiled ctypes shim",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print input, SMaLL output, and PyTorch reference arrays",
    )
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0 or args.logical_channels <= 0:
        raise ValueError("height, width, and logical-channels must be positive")
    return args


def detect_build_uarch(build_dir: Path) -> str | None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return None

    for line in cache_path.read_text().splitlines():
        if line.startswith("CMAKE_UARCH:"):
            _, value = line.split("=", 1)
            value = value.strip()
            return value or None
    return None


def resolve_platform_dir(args: argparse.Namespace) -> Path:
    if args.platform_dir is not None:
        platform_dir = args.platform_dir
    else:
        uarch = detect_build_uarch(args.build_dir)
        platform_map = {
            "ZEN4": "zen4",
            "ZEN2": "zen2",
            "ARM-A72": "arm_a72",
            "ARM-A55": "arm_a55",
            "ARM-A78": "arm_a78",
            "ARM-X1": "arm_x1",
            "Q-ARM7E": "quantized_arm7E",
            "REF": "reference",
        }
        platform_dir = REPO_ROOT / "include" / "small" / "platforms" / platform_map.get(
            uarch or "ZEN2",
            DEFAULT_PLATFORM.name,
        )

    if not platform_dir.exists():
        raise ValueError(f"platform dir not found: {platform_dir}")
    return platform_dir


def uarch_compile_flags(build_dir: Path) -> list[str]:
    uarch = detect_build_uarch(build_dir)
    if uarch == "ZEN4":
        return ["-mavx2", "-mavx512f", "-mavx512vnni", "-mfma", "-march=native"]
    if uarch == "ZEN2":
        return ["-mavx2", "-mfma", "-march=native"]
    if uarch == "ARM-A72":
        return ["-march=armv8.2-a"]
    if uarch == "ARM-A55":
        return ["-static", "-march=armv8.2-a"]
    if uarch == "ARM-A78":
        return ["-static", "-march=armv8.2-a"]
    if uarch == "ARM-X1":
        return ["-static", "-march=armv8.2-a"]
    return ["-DUARCH_REF"]


def logical_numel(args: argparse.Namespace) -> int:
    return args.logical_channels * args.height * args.width


def make_input(args: argparse.Namespace) -> np.ndarray:
    size = logical_numel(args)
    if args.input_values is not None:
        values = np.fromstring(args.input_values, sep=",", dtype=np.float32)
        if values.size != size:
            raise ValueError(
                f"--input-values has {values.size} elements, expected {size} "
                "(logical_channels*height*width)"
            )
        return values.reshape(args.logical_channels, args.height, args.width)

    if args.init == "zeros":
        return np.zeros((args.logical_channels, args.height, args.width), dtype=np.float32)
    if args.init == "ones":
        return np.ones((args.logical_channels, args.height, args.width), dtype=np.float32)
    if args.init == "linspace":
        values = np.linspace(args.value_min, args.value_max, num=size, dtype=np.float32)
        return values.reshape(args.logical_channels, args.height, args.width)

    rng = np.random.default_rng(args.seed)
    values = rng.uniform(args.value_min, args.value_max, size=size).astype(np.float32)
    return values.reshape(args.logical_channels, args.height, args.width)


def pytorch_softmax_reference(logical_input: np.ndarray) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the reference comparison. "
            "Install torch in this environment or rerun with --skip-pytorch."
        ) from exc

    input_tensor = torch.from_numpy(logical_input).unsqueeze(0).to(torch.float32)
    output = torch.softmax(input_tensor, dim=1)
    return output.squeeze(0).cpu().numpy()


def render_ctypes_shim() -> str:
    return r"""
#include <cstdint>
#include <cstring>
#include <exception>
#include <numeric>

#include <small.h>
#include <small/interface_abstract.hpp>

extern "C" int small_softmax_float(
    uint32_t logical_channels,
    uint32_t height,
    uint32_t width,
    float const *logical_input_raw,
    float *logical_output_raw,
    char *error_buf,
    size_t error_buf_size)
{
    try
    {
        if (logical_channels == 0 || height == 0 || width == 0)
        {
            throw std::invalid_argument("logical_channels, height, and width must be positive");
        }

        using BufferT = small::FloatBuffer;
        uint32_t const channel_block =
            std::lcm(static_cast<uint32_t>(BufferT::C_ib),
                     static_cast<uint32_t>(BufferT::C_ob));
        uint32_t const storage_channels =
            ((logical_channels + channel_block - 1) / channel_block) * channel_block;
        size_t const logical_size = static_cast<size_t>(logical_channels) * height * width;

        BufferT input(storage_channels * height * width);
        small::init_zeros(input, input.size());

        for (uint32_t c = 0; c < logical_channels; ++c)
        {
            for (uint32_t h = 0; h < height; ++h)
            {
                for (uint32_t w = 0; w < width; ++w)
                {
                    size_t logical_ix = (c * height + h) * width + w;
                    size_t storage_ix = logical_ix;
                    input[storage_ix] = logical_input_raw[logical_ix];
                }
            }
        }

        BufferT packed_input(input.size());
        small::pack_buffer(input, small::INPUT,
                           1U, storage_channels, height, width,
                           BufferT::C_ib, BufferT::C_ob,
                           packed_input);

        BufferT packed_output(storage_channels * height * width);
        small::SoftMax(storage_channels, logical_channels, height, width,
                       packed_input, packed_output);

        BufferT output(storage_channels * height * width);
        small::unpack_buffer(packed_output, small::OUTPUT,
                             1U, storage_channels, height, width,
                             BufferT::C_ib, BufferT::C_ob,
                             output);

        for (size_t ix = 0; ix < logical_size; ++ix)
        {
            logical_output_raw[ix] = output[ix];
        }

        if (error_buf != nullptr && error_buf_size > 0)
        {
            error_buf[0] = '\0';
        }
        return 0;
    }
    catch (std::exception const &ex)
    {
        if (error_buf != nullptr && error_buf_size > 0)
        {
            std::strncpy(error_buf, ex.what(), error_buf_size - 1);
            error_buf[error_buf_size - 1] = '\0';
        }
        return 1;
    }
    catch (...)
    {
        if (error_buf != nullptr && error_buf_size > 0)
        {
            std::strncpy(error_buf, "unknown exception", error_buf_size - 1);
            error_buf[error_buf_size - 1] = '\0';
        }
        return 1;
    }
}
"""


def cached_library_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    platform_dir = resolve_platform_dir(args)
    flags = uarch_compile_flags(args.build_dir)
    cache_key = hashlib.sha256(
        (
            str(platform_dir.resolve())
            + "\n"
            + str((REPO_ROOT / "include").resolve())
            + "\n"
            + " ".join(flags)
            + "\n"
            + render_ctypes_shim()
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = args.cache_dir / cache_key
    return cache_dir / "softmax_ctypes.cpp", cache_dir / "libsmall_softmax_ctypes.so"


def ensure_small_ctypes_library(args: argparse.Namespace) -> Path:
    platform_dir = resolve_platform_dir(args)
    source, library = cached_library_paths(args)
    source.parent.mkdir(parents=True, exist_ok=True)
    shim_source = render_ctypes_shim()

    if args.rebuild or not source.exists() or source.read_text() != shim_source:
        source.write_text(shim_source)

    if args.rebuild or not library.exists():
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-fopenmp",
            "-O0",
            "-g",
            "-fpermissive",
            *uarch_compile_flags(args.build_dir),
            f"-I{REPO_ROOT / 'include'}",
            f"-I{platform_dir}",
            str(source),
            "-o",
            str(library),
        ]
        subprocess.run(compile_cmd, check=True)

    return library


def run_small_ctypes(args: argparse.Namespace, logical_input: np.ndarray) -> np.ndarray:
    library_path = ensure_small_ctypes_library(args)
    library = ctypes.CDLL(str(library_path))

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

    input_array = np.ascontiguousarray(logical_input.reshape(-1), dtype=np.float32)
    output_array = np.zeros_like(input_array)
    error_buffer = ctypes.create_string_buffer(4096)

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
        error_text = error_buffer.value.decode("utf-8", errors="replace") or "unknown error"
        raise RuntimeError(f"SMaLL SoftMax ctypes call failed: {error_text}")

    return output_array.reshape(args.logical_channels, args.height, args.width)


def summarize_mismatch(
    actual: np.ndarray,
    reference: np.ndarray,
    logical_input: np.ndarray,
) -> str:
    abs_diff = np.abs(actual - reference)
    max_flat_ix = int(np.argmax(abs_diff))
    c, h, w = np.unravel_index(max_flat_ix, abs_diff.shape)
    return (
        "allclose failed: "
        f"max_abs_diff={abs_diff[c, h, w]:.9g} at (c={c}, h={h}, w={w})\n"
        f"input={logical_input[c, h, w]:.9g} "
        f"small={actual[c, h, w]:.9g} "
        f"pytorch={reference[c, h, w]:.9g}"
    )


def main() -> int:
    args = parse_args()
    logical_input = make_input(args)
    actual = run_small_ctypes(args, logical_input)

    print(
        f"Ran SMaLL SoftMax with logical_channels={args.logical_channels}, "
        f"H={args.height}, W={args.width}, init={args.init if args.input_values is None else 'manual'}, "
        f"platform={resolve_platform_dir(args).name}",
        flush=True,
    )

    if args.verbose:
        print("Input:", logical_input.reshape(-1), flush=True)
        print("SMaLL:", actual.reshape(-1), flush=True)

    if not args.skip_pytorch:
        reference = pytorch_softmax_reference(logical_input)
        if args.verbose:
            print("PyTorch:", reference.reshape(-1), flush=True)

        if not np.allclose(actual, reference, rtol=args.rtol, atol=args.atol):
            raise AssertionError(summarize_mismatch(actual, reference, logical_input))

        max_abs_diff = float(np.max(np.abs(actual - reference)))
        print(
            f"numpy.allclose passed against PyTorch fp32 "
            f"(rtol={args.rtol}, atol={args.atol}, max_abs_diff={max_abs_diff:.9g})",
            flush=True,
        )
    else:
        print("Skipped PyTorch comparison (--skip-pytorch).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

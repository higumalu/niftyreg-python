#!/usr/bin/env python3
"""Smoke test after install: run a tiny F3D job with ``use_cuda=True``.

This script checks that ``niftyreg`` imports correctly and that a CUDA build can
execute the GPU code path (requires a CUDA-enabled wheel/binary, working NVIDIA
driver, and a visible GPU).

What it does *not* guarantee by itself:
- That your pip wheel was compiled *without* CUDA (``use_cuda=True`` may still
  run on CPU in that case). Use ``--verbose`` and look for stderr from the
  binding: ``use_cuda=True -> PlatformType::Cuda`` on CUDA-enabled builds.

Usage::

    uv run python examples/verify_gpu_install.py
    uv run python examples/verify_gpu_install.py --verbose
    uv run python examples/verify_gpu_install.py --skip-nvidia-smi
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def check_nvidia_smi() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "unknown error").strip()
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        return True, lines[0] if lines else proc.stdout.strip()
    except FileNotFoundError:
        return False, "nvidia-smi not found (driver not installed or not on PATH)"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"


def run_gpu_f3d(*, verbose: bool) -> None:
    import numpy as np
    import SimpleITK as sitk

    import niftyreg

    rng = np.random.default_rng(42)
    shape = (24, 24, 24)
    arr = (100.0 + 20.0 * rng.standard_normal(shape)).astype(np.float32)
    fixed = sitk.GetImageFromArray(arr)
    fixed.SetSpacing((1.0, 1.0, 1.0))
    moving = sitk.GetImageFromArray(np.roll(arr, shift=2, axis=2))
    moving.CopyInformation(fixed)

    result = niftyreg.f3d(
        fixed,
        moving,
        options=niftyreg.F3DOptions(
            use_cuda=True,
            use_velocity=False,
            use_symmetric=False,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            spacing_x=-6.0,
            padding_value=0.0,
            verbose=verbose,
        ),
    )

    warped = sitk.GetArrayFromImage(result.warped)
    if not np.all(np.isfinite(warped)):
        raise RuntimeError("warped image contains non-finite values")
    if result.warped.GetSize() != fixed.GetSize():
        raise RuntimeError(
            f"warped size {result.warped.GetSize()} != fixed size {fixed.GetSize()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify niftyreg is installed and can run F3D with use_cuda=True.",
    )
    parser.add_argument(
        "--skip-nvidia-smi",
        action="store_true",
        help="Do not run nvidia-smi (useful in CI without a GPU node).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable NiftyReg backend logs; CUDA builds may print PlatformType::Cuda on stderr.",
    )
    args = parser.parse_args()

    if not args.skip_nvidia_smi:
        ok, msg = check_nvidia_smi()
        if ok:
            print("[check] nvidia-smi:", msg)
        else:
            print("[warn] nvidia-smi:", msg, file=sys.stderr)

    print("[run] minimal F3D (use_cuda=True) on synthetic 24³ volumes …")
    try:
        run_gpu_f3d(verbose=args.verbose)
    except Exception as exc:
        print("[fail]", exc, file=sys.stderr)
        sys.exit(1)

    print("[ok] niftyreg.f3d finished with use_cuda=True.")
    if not args.verbose:
        print(
            "[tip] Re-run with --verbose; on CUDA-enabled builds stderr may show "
            "'use_cuda=True -> PlatformType::Cuda'."
        )


if __name__ == "__main__":
    main()

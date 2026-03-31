from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

import niftyreg


def main() -> None:
    parser = argparse.ArgumentParser(description="Register moving to fixed using GPU-accelerated F3D.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for warped moving image (.nii.gz). Default: examples/outputs/moving_f3d_gpu_warped.nii.gz",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    output_path = args.output
    if output_path is None:
        output_path = repo_root / "niftyreg-python" / "examples" / "outputs" / "moving_f3d_gpu_warped.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed = sitk.ReadImage(str(data_dir / "fixed.nii.gz"), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(data_dir / "moving.nii.gz"), sitk.sitkFloat32)

    result = niftyreg.f3d(
        fixed,
        moving,
        options=niftyreg.F3DOptions(
            use_cuda=True,  # Use CUDA backend for GPU acceleration.
            use_velocity=False,  # Use classic F3D (CPP) instead of velocity-field integration.
            use_symmetric=False,  # Disable symmetric optimization (single forward registration).
            number_of_levels=3,  # Total multi-resolution pyramid levels.
            levels_to_perform=3,  # Number of levels actually executed from coarse to fine.
            max_iterations=25,  # Iteration cap at the finest level.
            spacing_x=-6.0,  # Final control-point spacing in voxels (negative means voxel units).
            bending_energy_weight=0.0001,  # Smoothness penalty (second-order deformation regularization).
            linear_energy_weight=0.002,  # First-order deformation regularization strength.
            padding_value=-1000.0,  # Ignore/background fill value for out-of-FOV sampled voxels.
            verbose=False,  # Disable NiftyReg console logging.
        ),
    )

    warped_int16 = sitk.Cast(result.warped, sitk.sitkInt16)
    sitk.WriteImage(warped_int16, str(output_path))
    print("Warped image saved to:", output_path)


if __name__ == "__main__":
    main()


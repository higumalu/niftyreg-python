from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    parser.add_argument(
        "--deformation-output",
        type=Path,
        default=None,
        help="Output path for deformation field (.nii.gz). Default: examples/outputs/moving_f3d_gpu_deformation.nii.gz",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    output_path = args.output
    if output_path is None:
        output_path = repo_root / "niftyreg-python" / "examples" / "outputs" / "moving_f3d_gpu_warped.nii.gz"
    deformation_output_path = args.deformation_output
    if deformation_output_path is None:
        deformation_output_path = repo_root / "niftyreg-python" / "examples" / "outputs" / "moving_f3d_gpu_deformation.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deformation_output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed = sitk.ReadImage(str(data_dir / "fixed.nii.gz"), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(data_dir / "moving.nii.gz"), sitk.sitkFloat32)

    options = niftyreg.F3DOptions(
        use_cuda=True,  # Use CUDA backend for GPU acceleration.
        use_velocity=False,  # Use classic F3D (CPP) instead of velocity-field integration.
        use_symmetric=False,  # Disable symmetric optimization (single forward registration).
        number_of_levels=3,  # Total multi-resolution pyramid levels.
        levels_to_perform=3,  # Number of levels actually executed from coarse to fine.
        max_iterations=25,  # Iteration cap at the finest level.
        spacing_x=-3.0,  # Final control-point spacing in voxels (negative means voxel units).
        bending_energy_weight=0.0001,  # Smoothness penalty (second-order deformation regularization).
        linear_energy_weight=0.002,  # First-order deformation regularization strength.
        padding_value=-1000.0,  # Ignore/background fill value for out-of-FOV sampled voxels.
        verbose=False,  # Disable NiftyReg console logging.
    )

    result = niftyreg.f3d(
        fixed,
        moving,
        options=options,
    )

    deformation = niftyreg.transform_to_deformation(fixed, result.cpp)
    # reg_f3d warped output matches cubic (order-3) resampling semantics.
    verify_interpolation = 3
    warped_from_deformation = niftyreg.resample(
        fixed,
        moving,
        deformation,
        interpolation=verify_interpolation,
        padding_value=options.padding_value,
    )
    diff = sitk.Abs(result.warped - warped_from_deformation)
    diff_arr = sitk.GetArrayViewFromImage(diff)
    max_abs_diff = float(np.max(diff_arr))
    mean_abs_diff = float(np.mean(diff_arr))

    atol = 1e-3
    rtol = 1e-5
    same_as_warped = bool(
        np.allclose(
            sitk.GetArrayViewFromImage(result.warped),
            sitk.GetArrayViewFromImage(warped_from_deformation),
            atol=atol,
            rtol=rtol,
        )
    )
    print(
        "Warp consistency (resample with deformation(result.cpp) vs result.warped):",
        "PASS" if same_as_warped else "CHECK",
        f"(max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}, interpolation={verify_interpolation}, atol={atol}, rtol={rtol})",
    )

    warped_int16 = sitk.Cast(result.warped, sitk.sitkInt16)
    sitk.WriteImage(warped_int16, str(output_path))
    print("Warped image saved to:", output_path)

    if deformation.image is None:
        raise RuntimeError("deformation transform does not contain an image")
    sitk.WriteImage(deformation.image, str(deformation_output_path))
    print(deformation.image)
    print("Deformation field saved to:", deformation_output_path)


if __name__ == "__main__":
    main()


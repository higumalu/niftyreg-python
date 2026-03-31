from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import SimpleITK as sitk

import niftyreg


PRESETS = {
    "quality": {
        "rigid": dict(
            symmetric=True,
            use_cuda=True,
            rigid_only=True,
            affine_direct=False,
            number_of_levels=3,
            levels_to_perform=3,
            max_iterations=5,
            padding_value=0.0,
        ),
        "f3d": dict(
            use_velocity=False,
            use_symmetric=False,
            use_cuda=True,
            number_of_levels=2,
            levels_to_perform=2,
            max_iterations=25,
            spacing_x=-8.0,
            bending_energy_weight=0.0005,
            linear_energy_weight=0.005,
            padding_value=0.0,
        ),
    },
    "balanced": {
        "rigid": dict(
            symmetric=False,
            use_cuda=True,
            rigid_only=True,
            affine_direct=False,
            number_of_levels=3,
            levels_to_perform=3,
            max_iterations=5,
            padding_value=0.0,
        ),
        "f3d": dict(
            use_velocity=False,
            use_symmetric=False,
            use_cuda=True,
            number_of_levels=2,
            levels_to_perform=2,
            max_iterations=25,
            spacing_x=-8.0,
            bending_energy_weight=0.0005,
            linear_energy_weight=0.005,
            padding_value=0.0,
        ),
    },
    "fast": {
        "rigid": dict(
            symmetric=False,
            use_cuda=True,
            rigid_only=True,
            affine_direct=False,
            number_of_levels=3,
            levels_to_perform=3,
            max_iterations=5,
            padding_value=0.0,
        ),
        "f3d": dict(
            use_velocity=False,
            use_symmetric=False,
            use_cuda=True,
            number_of_levels=2,
            levels_to_perform=2,
            max_iterations=15,
            spacing_x=-10.0,
            bending_energy_weight=0.0005,
            linear_energy_weight=0.005,
            padding_value=0.0,
        ),
    },
}


def affine_lps_to_sitk_transform(matrix_lps: np.ndarray, dimension: int) -> sitk.Transform:
    affine = sitk.AffineTransform(dimension)
    linear = matrix_lps[:dimension, :dimension].reshape(-1)
    translation = matrix_lps[:dimension, 3]
    affine.SetMatrix(tuple(float(v) for v in linear))
    affine.SetTranslation(tuple(float(v) for v in translation))
    return affine


def displacement_transform_from_cpp(reference: sitk.Image, cpp: niftyreg.Transform) -> tuple[sitk.Transform, sitk.Image]:
    displacement = niftyreg.transform_to_displacement(reference, cpp).image
    displacement = sitk.Cast(displacement, sitk.sitkVectorFloat64)
    displacement_for_transform = sitk.Image(displacement)
    return sitk.DisplacementFieldTransform(displacement_for_transform), displacement


def compare_to_fixed(fixed: sitk.Image, moving: sitk.Image, label: str) -> None:
    if (
        fixed.GetSize() != moving.GetSize()
        or fixed.GetSpacing() != moving.GetSpacing()
        or fixed.GetOrigin() != moving.GetOrigin()
        or fixed.GetDirection() != moving.GetDirection()
    ):
        moving = sitk.Resample(
            moving,
            fixed,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )

    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_array = sitk.GetArrayFromImage(moving)
    mae = float(np.mean(np.abs(fixed_array - moving_array)))
    rmse = float(np.sqrt(np.mean((fixed_array - moving_array) ** 2)))
    corr = float(np.corrcoef(fixed_array.ravel(), moving_array.ravel())[0, 1])
    metrics = niftyreg.measure(fixed, moving, metrics=("nmi", "ssd", "ncc"))

    print(f"{label} vs fixed:")
    print(
        "  metrics:",
        f"NMI={metrics['nmi']:.6f}",
        f"SSD={metrics['ssd']:.6f}",
        f"NCC={metrics['ncc']:.6f}",
    )
    print(
        "  voxel diff:",
        f"MAE={mae:.6f}",
        f"RMSE={rmse:.6f}",
        f"Corr={corr:.6f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="balanced",
        help="quality: best MAE, balanced: strong quality/time tradeoff, fast: slightly worse MAE but fastest useful preset",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    output_dir = repo_root / "niftyreg-python" / "examples" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = PRESETS[args.preset]

    fixed = sitk.ReadImage(str(data_dir / "fixed.nii.gz"), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(data_dir / "moving.nii.gz"), sitk.sitkFloat32)

    print(f"Preset: {args.preset}")
    print("Input comparison:")
    compare_to_fixed(fixed, moving, "Original moving")

    rigid_start = perf_counter()
    rigid_result = niftyreg.aladin(
        fixed,
        moving,
        options=niftyreg.AladinOptions(**preset["rigid"]),
    )
    rigid_elapsed = perf_counter() - rigid_start
    rigid_moving = rigid_result.warped
    rigid_transform = affine_lps_to_sitk_transform(rigid_result.affine_lps, fixed.GetDimension())

    sitk.WriteImage(rigid_moving, str(output_dir / "moving_rigid.nii.gz"))
    sitk.WriteTransform(rigid_transform, str(output_dir / "moving_rigid.tfm"))

    print(f"Rigid registration time: {rigid_elapsed:.3f}s")
    compare_to_fixed(fixed, rigid_moving, "Rigid moving")

    f3d_start = perf_counter()
    f3d_result = niftyreg.f3d(
        fixed,
        moving,
        initial_transform=rigid_result.transform,
        options=niftyreg.F3DOptions(**preset["f3d"]),
    )
    f3d_elapsed = perf_counter() - f3d_start
    deformed_moving = f3d_result.warped
    deformation_transform, displacement_field = displacement_transform_from_cpp(fixed, f3d_result.cpp)

    sitk.WriteImage(deformed_moving, str(output_dir / "moving_rigid_f3d.nii.gz"))
    sitk.WriteImage(displacement_field, str(output_dir / "moving_rigid_f3d_displacement.nii.gz"))
    sitk.WriteTransform(deformation_transform, str(output_dir / "moving_rigid_f3d.h5"))

    print(f"F3D registration time: {f3d_elapsed:.3f}s")
    compare_to_fixed(fixed, deformed_moving, "Deformed moving")

    print("Rigid moving image:", output_dir / "moving_rigid.nii.gz")
    print("Rigid transform:", rigid_transform)
    print("Rigid transform file:", output_dir / "moving_rigid.tfm")
    print("F3D deformed image:", output_dir / "moving_rigid_f3d.nii.gz")
    print("F3D displacement transform:", deformation_transform)
    print("F3D displacement field:", output_dir / "moving_rigid_f3d_displacement.nii.gz")
    print("F3D transform file:", output_dir / "moving_rigid_f3d.h5")


if __name__ == "__main__":
    main()

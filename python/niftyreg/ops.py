from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import SimpleITK as sitk

from . import _niftyreg
from ._interop import (
    Transform,
    _image_to_xyz_array,
    _matrix_ras_to_lps,
    _metadata_for_cpp,
    _serialize_transform,
    _xyz_array_to_sitk,
    affine_transform,
    image_transform,
)


@dataclass(slots=True)
class JacobianResult:
    determinant: Optional[sitk.Image] = None
    log_determinant: Optional[sitk.Image] = None
    matrix: Optional[sitk.Image] = None


def resample(
    reference: sitk.Image,
    floating: sitk.Image,
    transform: Transform,
    *,
    interpolation: int = 3,
    padding_value: float = 0.0,
) -> sitk.Image:
    payload = _niftyreg.resample_image(
        _image_to_xyz_array(reference, dtype="float32"),
        _image_to_xyz_array(floating, dtype="float32"),
        _metadata_for_cpp(reference),
        _metadata_for_cpp(floating),
        _serialize_transform(transform),
        interpolation,
        padding_value,
    )
    return _xyz_array_to_sitk(payload, reference)


def transform_to_deformation(reference: sitk.Image, transform: Transform) -> Transform:
    payload = _niftyreg.transform_to_deformation(_metadata_for_cpp(reference), _serialize_transform(transform))
    return image_transform("deformation", _xyz_array_to_sitk(payload["array"], reference, is_vector=True))


def transform_to_displacement(reference: sitk.Image, transform: Transform) -> Transform:
    payload = _niftyreg.transform_to_displacement(_metadata_for_cpp(reference), _serialize_transform(transform))
    return image_transform("displacement", _xyz_array_to_sitk(payload["array"], reference, is_vector=True))


def transform_to_flow(reference: sitk.Image, transform: Transform) -> Transform:
    payload = _niftyreg.transform_to_flow(_metadata_for_cpp(reference), _serialize_transform(transform))
    return image_transform("flow", _xyz_array_to_sitk(payload["array"], reference, is_vector=True))


def compose_transforms(
    reference: sitk.Image,
    transform1: Transform,
    transform2: Transform,
    *,
    reference2: Optional[sitk.Image] = None,
) -> Transform:
    payload = _niftyreg.compose_transforms(
        _metadata_for_cpp(reference),
        _serialize_transform(transform1),
        _serialize_transform(transform2),
        None if reference2 is None else _metadata_for_cpp(reference2),
    )
    return image_transform("deformation", _xyz_array_to_sitk(payload["array"], reference, is_vector=True))


def invert_transform(reference: sitk.Image, transform: Transform) -> Transform:
    payload = _niftyreg.invert_transform(_metadata_for_cpp(reference), _serialize_transform(transform))
    if payload["kind"] == "affine":
        return affine_transform(_matrix_ras_to_lps(payload["matrix_ras"]))
    return image_transform(payload["kind"], _xyz_array_to_sitk(payload["array"], reference, is_vector=True))


def jacobian(
    reference: sitk.Image,
    transform: Transform,
    *,
    determinant: bool = True,
    log_determinant: bool = False,
    matrix: bool = False,
) -> JacobianResult:
    payload = _niftyreg.jacobian(
        _metadata_for_cpp(reference),
        _serialize_transform(transform),
        determinant,
        log_determinant,
        matrix,
    )
    return JacobianResult(
        determinant=None if payload["determinant"] is None else _xyz_array_to_sitk(payload["determinant"], reference),
        log_determinant=None if payload["log_determinant"] is None else _xyz_array_to_sitk(payload["log_determinant"], reference),
        matrix=None if payload["matrix"] is None else _xyz_array_to_sitk(payload["matrix"], reference, is_vector=True),
    )


def measure(
    reference: sitk.Image,
    floating: sitk.Image,
    *,
    reference_mask: Optional[sitk.Image] = None,
    transform: Optional[Transform] = None,
    interpolation: int = 3,
    padding_value: float = float("nan"),
    metrics: Sequence[str] = ("nmi",),
) -> dict[str, float]:
    return _niftyreg.measure_similarity(
        _image_to_xyz_array(reference, dtype="float32"),
        _image_to_xyz_array(floating, dtype="float32"),
        _metadata_for_cpp(reference),
        _metadata_for_cpp(floating),
        None if reference_mask is None else _image_to_xyz_array(reference_mask, dtype="uint8"),
        None if transform is None else _serialize_transform(transform),
        interpolation,
        padding_value,
        list(metrics),
    )


def average_affines(affines_lps: Iterable[np.ndarray], *, lts_inlier: float = 1.0) -> np.ndarray:
    payload = _niftyreg.average_affines([np.asarray(matrix, dtype=np.float64) for matrix in affines_lps], lts_inlier)
    return _matrix_ras_to_lps(payload)


def average_images(images: Sequence[sitk.Image]) -> sitk.Image:
    if not images:
        raise ValueError("images must not be empty")
    arrays = [sitk.GetArrayFromImage(image).astype(np.float32) for image in images]
    mean = np.mean(arrays, axis=0)
    out = sitk.GetImageFromArray(mean, isVector=images[0].GetNumberOfComponentsPerPixel() > 1)
    out.CopyInformation(images[0])
    return out


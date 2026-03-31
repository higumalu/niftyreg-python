from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import SimpleITK as sitk

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


@dataclass(slots=True)
class Transform:
    kind: str
    affine_lps: Optional[np.ndarray] = None
    image: Optional[sitk.Image] = None

    def __post_init__(self) -> None:
        if self.kind == "affine":
            if self.affine_lps is None:
                raise ValueError("affine_lps is required for affine transforms")
            self.affine_lps = np.asarray(self.affine_lps, dtype=np.float64)
            if self.affine_lps.shape != (4, 4):
                raise ValueError("affine_lps must have shape (4, 4)")
        else:
            if self.image is None:
                raise ValueError(f"image is required for transform kind={self.kind}")


def affine_transform(matrix_lps: np.ndarray) -> Transform:
    return Transform(kind="affine", affine_lps=np.asarray(matrix_lps, dtype=np.float64))


def image_transform(kind: str, image: sitk.Image) -> Transform:
    return Transform(kind=kind, image=image)


def _image_to_xyz_array(image: sitk.Image, *, dtype: np.dtype) -> np.ndarray:
    array = sitk.GetArrayViewFromImage(image)
    if image.GetNumberOfComponentsPerPixel() > 1 and image.GetDimension() not in (2, 3):
        raise ValueError(f"Only 2D/3D vector images are supported, got dimension={image.GetDimension()}")
    if image.GetNumberOfComponentsPerPixel() == 1 and array.ndim not in (2, 3):
        raise ValueError(f"Only 2D/3D scalar images are supported, got ndim={array.ndim}")
    return np.ascontiguousarray(array, dtype=dtype)


def _sitk_affine_lps(image: sitk.Image) -> np.ndarray:
    dimension = image.GetDimension()
    if dimension not in (2, 3):
        raise ValueError(f"Only 2D/3D images are supported, got dimension={dimension}")
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(dimension, dimension)
    affine = np.eye(4, dtype=np.float64)
    affine[:dimension, :dimension] = direction @ np.diag(spacing)
    affine[:dimension, 3] = origin
    return affine


def _metadata_for_cpp(image: sitk.Image) -> dict[str, np.ndarray]:
    affine_lps = _sitk_affine_lps(image)
    affine_ras = _LPS_TO_RAS @ affine_lps
    return {
        "shape_xyz": np.asarray(image.GetSize(), dtype=np.int64),
        "spacing_xyz": np.asarray(image.GetSpacing(), dtype=np.float64),
        "origin_xyz": np.asarray(image.GetOrigin(), dtype=np.float64),
        "direction": np.asarray(image.GetDirection(), dtype=np.float64),
        "affine_ras": affine_ras,
    }


def _copy_information(target: sitk.Image, source: sitk.Image) -> sitk.Image:
    target.SetOrigin(source.GetOrigin())
    target.SetSpacing(source.GetSpacing())
    target.SetDirection(source.GetDirection())
    return target


def _same_space(lhs: sitk.Image, rhs: sitk.Image, *, atol: float = 1e-5) -> bool:
    return (
        lhs.GetDimension() == rhs.GetDimension()
        and lhs.GetSize() == rhs.GetSize()
        and np.allclose(lhs.GetSpacing(), rhs.GetSpacing(), atol=atol)
        and np.allclose(lhs.GetOrigin(), rhs.GetOrigin(), atol=atol)
        and np.allclose(lhs.GetDirection(), rhs.GetDirection(), atol=atol)
    )


def _xyz_array_to_sitk(array_xyz: np.ndarray, reference: sitk.Image, *, is_vector: bool = False) -> sitk.Image:
    array_xyz = np.asarray(array_xyz)
    if is_vector and array_xyz.ndim not in (3, 4):
        raise ValueError(f"Unexpected vector output ndim={array_xyz.ndim}")
    if not is_vector and array_xyz.ndim not in (2, 3):
        raise ValueError(f"Unexpected scalar output ndim={array_xyz.ndim}")
    image = sitk.GetImageFromArray(array_xyz, isVector=is_vector)
    return _copy_information(image, reference)


def _matrix_ras_to_lps(matrix_ras: np.ndarray) -> np.ndarray:
    return _LPS_TO_RAS @ np.asarray(matrix_ras, dtype=np.float64) @ _LPS_TO_RAS


def _matrix_lps_to_ras(matrix_lps: np.ndarray) -> np.ndarray:
    return _LPS_TO_RAS @ np.asarray(matrix_lps, dtype=np.float64) @ _LPS_TO_RAS


def _xyz_array_to_sitk_from_meta(array_xyz: np.ndarray, meta: dict, *, is_vector: bool = False) -> sitk.Image:
    if is_vector and array_xyz.ndim not in (3, 4):
        raise ValueError(f"Unexpected vector output ndim={array_xyz.ndim}")
    if not is_vector and array_xyz.ndim not in (2, 3):
        raise ValueError(f"Unexpected scalar output ndim={array_xyz.ndim}")
    image = sitk.GetImageFromArray(np.asarray(array_xyz), isVector=is_vector)

    spacing = tuple(float(v) for v in np.asarray(meta["spacing_xyz"]).tolist())
    origin = tuple(float(v) for v in np.asarray(meta["origin_xyz"]).tolist())
    direction = tuple(float(v) for v in np.asarray(meta["direction"]).tolist())
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def _ensure_same_dimension(reference: sitk.Image, floating: sitk.Image) -> None:
    if reference.GetDimension() != floating.GetDimension():
        raise ValueError("reference and floating images must have the same dimensionality")


def _serialize_transform(transform: Transform) -> dict:
    if transform.kind == "affine":
        return {
            "kind": "affine",
            "matrix_ras": _matrix_lps_to_ras(transform.affine_lps),
        }
    image = transform.image
    assert image is not None
    return {
        "kind": transform.kind,
        "array": _image_to_xyz_array(image, dtype=np.float32),
        "meta": _metadata_for_cpp(image),
    }


def _deserialize_transform(payload: dict, reference: sitk.Image) -> Transform:
    kind = payload["kind"]
    if kind == "affine":
        return affine_transform(_matrix_ras_to_lps(payload["matrix_ras"]))
    return image_transform(kind, _xyz_array_to_sitk(payload["array"], reference, is_vector=True))

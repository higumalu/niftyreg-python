from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import SimpleITK as sitk

from . import _niftyreg
from ._interop import (
    Transform,
    _ensure_same_dimension,
    _image_to_xyz_array,
    _matrix_ras_to_lps,
    _metadata_for_cpp,
    _same_space,
    _xyz_array_to_sitk,
    affine_transform,
)


@dataclass(slots=True)
class AladinOptions:
    symmetric: bool = True
    # Select NiftyReg execution platform.
    # NOTE: requires a CUDA-enabled build and supported GPU.
    use_cuda: bool = False
    rigid_only: bool = False
    affine_direct: bool = False
    max_iterations: int = 5
    number_of_levels: int = 3
    levels_to_perform: Optional[int] = None
    block_percentage: int = 50
    inlier_lts: int = 50
    block_step_size: int = 1
    interpolation: int = 1
    floating_sigma: float = 0.0
    reference_sigma: float = 0.0
    reference_lower_threshold: Optional[float] = None
    reference_upper_threshold: Optional[float] = None
    floating_lower_threshold: Optional[float] = None
    floating_upper_threshold: Optional[float] = None
    padding_value: float = float("nan")
    align_centre: bool = True
    align_centre_mass: int = 0
    verbose: bool = False


@dataclass(slots=True)
class AladinResult:
    warped: sitk.Image
    affine_lps: np.ndarray
    transform: Transform


def aladin(
    reference: sitk.Image,
    floating: sitk.Image,
    reference_mask: Optional[sitk.Image] = None,
    floating_mask: Optional[sitk.Image] = None,
    options: Optional[AladinOptions] = None,
) -> AladinResult:
    if options is None:
        options = AladinOptions()
    _ensure_same_dimension(reference, floating)

    ref_array = _image_to_xyz_array(reference, dtype=np.float32)
    flo_array = _image_to_xyz_array(floating, dtype=np.float32)

    ref_mask_array = None
    flo_mask_array = None
    if reference_mask is not None:
        if not _same_space(reference_mask, reference):
            raise ValueError("reference_mask must match the reference image geometry")
        ref_mask_array = _image_to_xyz_array(reference_mask, dtype=np.uint8)
    if floating_mask is not None:
        if not _same_space(floating_mask, floating):
            raise ValueError("floating_mask must match the floating image geometry")
        flo_mask_array = _image_to_xyz_array(floating_mask, dtype=np.uint8)

    payload = _niftyreg.run_aladin(
        ref_array,
        flo_array,
        _metadata_for_cpp(reference),
        _metadata_for_cpp(floating),
        ref_mask_array,
        flo_mask_array,
        asdict(options),
    )

    warped = _xyz_array_to_sitk(payload["warped"], reference)
    affine_lps = _matrix_ras_to_lps(payload["affine_ras"])
    return AladinResult(warped=warped, affine_lps=affine_lps, transform=affine_transform(affine_lps))


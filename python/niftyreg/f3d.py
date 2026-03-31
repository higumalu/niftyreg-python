from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import SimpleITK as sitk

from . import _niftyreg
from ._interop import (
    Transform,
    _ensure_same_dimension,
    _image_to_xyz_array,
    _metadata_for_cpp,
    _same_space,
    _serialize_transform,
    _xyz_array_to_sitk,
    _xyz_array_to_sitk_from_meta,
    image_transform,
)


@dataclass(slots=True)
class F3DOptions:
    use_velocity: bool = False
    use_symmetric: bool = True
    # Select NiftyReg execution platform.
    # NOTE: NiftyReg GPU requires a CUDA-enabled build and a supported GPU.
    use_cuda: bool = False
    max_iterations: int = 150
    number_of_levels: int = 3
    levels_to_perform: Optional[int] = None
    use_pyramid: bool = True
    use_conjugate_gradient: bool = True
    use_approx_gradient: bool = False
    perturbation_number: int = 0
    interpolation: int = 1
    reference_sigma: float = 0.0
    floating_sigma: float = 0.0
    gradient_sigma: float = 0.0
    spacing_x: float = -5.0
    spacing_y: Optional[float] = None
    spacing_z: Optional[float] = None
    bending_energy_weight: float = 0.001
    linear_energy_weight: float = 0.01
    jacobian_log_weight: float = 0.0
    approximate_jacobian_log: bool = True
    inverse_consistency_weight: float = 0.0
    use_gradient_cumulative_exp: bool = True
    bch_terms: Optional[int] = None
    robust_range: bool = False
    padding_value: float = float("nan")
    reference_lower_threshold: Optional[float] = None
    reference_upper_threshold: Optional[float] = None
    floating_lower_threshold: Optional[float] = None
    floating_upper_threshold: Optional[float] = None
    similarity: str = "nmi"
    lncc_sigma: float = 2.0
    mind_offset: int = 1
    nmi_reference_bins: int = 64
    nmi_floating_bins: int = 64
    ssd_normalize: bool = True
    verbose: bool = False


@dataclass(slots=True)
class F3DResult:
    warped: sitk.Image
    cpp: Transform
    backward_cpp: Optional[Transform] = None


def f3d(
    reference: sitk.Image,
    floating: sitk.Image,
    reference_mask: Optional[sitk.Image] = None,
    floating_mask: Optional[sitk.Image] = None,
    initial_transform: Optional[Transform] = None,
    initial_cpp: Optional[Transform] = None,
    options: Optional[F3DOptions] = None,
) -> F3DResult:
    if options is None:
        options = F3DOptions()
    _ensure_same_dimension(reference, floating)

    if reference_mask is not None and not _same_space(reference_mask, reference):
        raise ValueError("reference_mask must match the reference image geometry")
    if floating_mask is not None and not _same_space(floating_mask, floating):
        raise ValueError("floating_mask must match the floating image geometry")

    if initial_transform is not None and initial_transform.kind != "affine":
        raise ValueError("initial_transform must be an affine transform")
    if initial_cpp is not None and initial_cpp.kind not in ("cpp", "spline_velocity"):
        raise ValueError("initial_cpp must be a cpp or spline_velocity transform")

    payload = _niftyreg.run_f3d(
        _image_to_xyz_array(reference, dtype="float32"),
        _image_to_xyz_array(floating, dtype="float32"),
        _metadata_for_cpp(reference),
        _metadata_for_cpp(floating),
        None if reference_mask is None else _image_to_xyz_array(reference_mask, dtype="uint8"),
        None if floating_mask is None else _image_to_xyz_array(floating_mask, dtype="uint8"),
        None if initial_transform is None else _serialize_transform(initial_transform),
        None if initial_cpp is None else _serialize_transform(initial_cpp),
        asdict(options),
    )

    cpp_kind = "spline_velocity" if options.use_velocity else "cpp"
    backward_cpp = None
    if payload["backward_cpp"] is not None:
        backward_cpp = image_transform(
            cpp_kind,
            _xyz_array_to_sitk_from_meta(payload["backward_cpp"], payload["backward_cpp_meta"], is_vector=True),
        )

    return F3DResult(
        warped=_xyz_array_to_sitk(payload["warped"], reference),
        cpp=image_transform(cpp_kind, _xyz_array_to_sitk_from_meta(payload["cpp"], payload["cpp_meta"], is_vector=True)),
        backward_cpp=backward_cpp,
    )

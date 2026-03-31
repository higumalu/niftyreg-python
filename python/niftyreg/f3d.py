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
    """Options for `reg_f3d` non-linear registration.

    Usage:
    - Start with defaults for most cases.
    - Tune `similarity`, `spacing_*`, and regularization weights first.
    - Enable `use_cuda` only when your NiftyReg build supports CUDA.
    """

    # Use stationary/spline velocity field parameterization.
    # True: diffeomorphic-style update via velocity + exponentiation.
    # False: optimize direct cubic B-spline control points.
    use_velocity: bool = False
    # Symmetric registration (estimate forward and backward fields together).
    # Usually improves inverse consistency at higher compute cost.
    use_symmetric: bool = True
    # Select NiftyReg execution platform.
    # NOTE: NiftyReg GPU requires a CUDA-enabled build and a supported GPU.
    use_cuda: bool = False
    # Maximum optimizer iterations per resolution level.
    # Increase for hard cases; lower for faster but less precise runs.
    max_iterations: int = 150
    # Number of multi-resolution pyramid levels to construct.
    # More levels help large deformations but take longer.
    number_of_levels: int = 3
    # Number of top levels to actually optimize.
    # None means use all `number_of_levels`.
    levels_to_perform: Optional[int] = None
    # Whether to use image pyramid (coarse-to-fine optimization).
    # Keep True unless you intentionally run single-scale.
    use_pyramid: bool = True
    # Use conjugate-gradient optimization update.
    # Usually converges faster than simple gradient descent.
    use_conjugate_gradient: bool = True
    # Use approximate gradient computation for speed.
    # May reduce accuracy; useful for quick experiments.
    use_approx_gradient: bool = False
    # Number of random perturbation evaluations for robust line search.
    # 0 disables perturbation-based strategy.
    perturbation_number: int = 0
    # Interpolation order/type used during resampling.
    # Common value: 1 (linear).
    interpolation: int = 1
    # Gaussian smoothing sigma for reference image (in voxel units).
    # >0 can stabilize noisy data.
    reference_sigma: float = 0.0
    # Gaussian smoothing sigma for floating image (in voxel units).
    floating_sigma: float = 0.0
    # Gaussian smoothing sigma for image gradients.
    # Can reduce noisy gradient-driven updates.
    gradient_sigma: float = 0.0
    # Initial control point spacing (x). Negative uses spacing in mm convention
    # from NiftyReg defaults.
    spacing_x: float = -5.0
    # Optional anisotropic control point spacing (y). None -> use `spacing_x`.
    spacing_y: Optional[float] = None
    # Optional anisotropic control point spacing (z). None -> use `spacing_x`.
    spacing_z: Optional[float] = None
    # Bending energy regularization weight (smoothness prior).
    # Increase to enforce smoother deformations.
    bending_energy_weight: float = 0.001
    # Linear elasticity-like regularization weight.
    # Increase to penalize local shearing/stretching.
    linear_energy_weight: float = 0.01
    # Jacobian log penalty weight for topology-preserving tendency.
    # Use >0 to discourage foldings (may slow convergence).
    jacobian_log_weight: float = 0.0
    # Use approximate Jacobian log penalty for speed.
    approximate_jacobian_log: bool = True
    # Inverse-consistency penalty weight (forward/backward agreement).
    # Effective only with symmetric optimization context.
    inverse_consistency_weight: float = 0.0
    # Accumulate updates through gradient cumulative exponentiation.
    # Recommended with velocity-based parameterization.
    use_gradient_cumulative_exp: bool = True
    # Number of BCH terms for velocity composition approximation.
    # None lets backend choose default behavior.
    bch_terms: Optional[int] = None
    # Robust intensity range handling (outlier resistant).
    robust_range: bool = False
    # Padding value used for out-of-FOV samples.
    # NaN lets backend treat it as invalid/ignored where supported.
    padding_value: float = float("nan")
    # Clamp lower intensity of reference image before similarity computation.
    # None disables lower threshold.
    reference_lower_threshold: Optional[float] = None
    # Clamp upper intensity of reference image before similarity computation.
    reference_upper_threshold: Optional[float] = None
    # Clamp lower intensity of floating image before similarity computation.
    floating_lower_threshold: Optional[float] = None
    # Clamp upper intensity of floating image before similarity computation.
    floating_upper_threshold: Optional[float] = None
    # Similarity metric: "nmi", "lncc", "ssd", or "mind".
    # Start with "nmi" for multimodal, "ssd"/"lncc" for near-monomodal.
    similarity: str = "nmi"
    # LNCC kernel sigma (used only when `similarity="lncc"`).
    lncc_sigma: float = 2.0
    # MIND neighborhood offset (used only when `similarity="mind"`).
    mind_offset: int = 1
    # Number of histogram bins for reference image NMI.
    # Used only when `similarity="nmi"`.
    nmi_reference_bins: int = 64
    # Number of histogram bins for floating image NMI.
    nmi_floating_bins: int = 64
    # Normalize SSD term by local/global scale (metric-specific behavior).
    # Used only when `similarity="ssd"`.
    ssd_normalize: bool = True
    # Print backend progress/logs.
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

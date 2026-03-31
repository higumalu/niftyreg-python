from __future__ import annotations

import numpy as np
import SimpleITK as sitk

import niftyreg


def test_f3d_returns_cpp(random_pair):
    fixed, moving = random_pair
    result = niftyreg.f3d(
        fixed,
        moving,
        options=niftyreg.F3DOptions(
            use_velocity=False,
            use_symmetric=False,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            padding_value=0.0,
            verbose=False,
        ),
    )
    assert result.warped.GetSize() == fixed.GetSize()
    assert result.cpp.kind == "cpp"
    assert result.cpp.image.GetNumberOfComponentsPerPixel() == 3


def test_f3d2_velocity_returns_backward_cpp(random_pair, mask_image):
    fixed, moving = random_pair
    result = niftyreg.f3d(
        fixed,
        moving,
        reference_mask=mask_image,
        floating_mask=mask_image,
        options=niftyreg.F3DOptions(
            use_velocity=True,
            use_symmetric=True,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            padding_value=0.0,
            verbose=False,
        ),
    )
    assert result.cpp.kind == "spline_velocity"
    assert result.backward_cpp is not None
    assert result.backward_cpp.image.GetSize() == result.cpp.image.GetSize()


def test_f3d_cpp_reconstructs_returned_warped(random_pair):
    fixed, moving = random_pair
    result = niftyreg.f3d(
        fixed,
        moving,
        options=niftyreg.F3DOptions(
            use_velocity=False,
            use_symmetric=False,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            padding_value=0.0,
            verbose=False,
        ),
    )
    warped_from_cpp = niftyreg.resample(
        fixed,
        moving,
        result.cpp,
        interpolation=3,
        padding_value=0.0,
    )

    returned = sitk.GetArrayFromImage(result.warped)
    reconstructed = sitk.GetArrayFromImage(warped_from_cpp)
    assert np.allclose(returned, reconstructed, atol=1e-5)

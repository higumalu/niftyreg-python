from __future__ import annotations

import numpy as np

import niftyreg


def _aladin_transform(fixed, moving):
    return niftyreg.aladin(
        fixed,
        moving,
        options=niftyreg.AladinOptions(
            symmetric=False,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            block_percentage=100,
            inlier_lts=100,
            padding_value=0.0,
        ),
    ).transform


def test_resample_measure_jacobian(random_pair):
    fixed, moving = random_pair
    transform = _aladin_transform(fixed, moving)
    warped = niftyreg.resample(fixed, moving, transform, interpolation=1, padding_value=0.0)
    jac = niftyreg.jacobian(fixed, transform, determinant=True, log_determinant=True, matrix=True)
    scores = niftyreg.measure(fixed, moving, transform=transform, metrics=("nmi", "ssd", "ncc"))

    assert warped.GetSize() == fixed.GetSize()
    assert jac.determinant.GetSize() == fixed.GetSize()
    assert jac.log_determinant.GetSize() == fixed.GetSize()
    assert jac.matrix.GetNumberOfComponentsPerPixel() in (4, 9)
    assert set(scores) == {"nmi", "ssd", "ncc"}


def test_transform_ops_and_average_affines(random_pair):
    fixed, moving = random_pair
    transform = _aladin_transform(fixed, moving)

    deformation = niftyreg.transform_to_deformation(fixed, transform)
    displacement = niftyreg.transform_to_displacement(fixed, transform)
    inverse = niftyreg.invert_transform(fixed, transform)
    composed = niftyreg.compose_transforms(fixed, transform, inverse)
    average = niftyreg.average_affines([transform.affine_lps, np.eye(4)])

    assert deformation.kind == "deformation"
    assert displacement.kind == "displacement"
    assert inverse.kind == "affine"
    assert composed.kind == "deformation"
    assert average.shape == (4, 4)


def test_transform_flow_from_velocity_cpp(random_pair):
    fixed, moving = random_pair
    result = niftyreg.f3d(
        fixed,
        moving,
        options=niftyreg.F3DOptions(
            use_velocity=True,
            use_symmetric=False,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            padding_value=0.0,
            verbose=False,
        ),
    )
    flow = niftyreg.transform_to_flow(fixed, result.cpp)
    jac = niftyreg.jacobian(fixed, result.cpp, determinant=True)
    assert flow.kind == "flow"
    assert flow.image.GetSize() == fixed.GetSize()
    assert jac.determinant.GetSize() == fixed.GetSize()


from __future__ import annotations

import numpy as np
import SimpleITK as sitk

import niftyreg


def main() -> None:
    fixed = sitk.GaussianSource(sitk.sitkFloat32, size=[32, 32, 32], sigma=[5.0, 5.0, 5.0], mean=[15.0, 15.0, 15.0])
    moving = sitk.GaussianSource(sitk.sitkFloat32, size=[32, 32, 32], sigma=[5.0, 5.0, 5.0], mean=[16.0, 14.0, 15.0])

    reg = niftyreg.aladin(
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
    )

    deformation = niftyreg.transform_to_deformation(fixed, reg.transform)
    inverse = niftyreg.invert_transform(fixed, reg.transform)
    composed = niftyreg.compose_transforms(fixed, reg.transform, inverse)
    jac = niftyreg.jacobian(fixed, reg.transform, determinant=True, matrix=True)
    scores = niftyreg.measure(fixed, moving, transform=reg.transform, metrics=("nmi", "ssd", "ncc"))

    print("Deformation kind:", deformation.kind)
    print("Inverse kind:", inverse.kind)
    print("Composed kind:", composed.kind)
    print("Jacobian components:", jac.matrix.GetNumberOfComponentsPerPixel())
    print("Scores:", scores)
    print("Average affine shape:", niftyreg.average_affines([reg.affine_lps, np.eye(4)]).shape)


if __name__ == "__main__":
    main()


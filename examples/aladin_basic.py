from __future__ import annotations

import SimpleITK as sitk

import niftyreg


def main() -> None:
    fixed = sitk.GaussianSource(
        sitk.sitkFloat32,
        size=[48, 48, 40],
        sigma=[7.0, 7.0, 6.0],
        mean=[22.0, 23.0, 19.0],
        scale=100.0,
    )
    moving = sitk.GaussianSource(
        sitk.sitkFloat32,
        size=[48, 48, 40],
        sigma=[7.0, 7.0, 6.0],
        mean=[24.0, 21.0, 19.0],
        scale=100.0,
    )

    result = niftyreg.aladin(
        fixed,
        moving,
        options=niftyreg.AladinOptions(
            symmetric=False,
            number_of_levels=2,
            levels_to_perform=2,
            max_iterations=3,
            block_percentage=100,
            inlier_lts=100,
            padding_value=0.0,
        ),
    )

    sitk.WriteImage(result.warped, "aladin_warped.nii.gz")
    print("Affine in LPS:")
    print(result.affine_lps)


if __name__ == "__main__":
    main()


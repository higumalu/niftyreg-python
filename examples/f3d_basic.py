from __future__ import annotations

import numpy as np
import SimpleITK as sitk

import niftyreg


def main() -> None:
    rng = np.random.default_rng(0)
    fixed = sitk.GetImageFromArray(rng.normal(size=(32, 32, 32)).astype(np.float32))
    moving = sitk.GetImageFromArray(np.roll(sitk.GetArrayFromImage(fixed), shift=1, axis=2))

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

    sitk.WriteImage(result.warped, "f3d_warped.nii.gz")
    sitk.WriteImage(result.cpp.image, "f3d_cpp.nii.gz")
    print("CPP size:", result.cpp.image.GetSize())


if __name__ == "__main__":
    main()


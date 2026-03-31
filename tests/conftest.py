from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk


@pytest.fixture
def random_pair() -> tuple[sitk.Image, sitk.Image]:
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(32, 32, 32)).astype(np.float32)
    fixed = sitk.GetImageFromArray(arr)
    moving = sitk.GetImageFromArray(np.roll(arr, shift=1, axis=2))
    for image in (fixed, moving):
        image.SetSpacing((1.1, 1.2, 1.3))
        image.SetOrigin((2.0, 3.0, 4.0))
        image.SetDirection((1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
    return fixed, moving


@pytest.fixture
def mask_image(random_pair: tuple[sitk.Image, sitk.Image]) -> sitk.Image:
    fixed, _ = random_pair
    mask = sitk.GetImageFromArray(np.ones((32, 32, 32), dtype=np.uint8))
    mask.CopyInformation(fixed)
    return mask


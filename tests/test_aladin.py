from __future__ import annotations

import numpy as np

import niftyreg


def test_aladin_returns_warped_and_affine(random_pair):
    fixed, moving = random_pair
    result = niftyreg.aladin(
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
    assert result.warped.GetSize() == fixed.GetSize()
    assert result.affine_lps.shape == (4, 4)
    assert result.transform.kind == "affine"
    assert np.isfinite(result.affine_lps).all()


def test_aladin_symmetric_masks(random_pair, mask_image):
    fixed, moving = random_pair
    result = niftyreg.aladin(
        fixed,
        moving,
        reference_mask=mask_image,
        floating_mask=mask_image,
        options=niftyreg.AladinOptions(
            symmetric=True,
            number_of_levels=1,
            levels_to_perform=1,
            max_iterations=2,
            block_percentage=100,
            inlier_lts=100,
            padding_value=0.0,
        ),
    )
    assert result.warped.GetSize() == fixed.GetSize()


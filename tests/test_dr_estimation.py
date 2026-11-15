import math

import nibabel as nib
import numpy as np
import pytest

from DR_estimation import calc_dr, get_a  # import must not need command line arguments


@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 2.2),  # pure noise
        (1, 1.8),
        (2.99, 2.2 - 0.4 * 2.99),
        (3, 1),  # threshold is inclusive
        (10, 1),
    ],
)
def test_get_a_follows_the_paper(x, expected):
    assert get_a(1.0, x) == pytest.approx(expected)


def test_get_a_scales_by_sigma():
    assert get_a(2.0, 2.0) == pytest.approx(1.8)  # x = Mn / sigma = 1


def reference_dr(img, mask):
    """The formula of Gabr et al. (2008), written as literally as possible."""
    sigma = img[mask == 0].std()
    total = 0.0
    for m in img.ravel():
        x = m / sigma
        a = 1.0 if x >= 3 else 2.2 - 0.4 * x
        d = m * m - a * sigma * sigma
        total += math.copysign(math.sqrt(abs(d)), d)
    return total / (sigma * math.sqrt(img.size))


def save(tmp_path, data, name, affine=None):
    path = str(tmp_path / name)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)
    return path


@pytest.fixture
def head(tmp_path):
    """Rayleigh background (magnitude noise) around a bright cube; exact float64 round trip."""
    rng = np.random.default_rng(0)
    img = rng.rayleigh(2.0, (10, 10, 10))
    mask = np.zeros(img.shape)
    mask[2:8, 2:8, 2:8] = 1
    img[mask == 1] += 50
    return img, mask


def test_dr_matches_the_paper_formula(tmp_path, head):
    img, mask = head
    dr = calc_dr(save(tmp_path, img, "img.nii"), save(tmp_path, mask, "mask.nii"))
    assert dr == pytest.approx(reference_dr(img, mask), rel=1e-6)


def test_sigma_ignores_head_voxels_zeroed_out_of_the_background(tmp_path):
    # background alternates 1/3 (std 1); a head covering 7/8 of the volume must not dilute it
    img = np.full((8, 8, 8), 100.0)
    mask = np.ones(img.shape)
    img[0, :, :] = 1.0
    img[0, ::2, :] = 3.0
    mask[0, :, :] = 0
    dr = calc_dr(save(tmp_path, img, "img.nii"), save(tmp_path, mask, "mask.nii"))
    assert dr == pytest.approx(reference_dr(img, mask), rel=1e-6)

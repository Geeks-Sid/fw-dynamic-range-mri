# calculation from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4264049/

import logging
import sys

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)


def get_a(sigma, Mn):
    """Noise-correction factor a: 2.2 - 0.4*(Mn/sigma) below Mn/sigma = 3, else 1. Accepts arrays."""
    x = Mn / sigma
    return np.where(x >= 3, 1.0, 2.2 - 0.4 * x)


def calc_dr(img_path, headmask_path):
    log.info('Loading input images')
    img = nib.load(img_path)
    mask = nib.load(headmask_path)
    # a broadcastable mask (e.g. 3D mask, 4D image) would otherwise be applied to the wrong voxels
    if img.shape != mask.shape:
        raise ValueError(f"MR_image shape {img.shape} does not match head_mask shape {mask.shape}")
    if not np.allclose(img.affine, mask.affine, atol=1e-3):
        log.warning("MR_image and head_mask have different affines; check that the mask is aligned")

    img_data = img.get_fdata()  # float64
    mask_data = mask.get_fdata()
    if not (np.isfinite(img_data).all() and np.isfinite(mask_data).all()):
        raise ValueError("MR_image and head_mask must not contain NaN or infinite values")
    if img_data.min() < 0:
        log.warning("MR_image has negative values; a magnitude image is expected")
    if not np.isin(mask_data, (0, 1)).all():
        log.warning("head_mask is not binary; every non-zero voxel is treated as head")

    background = mask_data == 0
    if not background.any():
        raise ValueError("head_mask has no background (0) voxels to estimate the noise from")
    if background.all():
        raise ValueError("head_mask has no head (non-zero) voxels")

    # noise standard deviation, from the background (non-head) voxels only
    sigma = np.std(img_data[background])
    if sigma == 0:
        raise ValueError(
            "the background of MR_image has zero variance (already masked or skull-stripped?); "
            "the dynamic range is undefined"
        )
    N = img_data.size  # total number of image points

    log.info('Calculating the dynamic range...')
    # sgn(Mn^2 - a*sigma^2) * sqrt(|Mn^2 - a*sigma^2|), for every voxel at once
    diff = img_data**2 - get_a(sigma, img_data) * sigma**2
    return np.sum(np.sign(diff) * np.sqrt(np.abs(diff))) / (sigma * np.sqrt(N))


if __name__ == "__main__":
    img_path = sys.argv[1]
    headmask_path = sys.argv[2]

    DR = calc_dr(img_path, headmask_path)
    log.info(f"Estimated dynamic range of the input image: {str(DR)}")
    print(f"Estimated dynamic range of the input image: {str(DR)}")

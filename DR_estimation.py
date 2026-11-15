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
    img_data = nib.load(img_path).get_fdata()  # float64
    mask_data = nib.load(headmask_path).get_fdata()

    # noise standard deviation, from the background (non-head) voxels only
    sigma = np.std(img_data[np.logical_not(mask_data)])
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

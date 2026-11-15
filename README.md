# Flywheel gear to calculate the dynamic range of an MR image

This gear estimates the dynamic range of a magnitude image using the calculation from [Gabr et al. (2008)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4264049/)

The noise standard deviation (sigma) is the standard deviation of the voxels
outside the head mask, taken as they are in the image (no correction for the
Rayleigh distribution of magnitude noise). The sum over all voxels (N = every
voxel of the image, 4D images included) is evaluated as in the paper.

### Inputs

* MR_image (nifti): image to calculate the DR of
* head_mask (nifti): binary mask of the head (1) vs background (0)

`MR_image` and `head_mask` must have the same shape. Any non-zero mask voxel is
treated as head.

### Configuration

* __debug__ (boolean, default False): Include debug statements in output.

### Output

`dynamic_range.json` in the gear output:

```json
{
  "dynamic_range": 311.12,
  "noise_sigma": 1.28,
  "n_voxels": 1000,
  "n_background_voxels": 784,
  "mr_image": "image.nii.gz",
  "head_mask": "mask.nii.gz",
  "gear_name": "d3b-dynamic-range-mri",
  "gear_version": "0.0.2"
}
```

`dynamic_range` is a linear ratio, not dB. The other fields record what the
number was computed from, so it can be checked later. The value is also logged.

### Failures and warnings

The gear fails (non-zero exit, nothing written) when the shapes differ, the
image or mask contains NaN/inf, the mask has no background or no head voxels,
or the background has zero variance (for example a skull-stripped image whose
background was set to 0).

It logs a warning, and continues, when the affines of the image and mask
differ, the image has negative values, or the mask is not binary.

### Versions

Before 0.0.2 sigma was computed over the whole image with the head voxels set
to 0, which underestimated sigma and overestimated the dynamic range. Results
from 0.0.1 are not comparable with later versions.

## Development

```
pip install -r requirements-dev.txt
pytest
```

The tests use small synthetic images; they do not need a gear environment.

import logging
from types import SimpleNamespace as NS

import nibabel as nib
import numpy as np
import pytest

from fw_dynamic_range_mri.main import run  # import must not need a gear config.json


@pytest.fixture
def ctx(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.rayleigh(2.0, (10, 10, 10))
    mask = np.zeros(img.shape)
    mask[2:8, 2:8, 2:8] = 1
    img[mask == 1] += 50
    paths = {}
    for name, data in {"MR_image": img, "head_mask": mask}.items():
        paths[name] = str(tmp_path / f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(data, np.eye(4)), paths[name])
    # a KeyError here means run() asked for an input the manifest does not declare
    return NS(get_input_path=paths.__getitem__, paths=paths)


def test_run_reports_the_dynamic_range(ctx, caplog):
    with caplog.at_level(logging.INFO):
        assert run(ctx) == 0
    assert "Estimated dynamic range of the input image:" in caplog.text


def test_run_fails_loudly_on_unusable_input(ctx):
    # the exit status of the gear must reflect the failure
    nib.save(nib.Nifti1Image(np.ones((10, 10, 10)), np.eye(4)), ctx.paths["head_mask"])  # no background
    with pytest.raises(ValueError, match="no background"):
        run(ctx)

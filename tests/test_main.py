import json
import logging
from pathlib import Path
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
    out = tmp_path / "output"
    out.mkdir()
    # a KeyError here means run() asked for an input the manifest does not declare
    return NS(
        get_input_path=paths.__getitem__,
        get_input_filename=lambda name: Path(paths[name]).name,
        output_dir=out,
        manifest={"name": "d3b-dynamic-range-mri", "version": "9.9.9"},
        paths=paths,
    )


def test_run_reports_the_dynamic_range(ctx, caplog):
    with caplog.at_level(logging.INFO):
        assert run(ctx) == 0
    assert "Estimated dynamic range of the input image:" in caplog.text


def test_run_writes_the_result_with_its_provenance(ctx):
    run(ctx)
    result = json.loads((ctx.output_dir / "dynamic_range.json").read_text())
    assert result.pop("dynamic_range") > 0
    assert result.pop("noise_sigma") > 0
    assert result == {
        "n_voxels": 1000,
        "n_background_voxels": 1000 - 6**3,
        "mr_image": "MR_image.nii.gz",
        "head_mask": "head_mask.nii.gz",
        "gear_name": "d3b-dynamic-range-mri",
        "gear_version": "9.9.9",
    }


def test_run_fails_loudly_on_unusable_input(ctx):
    # the exit status of the gear must reflect the failure, and no result may be left behind
    nib.save(nib.Nifti1Image(np.ones((10, 10, 10)), np.eye(4)), ctx.paths["head_mask"])  # no background
    with pytest.raises(ValueError, match="no background"):
        run(ctx)
    assert not list(ctx.output_dir.iterdir())

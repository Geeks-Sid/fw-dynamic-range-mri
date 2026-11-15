import json
import logging
from pathlib import Path

from flywheel_gear_toolkit import GearToolkitContext

from .dr_estimation import calc_dr

log = logging.getLogger(__name__)

RESULT_FILENAME = "dynamic_range.json"


def run(gtk_context: GearToolkitContext) -> int:
    """Main entrypoint

    Args:
        gtk_context (GearToolkitContext)
    """
    result = calc_dr(
        gtk_context.get_input_path("MR_image"),
        gtk_context.get_input_path("head_mask"),
    )
    log.info(f"Estimated dynamic range of the input image: {result['dynamic_range']}")

    # keep the result with what produced it, so it can be audited without the job log
    result.update(
        mr_image=gtk_context.get_input_filename("MR_image"),
        head_mask=gtk_context.get_input_filename("head_mask"),
        gear_name=gtk_context.manifest["name"],
        gear_version=gtk_context.manifest["version"],
    )
    (Path(gtk_context.output_dir) / RESULT_FILENAME).write_text(json.dumps(result, indent=2))
    return 0

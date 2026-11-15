import logging

from flywheel_gear_toolkit import GearToolkitContext

from .dr_estimation import calc_dr

log = logging.getLogger(__name__)


def run(gtk_context: GearToolkitContext) -> int:
    """Main entrypoint

    Args:
        gtk_context (GearToolkitContext)
    """
    dr = calc_dr(
        gtk_context.get_input_path("MR_image"),
        gtk_context.get_input_path("head_mask"),
    )
    log.info(f"Estimated dynamic range of the input image: {dr}")
    return 0

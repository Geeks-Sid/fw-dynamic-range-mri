#!/usr/bin/env python
"""The run script."""
import sys

from flywheel_gear_toolkit import GearToolkitContext

from fw_dynamic_range_mri.main import run

if __name__ == "__main__":  # pragma: no cover
    with GearToolkitContext() as gear_context:
        gear_context.init_logging()  # honors the 'debug' config option
        sys.exit(run(gear_context))

"""Start the Ramp Polarimeter: Run this file in VS Code (the Run button, or the
"Ramp Polarimeter" launch configuration), or `python polarimeter.py`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampol.gui import main  # noqa: E402

main()

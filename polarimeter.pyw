"""Double-click (or `pythonw polarimeter.pyw`) to start the Ramp Polarimeter."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rampol.gui import main  # noqa: E402

main()

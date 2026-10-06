#!/usr/bin/env python3
"""Measured polarization against the ILC's original target, and the target
correction for the EOM-ILC panel's Corrections tab.

    python tools/target_compare.py SCAN_FOLDER --x1 EOM-ILC/run/drive_P92PX1H.state.npz
                                               --x2 EOM-ILC/run/drive_P92PX2A.state.npz

The work is rampol.ilc_target.compare (the GUI's 'ILC target' button runs the
same thing); its docstring explains the timing map, why the correction is
minus (light - monitors) and not minus (light - target), and the line-ripple
removal. Outputs go to SCAN_FOLDER/analysis/target_compare/.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import ilc_target  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scan")
    ap.add_argument("--x1", required=True, help="ILC state for EO1 (drive_*.state.npz)")
    ap.add_argument("--x2", required=True, help="ILC state for EO2")
    ap.add_argument("--f-cut", type=float, default=2000.0,
                    help="correction bandwidth, Hz (default 2000: below the 4.94 kHz "
                         "motional resonance the ILC's own error was found to heat)")
    ap.add_argument("--lock-tol", type=float, default=0.006)
    ap.add_argument("--leg-gap-ms", type=float, default=16.667)
    ap.add_argument("--pd-delay-us", type=float, default=0.0,
                    help="the photodiode chain's own delay (e.g. an anti-alias RC: "
                         "3.1 us), taken out of the light before it is compared")
    ap.add_argument("--line-hz", type=float, default=60.0)
    ap.add_argument("--line-harmonics", type=int, default=3)
    ap.add_argument("--split", type=float, default=0.5,
                    help="fraction of the rotation correction put on X1 (rest on X2)")
    a = ap.parse_args(argv)
    return ilc_target.compare(a.scan, a.x1, a.x2, f_cut=a.f_cut, lock_tol=a.lock_tol,
                              leg_gap_ms=a.leg_gap_ms, pd_delay_us=a.pd_delay_us,
                              line_f=a.line_hz, line_h=a.line_harmonics, split=a.split)


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k != "line_ripple"}, indent=1))

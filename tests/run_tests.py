"""Run every test. No instrument, no pytest.

    python tests/run_tests.py

test_gui.py needs a desktop session (it builds the Tk window, off screen).
Everything reads Scope Grab from config.DEFAULTS["scope_grab_path"]
(scope-grab-multi), which must be at dcf2c0b or later.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_ell14.py", "test_analysis.py", "test_checks.py", "test_bias.py",
          "test_gui.py"]


def main():
    failed = []
    for name in SUITES:
        print(f"{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        if subprocess.run([sys.executable, os.path.join(HERE, name)]).returncode:
            failed.append(name)
        print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(SUITES)} suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

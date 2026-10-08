"""Run every test. No instrument, no pytest.

    python tests/run_tests.py              every suite, side by side
    python tests/run_tests.py gui awg      only these (test_<name>.py)
    python tests/run_tests.py -q           only the FAIL lines and the times
    python tests/run_tests.py --serial     one after another, output as it comes

The suites run in parallel (each works in its own temp folder; test_gui
sandboxes the config), so a full run takes about as long as test_gui alone
(~3.5 min on the lab PC, 7 Oct 2026) instead of the ~5 min of the sum.

test_gui.py needs a desktop session (it builds the Tk window, off screen).
Everything reads Scope Grab from config.DEFAULTS["scope_grab_path"]
(scope-grab-multi), which must be at dcf2c0b or later.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_ell14.py", "test_analysis.py", "test_checks.py", "test_bias.py",
          "test_awg.py", "test_gui.py"]


def main(argv):
    quiet = "-q" in argv
    serial = "--serial" in argv
    names = [a for a in argv if not a.startswith("-")]
    suites = [s for s in SUITES if not names or s[5:-3] in names or s in names]
    if not suites:
        print(f"no suite matches {names}; the suites: {', '.join(s[5:-3] for s in SUITES)}")
        return 2
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed, took = [], {}
    if serial:
        for name in suites:
            print(f"{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
            t0 = time.time()
            if subprocess.run([sys.executable, os.path.join(HERE, name)], env=env).returncode:
                failed.append(name)
            took[name] = time.time() - t0
            print()
    else:
        import tempfile
        t0 = time.time()
        # each into a file of its own: a pipe that fills would stall a suite
        # until its turn to be read, and the runs would no longer overlap
        files = {name: tempfile.TemporaryFile() for name in suites}
        procs = {name: subprocess.Popen([sys.executable, os.path.join(HERE, name)], env=env,
                                        stdout=files[name], stderr=subprocess.STDOUT)
                 for name in suites}
        outs = {}
        left = dict(procs)
        while left:
            for name, p in list(left.items()):
                if p.poll() is not None:
                    took[name] = time.time() - t0
                    if p.returncode:
                        failed.append(name)
                    files[name].seek(0)
                    outs[name] = files[name].read().decode("utf-8", "replace")
                    files[name].close()
                    del left[name]
            time.sleep(0.2)
        for name in suites:
            lines = outs[name].splitlines()
            if quiet:
                lines = [ln for ln in lines if "FAIL" in ln or "Error" in ln
                         or ln.startswith("Traceback")]
                if not lines:
                    continue
            print(f"{'=' * 70}\n{name}\n{'=' * 70}")
            print("\n".join(lines))
            print()
    print("times: " + ", ".join(f"{n[5:-3]} {took[n]:.0f} s" for n in suites))
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(suites)} suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

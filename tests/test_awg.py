"""The AWG mode without an instrument: waveforms, checks, ILC drive files and
the output bookkeeping, against the simulator's FakeAWG.

    python tests/test_awg.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import awg, bias, config, hw, sim  # noqa: E402

FAILS = []
ILC = config.DEFAULTS["eomilc_path"]


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return None


def wave_checks(eom):
    print("\nramp to a rotation and back")
    idle = {"EO1": 0.026, "EO2": 0.078}
    w = awg.ramp_hold(45.0, {}, idle=idle)
    check("the ILC's record: 5501 points at 2 us, FRQ 90.893 Hz",
          w.n == 5501 and abs(1 / w.period - 90.893) < 1e-3, f"{w.n}, {1/w.period:.4f} Hz")
    check("both ends exactly at idle", all(w.u[k][0] == idle[k] == w.u[k][-1] for k in idle))
    _, rot = awg.predict(w)
    m = (w.t > w.hold[0]) & (w.t < w.hold[1])
    check("the hold is at the commanded rotation", np.allclose(rot[m], 45.0, atol=1e-9),
          f"{rot[m].mean():.6f} deg")
    lin = awg.ramp_hold(45.0, {"edge": "linear"}, idle=idle)
    k = int(w.hold[0] / w.dt) - 375          # a quarter into the rise (mid-rise both are 0.5)
    check("linear edges differ from cosine a quarter into the rise",
          abs(lin.u["EO1"][k] - w.u["EO1"][k]) > 0.01)
    check("split outside 0..1 refused", raises(awg.ramp_hold, 45.0, {"split": 1.5}) is not None)
    check("segments longer than the record refused",
          "does not fit" in (raises(awg.ramp_hold, 45.0, {"hold_ms": 20.0}) or ""))

    print("\nthe checks")
    f = awg.check(w, eom, trig_hz=3.7)
    check("45 deg ramp at 3.7 Hz passes", awg.worst(f) != "FAIL", [m for lv, m in f if lv != "INFO"])
    big = awg.ramp_hold(45.0, {"record_ms": 40.0, "dt_us": 2.0})
    check("over 16384 points fails", any(lv == "FAIL" and "points" in m
                                          for lv, m in awg.check(big, None)))
    check("a record over 80 % of the trigger period fails",
          any(lv == "FAIL" and "trigger" in m for lv, m in awg.check(w, None, trig_hz=100)))
    hot = awg.ramp_hold(45.0, {"hold_ms": 60.0, "record_ms": 70.0, "dt_us": 20.0})
    check("long kV holds warn about the duty",
          any(lv == "WARN" and "duty" in m for lv, m in awg.check(hot, None, trig_hz=3.7)))
    off = awg.ramp_hold(45.0, {}, idle={"EO1": 0.2, "EO2": 0.0})
    check("an idle past 100 mV fails",
          any(lv == "FAIL" and "idle cap" in m for lv, m in awg.check(off, None)))
    over = awg.ramp_hold(200.0, {"split": 1.0})
    check("past the 9.6 V cap fails", any(lv == "FAIL" and "cap" in m for lv, m in awg.check(over, None)))
    n1 = awg.wave_name(w.u["EO1"], 1)
    check("names: 11 characters, the same samples give the same name",
          len(n1) == 11 and n1 == awg.wave_name(w.u["EO1"].copy(), 1)
          and n1 != awg.wave_name(lin.u["EO1"], 1), n1)


def file_checks(eom):
    print("\nILC drive files")
    tmp = tempfile.mkdtemp(prefix="rampol-awg-")
    bare = os.path.join(tmp, "bare.csv")
    with open(bare, "w") as fh:
        fh.write("0.1\n0.2\n")
    check("a header-less file is refused", "header" in (raises(awg.load_drive, bare) or ""))
    tgt = os.path.join(ILC, "waveforms")
    tfiles = [os.path.join(tgt, f) for f in os.listdir(tgt)] if os.path.isdir(tgt) else []
    tfiles = [p for p in tfiles if os.path.basename(p).startswith("target_") and p.endswith(".csv")]
    if tfiles:
        msg = raises(awg.load_drive, tfiles[0]) or ""
        check("a TARGET file (EOM volts) is refused as a drive", "TARGET" in msg, msg[:80])
    run = os.path.join(ILC, "run")
    p1, p2 = (os.path.join(run, "drive_S92PX1B_i15.csv"),
              os.path.join(run, "drive_S92PX2B_i13.csv"))
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("  (EOM-ILC keeper drives not found - skipped)")
        return
    w = awg.from_files(p1, p2)
    check("two keepers: same grid, idle trims kept (not zeroed)",
          w.n == 5501 and w.u["EO1"][0] > 0.01 and w.u["EO2"][0] > 0.05,
          f"idle {w.u['EO1'][0]*1e3:.1f} / {w.u['EO2'][0]*1e3:.1f} mV")
    if eom is None:
        return
    check("their targets found beside them", set(w.target) == {"EO1", "EO2"}, list(w.target))
    f = awg.check(w, eom, trig_hz=3.7)
    check("checked against their own targets they pass", awg.worst(f) != "FAIL",
          [m for lv, m in f if lv == "FAIL"])
    w.target = {}
    f = awg.check(w, eom, trig_hz=3.7)
    check("checked as u x gain the X1 keeper fails the current limit (why targets are used)",
          any(lv == "FAIL" and "EO1" in m for lv, m in f),
          [m[:70] for lv, m in f if lv == "FAIL"])


class FlakyAWG(sim.FakeAWG):
    """CH2 refuses to switch on."""
    def set_output(self, ch, on):
        if ch == 2 and on:
            raise RuntimeError("CH2 refused")
        super().set_output(ch, on)


def session_checks():
    print("\nthe session: outputs, names, park")
    bench = sim.Bench(seed=1)
    a = sim.FakeAWG(bench)
    s = awg.Session(a, None, log=lambda *_: None)
    w = awg.ramp_hold(30.0, {}, idle={"EO1": 0.0, "EO2": 0.0})
    check("ON before anything is loaded is refused", raises(s.on) is not None)
    s.load(w)
    n_stored = len(a.stored)
    s.load(w)
    check("the same waveform again is selected, not stored again",
          len(a.stored) == n_stored == 2, len(a.stored))
    s.on()
    check("both outputs on and owned", all(bench.awg_on.values()) and s.owned == {1, 2})
    w2 = awg.ramp_hold(60.0, {})
    s.load(w2)
    check("a change under the 'off' policy switches the outputs off first",
          not any(bench.awg_on.values()) and not s.owned)
    bench.awg_on[1] = True                     # someone else's
    check("a channel ON that this session did not switch on is refused",
          "did not" in (raises(s.load, w) or ""))
    bench.awg_on[1] = False
    s.on()
    s.end("park")
    flat = bench.awg_drive[1][1]
    check("park: idle waveform, outputs left ON",
          all(bench.awg_on.values()) and np.ptp(flat) == 0 and s.parked)
    s.off()
    check("off: both off", not any(bench.awg_on.values()))
    s.load(awg.ramp_hold(30.0, {"record_ms": 12.0}), keep_on=False)
    s.on()
    msg = raises(s.load, awg.ramp_hold(30.0, {"record_ms": 14.0}), keep_on=True)
    check("a live record-length change (FRQ) is refused under park",
          msg is not None and "FRQ" in msg, (msg or "")[:70])
    s.off()
    fb = sim.Bench(seed=2)
    fs = awg.Session(FlakyAWG(fb), None, log=lambda *_: None)
    fs.load(w)
    check("CH2 refusing to switch on raises", raises(fs.on) is not None)
    check("and CH1 is not left on", not any(fb.awg_on.values()) and not fs.owned,
          fb.awg_on)


def bias_checks():
    print("\nbias plateaus with an idle trim")
    p = dict(bias.PLAN, idle={"EO1": 0.026, "EO2": 0.078})
    t, u = bias.plateau(4.0, p, 0.026)
    check("ends at idle, holds idle + amplitude, 5501 points",
          u[0] == u[-1] == 0.026 and np.isclose(u.max(), 4.026) and len(u) == 5501)
    check("an idle past the cap is refused by the bias plan",
          raises(bias.check_plateaus, [30], dict(bias.PLAN, idle={"EO1": 0.2})) is not None)


def main():
    eom = None
    try:
        eom = hw.load_eomilc(ILC)
    except Exception as exc:
        print(f"(EOM-ILC not loaded: {exc} - limit checks skipped)")
    wave_checks(eom)
    file_checks(eom)
    session_checks()
    bias_checks()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("AWG OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

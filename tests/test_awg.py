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

from rampol import awg, bias, calib, config, hw, sim  # noqa: E402

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
    long_ = awg.ramp_hold(45.0, {"hold_ms": 20.0, "tail_ms": 2.0})
    check("the record is lead + rise + hold + fall + after (24.5 ms -> 12251 points)",
          abs(long_.period - 24.502e-3) < 1e-9 and long_.n == 12251
          and abs(awg.record_ms({"hold_ms": 20.0, "tail_ms": 2.0}) - 24.5) < 1e-9,
          f"{long_.period*1e3:.3f} ms, {long_.n} points")
    check("lead and after must be > 0 (both ends at idle)",
          raises(awg.ramp_hold, 45.0, {"tail_ms": 0.0}) is not None)

    print("\nthe checks")
    f = awg.check(w, eom, trig_hz=3.7)
    check("45 deg ramp at 3.7 Hz passes", awg.worst(f) != "FAIL", [m for lv, m in f if lv != "INFO"])
    big = awg.ramp_hold(45.0, {"hold_ms": 37.5, "dt_us": 2.0})
    check("over 16384 points fails", any(lv == "FAIL" and "points" in m
                                          for lv, m in awg.check(big, None)))
    check("a record over 80 % of the trigger period fails",
          any(lv == "FAIL" and "trigger" in m for lv, m in awg.check(w, None, trig_hz=100)))
    hot = awg.ramp_hold(45.0, {"hold_ms": 60.0, "dt_us": 20.0})
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
    # the mechanics first, with both rules off
    s = awg.Session(a, None, log=lambda *_: None, never_float=False, require_dry_run=False)
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
    s.load(awg.ramp_hold(30.0, {"tail_ms": 1.5}), keep_on=False)
    s.on()
    seen = []
    put0 = s._put

    def spy(ch, u):
        seen.append((ch, float(np.ptp(u)), len(u), round(1 / s.awg.frq[ch], 9)))
        return put0(ch, u)
    s._put = spy
    w14 = awg.ramp_hold(30.0, {"tail_ms": 3.5})
    s.load(w14, keep_on=True)
    s._put = put0
    first = seen[:2]
    check("a live record-length change: idle on the OLD grid first, then the new record",
          all(ptp == 0 and n == 6001 and abs(per - 12.002e-3) < 1e-9
              for _ch, ptp, n, per in first)
          and all(n == 7001 for _ch, _p, n, _per in seen[2:]) and len(seen) == 4
          and all(bench.awg_on.values()), seen)
    check("the AWG plays the new record at its own FRQ",
          abs(1 / s.awg.frq[1] - w14.period) < 1e-12 and abs(1 / s.awg.frq[2] - w14.period) < 1e-12)
    s.park()
    check("park goes back to the ILC's 11 ms record (its FRQ check passes after)",
          s.parked and abs(s.wave.period - 11.002e-3) < 1e-9 and s.wave.n == 5501
          and abs(1 / s.awg.frq[1] - s.wave.period) < 1e-12, f"{s.wave.period*1e3:.3f} ms")
    s.off()
    fb = sim.Bench(seed=2)
    fs = awg.Session(FlakyAWG(fb), None, log=lambda *_: None, never_float=False,
                     require_dry_run=False)
    fs.load(w)
    check("CH2 refusing to switch on raises", raises(fs.on) is not None)
    check("and CH1 is not left on", not any(fb.awg_on.values()) and not fs.owned,
          fb.awg_on)


def rule_checks():
    print("\nnever float, and a dry run before anything drives the Treks")
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    scope, _ell, bench = sim.make(sg, roles={1: "PD", 3: "MonX1", 4: "MonX2"})
    link = hw.ScopeLink(scope, log=lambda *a: None)
    s = awg.Session(sim.FakeAWG(bench), None, log=lambda *a: None)
    check("the rules are on by default", s.never_float and s.require_dry_run)
    w = awg.ramp_hold(45.0, {}, idle={"EO1": 0.026, "EO2": 0.078})
    s.load(w)
    msg = raises(s.on)
    check("ON refused for a waveform that has not passed a dry run",
          msg is not None and "dry run" in msg, (msg or "")[:70])
    check("OFF refused without force under the rule", raises(s.off) is not None)
    bench.wiring, bench.awg_scope = "scope", {1: 3, 2: 4}
    tb0 = scope.get(":TIMebase:SCALe")
    v0 = link.channel_state([3, 4])
    rep = awg.dry_run(s, link, w, {"EO1": 3, "EO2": 4}, shots=3, log=lambda *a: None)
    b = rep["steps"]["both"]
    check("dry run passes: gain, delay, time scale and shape as meant", rep["ok"],
          rep["problems"][:2])
    check("it sees the generator's zero-code error at idle (-12 / -40 mV)",
          abs(b["EO1"]["idle_meas_V"] - (0.026 - 0.012)) < 0.004
          and abs(b["EO2"]["idle_meas_V"] - (0.078 - 0.040)) < 0.004,
          f"{b['EO1']['idle_meas_V']*1e3:.1f} / {b['EO2']['idle_meas_V']*1e3:.1f} mV")
    check("the scope is put back", scope.get(":TIMebase:SCALe") == tb0
          and link.channel_state([3, 4]) == v0)
    check("the waveform is now verified", s.is_verified(w))
    s.end()
    check("the end under the rule is park: idle waveform, outputs ON",
          all(bench.awg_on.values()) and s.parked and np.ptp(bench.awg_drive[1][1]) == 0)
    bench.wiring = "treks"
    s.load(w)
    s.on()
    check("a verified waveform goes onto the live outputs and ON", all(bench.awg_on.values())
          and not s.parked)
    w2 = awg.ramp_hold(60.0, {})
    check("an unverified one is refused on live outputs",
          "dry run" in (raises(s.load, w2) or ""))
    bench.wiring, bench.awg_scope = "scope", {1: 4, 2: 3}
    rep = awg.dry_run(s, link, w2, {"EO1": 3, "EO2": 4}, shots=2, log=lambda *a: None)
    check("swapped cabling fails the dry run and says so",
          not rep["ok"] and "swapped" in " ".join(rep["problems"]), rep["problems"][:1])
    check("and the waveform stays unverified", not s.is_verified(w2))
    bench.awg_scope = {1: 3, 2: 4}
    s.off(force=True)
    s._period = {}
    w12 = awg.ramp_hold(45.0, {"tail_ms": 1.5})
    s.load(w12)
    bench.awg_drive = {ch: (w.period, d) for ch, (_p, d) in bench.awg_drive.items()}
    s.awg.frq = {1: 1 / w.period, 2: 1 / w.period}
    s.dry = True
    s.on()
    s.dry = False
    rep = awg.dry_run(s, link, w12, {"EO1": 3, "EO2": 4}, shots=2, log=lambda *a: None,
                      identify=False)
    check("a record played at the wrong FRQ fails (time scale)",
          not rep["ok"] and any("as long as meant" in x for x in rep["problems"]),
          rep["problems"][:1])
    s.off(force=True)


def calib_checks():
    print("\nthe EOM calibration")
    cal = calib.get({})
    check("defaults: 17.55 deg per monitor V on EO1",
          abs(calib.deg_per_mon_v(cal, "EO1") - 17.550) < 0.002)
    r = calib.convert(cal, "EO1", deg=45.0)
    check("45 deg on EO1 alone = 9.168 / 2 x ... AWG V round trip",
          abs(calib.convert(cal, "EO1", awg=r["awg"])["deg"] - 45.0) < 1e-9
          and abs(r["kv"] - 5.1283 / 2) < 1e-6, f"{r['awg']:.4f} V AWG, {r['kv']:.4f} kV")
    cfg = {"analysis": {}}
    new = calib.get({"calibration": {"EO1": {"gain": 0.60}}})
    calib.apply(new, cfg)
    v = bias.awg_volts(90, 1.0)
    check("applied: the AWG waveforms use it", abs(v["EO1"] - 5.1283 / 0.60) < 1e-6,
          f"{v['EO1']:.4f} V")
    check("applied: the analysis' degrees per monitor volt follow",
          abs(cfg["analysis"]["deg_per_mon_v"]["MonX1"] - 90 / 5.1283) < 1e-9)
    calib.apply(calib.get({}), {"analysis": {}})
    check("a nonsense value is refused",
          raises(calib.validate, calib.get({"calibration": {"EO2": {"v90_kv": -1}}})) is not None)
    man = {"name": "fake", "transfer": {"gain": 1.01, "rms_resid": 0.01},
           "points": [{"awg": {"EO1": a1, "EO2": a1 * 0.95},
                       "mon_V": {"MonX1": 0.57 * a1, "MonX2": 0.58 * a1 * 0.95}}
                      for a1 in (0.0, 2.0, 4.0, 6.0)]}
    fit, rep = calib.from_bias(man, calib.get({}))
    check("fit to a bias run: gains from monitor vs AWG, V90 scaled by the light",
          abs(fit["EO1"]["gain"] - 0.57) < 1e-9 and abs(fit["EO2"]["gain"] - 0.58) < 1e-9
          and abs(fit["EO1"]["v90_kv"] - 5.1283 / 1.01) < 1e-9, rep[0])


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
    rule_checks()
    calib_checks()
    bias_checks()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("AWG OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

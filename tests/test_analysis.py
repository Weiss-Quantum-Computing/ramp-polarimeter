"""The analysis against data with known answers - synthetic arrays first (no
files), then a full simulated scan written to disk and read back.

    python tests/test_analysis.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import analysis as an, config, hw, scan, sim  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def malus(theta_deg, psi_deg, imax, er, dark=0.0):
    d = np.deg2rad(np.subtract.outer(theta_deg, psi_deg))
    return dark + imax * (np.cos(d) ** 2 + np.sin(d) ** 2 / er)


def harmonic_checks():
    print("\nper-sample harmonic fit, noise-free")
    th = np.arange(0, 360, 10.0)
    psi = np.linspace(-80, 170, 500)
    er = np.full(500, 1000.0)
    I = malus(th, psi, 4.0, er)
    f = an.harmonic_fit(th, I)
    dpsi = (f["psi"] - psi + 90) % 180 - 90
    check("azimuth exact mod 180", np.max(np.abs(dpsi)) < 1e-9, f"{np.max(np.abs(dpsi)):.2e}")
    check("Imax and ER exact", np.allclose(f["imax"], 4.0) and np.allclose(f["er"], 1000.0))
    check("diagnostic terms fitted and zero for pure Malus",
          "c1" in f and np.max(np.abs(f["c4"])) < 1e-9 and np.max(np.abs(f["c1"])) < 1e-9)
    print("\nwith noise: errors match the stated sigma")
    rng = np.random.default_rng(0)
    psi0 = np.full(4000, 30.0)
    I = malus(th, psi0, 4.0, np.full(4000, 1000.0)) + rng.normal(0, 2e-3, (len(th), 4000))
    f = an.harmonic_fit(th, I)
    z_psi = ((f["psi"] - 30.0 + 90) % 180 - 90) / f["sig_psi"]
    z_imin = (f["imin"] - 4e-3) / f["sig_imin"]
    check("psi pull has unit spread", 0.85 < np.std(z_psi) < 1.15, f"{np.std(z_psi):.3f}")
    check("Imin pull has unit spread", 0.85 < np.std(z_imin) < 1.15, f"{np.std(z_imin):.3f}")
    print("\nlower bound when Imin is unresolved")
    I = malus(th, psi0[:200], 4.0, np.full(200, 1e7)) + rng.normal(0, 2e-3, (len(th), 200))
    f = an.harmonic_fit(th, I)
    check("ER flagged as a lower bound", f["er_lower"].mean() > 0.9, f"{f['er_lower'].mean():.2f}")
    check("unwrap makes a 0 -> 180 sweep continuous",
          np.allclose(an.unwrap_psi(np.r_[np.linspace(0, 89, 50), np.linspace(-90, 0, 50)])[-1], 180))


def sim_scan(tmp, sg, **bench_kw):
    bench = sim.Bench(seed=3, **bench_kw)
    scope, ell, bench = sim.make(sg, bench=bench)
    link, rot = hw.ScopeLink(scope, log=lambda *_: None), hw.Rotator(ell, log=lambda *_: None)
    chans = {1: ("PD", "Analyzer PD"), 3: ("MonX1", "X1"), 4: ("MonX2", "X2")}
    run = scan.ScanRun(tmp, "t", sg, link, rot, chans, log=lambda *_: None,
                       clock=lambda: bench.clock)
    plan = dict(config.DEFAULTS["scan"], mode="average", shots=64, blocks=4)
    angles = scan.ordered(scan.angle_list(0, 355, 5), "shuffled", seed=2)
    run.new(plan, [{"kind": "dark", "target": 0}] + scan.build_steps(angles, 8, 45))
    imax, bench.imax = bench.imax, 0.0
    run.run(kinds={"dark"})
    bench.imax = imax
    run.run()
    return run, bench


def scan_checks(sg):
    print("\nsimulated scan, 72 angles every 5 deg, shuffled, written and read back")
    tmp = tempfile.mkdtemp(prefix="rampol-an-")
    run, bench = sim_scan(tmp, sg, drift=0.01, drift_period_s=400)
    d = an.load_scan(run.folder, sg.load_capture)
    names = os.listdir(run.folder)
    check("files are named for Scope Grab's compare box",
          any(n.startswith("t_a045.00_001.") for n in names)
          and sg.split_capture_name("t_a045.00_003.npz") == ("t_a045.00", "003"))
    pol = an.polarization(d)
    true = bench.rotation(d.t)
    err = pol["rotation"] - true
    check("rotation within 0.02 deg rms", np.std(err) < 0.02, f"{np.std(err) * 1e3:.1f} mdeg")
    check("rest azimuth = mount offset of the rest polarization",
          abs(pol["psi_rest"] - bench.mount_of_rest_pol) < 0.05, f"{pol['psi_rest']:.3f}")
    # 1 % intensity drift across a shuffled scan leaks into the angle fit as
    # scatter between angles; the fitted Imin at rest is where it shows
    pol_nd = an.polarization(d, correct_drift=False)
    rest = an.rest_index(d.t)
    imin_true = float(np.mean(bench.imax / (1 / (1 / bench.er(0) + 1 / bench.er_pol))))
    e_on = abs(np.mean(pol["imin"][rest]) - imin_true)
    e_off = abs(np.mean(pol_nd["imin"][rest]) - imin_true)
    check("drift correction from the ref returns brings rest Imin closer to the model",
          e_on < e_off, f"{e_on * 1e3:.3f} vs {e_off * 1e3:.3f} mV uncorrected "
          f"(model {imin_true * 1e3:.2f} mV)")
    dips = an.dip_er(pol, polarizer_er=1e4)
    check("two dips per analyzer angle (up and down ramp)", len(dips) >= 130, len(dips))
    erl = 1 / (1 / bench.er(true) + 1 / bench.er_pol)
    rel = [abs(x["er"] / np.interp(x["t"], d.t, erl) - 1) for x in dips if not x["er_lower"]]
    check("dip ER within 15 % of the model (median)", np.median(rel) < 0.15,
          f"median {np.median(rel) * 100:.1f} %, 90th pct {np.percentile(rel, 90) * 100:.0f} %")
    segs = [s["kind"] for s in an.segments(d.t, pol["rotation"])]
    check("segments rest/up/hold/down/after", segs == ["rest", "up", "hold", "down", "after"], segs)
    mon = an.monitor_prediction(d, pol, config.DEG_PER_MON_V)
    check("monitor prediction tracks the light", mon is not None and np.std(mon[1]) < 0.05,
          f"{np.std(mon[1]) * 1e3:.1f} mdeg rms")
    check("windows parse", an.parse_windows("-1.5-0.5, 4.8-5.6") == [(-1.5e-3, 0.5e-3), (4.8e-3, 5.6e-3)])


def no_light_checks(sg):
    print("\nno light (the 5 Oct dry run): nothing to report, and it says so")
    tmp = tempfile.mkdtemp(prefix="rampol-dark-")
    run, bench = sim_scan(tmp, sg, imax=0.0)
    d = an.load_scan(run.folder, sg.load_capture)
    pol = an.polarization(d)
    check("modulation flagged as unresolved", pol["mod_snr"] < 10, f"{pol['mod_snr']:.1f}")
    check("no dips found in noise", len(an.dip_er(pol)) == 0, len(an.dip_er(pol)))


def spin_echo_checks(sg):
    print("\nspin echo: two legs 16.667 ms apart in one record, HRES single shots,"
          " 1 in 6 shots with the lock missed")
    tmp = tempfile.mkdtemp(prefix="rampol-se-")
    bench = sim.Bench(seed=11, legs_ms=(0.0, 16.667), ramp_up_ms=4.5, hold_ms=0.5,
                      lock_miss=1 / 6)
    scope, ell, bench = sim.make(sg, bench=bench,
                                 roles={1: "CmdX1", 2: "PD", 3: "MonX1", 4: "MonX2"})
    for k, v in config.PRESETS["Spin echo 16.7 ms (2 legs)"]["scope"].items():
        scope.put(k, v)
    link, rot = hw.ScopeLink(scope, log=lambda *_: None), hw.Rotator(ell, log=lambda *_: None)
    chans = {1: ("CmdX1", "Trek X1 command"), 2: ("PD", "Analyzer PD"), 3: ("MonX1", "X1"),
             4: ("MonX2", "X2")}
    run = scan.ScanRun(tmp, "se", sg, link, rot, chans, log=lambda *_: None,
                       clock=lambda: bench.clock)
    plan = dict(config.DEFAULTS["scan"], mode="single", shots=8, points=20000)
    run.new(plan, [{"kind": "dark", "target": 0}]
            + scan.build_steps(scan.angle_list(0, 170, 10), 6, 45))
    imax, bench.imax = bench.imax, 0.0
    run.run(kinds={"dark"})
    bench.imax = imax
    run.run()
    files = [f for f in os.listdir(run.folder) if f.endswith(".npz")]
    check("one file per shot (dark + 18 angles + 4 refs)", len(files) == 8 * 23, len(files))
    d = an.load_scan(run.folder, sg.load_capture, lock_tol=0.006)
    check("record spans -12 to 38 ms (preset, LEFT reference)",
          abs(d.t[0] + 12e-3) < 5e-5 and abs(d.t[-1] - 38e-3) < 5e-5,
          f"{d.t[0] * 1e3:.3f} .. {d.t[-1] * 1e3:.3f} ms")
    rej = sum(s["rejected"] for s in d.steps)
    check("missed-lock shots dropped (~1 in 6 of 184)", 15 <= rej <= 50, rej)
    pol = an.polarization(d)
    err = pol["rotation"] - bench.rotation(d.t)
    check("rotation through both legs within 0.05 deg rms", np.std(err) < 0.05,
          f"{np.std(err) * 1e3:.1f} mdeg")
    kinds = [s["kind"] for s in an.segments(d.t, pol["rotation"])]
    check("segments numbered by leg",
          kinds == ["rest", "up 1", "hold 1", "down 1", "after 1", "up 2", "hold 2",
                    "down 2", "after 2"], kinds)
    d_all = an.load_scan(run.folder, sg.load_capture, lock_tol=0.0)
    p_all = an.polarization(d_all)
    rest = an.rest_index(d.t)
    imin_true = bench.imax / (1 / (1 / bench.er(0) + 1 / bench.er_pol))
    e_rej = abs(np.mean(pol["imin"][rest]) - imin_true)
    e_all = abs(np.mean(p_all["imin"][rest]) - imin_true)
    check("dropping them brings rest Imin closer to the model", e_rej < e_all,
          f"{e_rej * 1e3:.2f} vs {e_all * 1e3:.2f} mV kept")
    check("dips found on all four ramps", len(an.dip_er(pol)) >= 60, len(an.dip_er(pol)))
    check("command channel recorded and read back by role",
          "CmdX1" in d.roles and abs(np.max(d.steps[1]["v"]["CmdX1"]) - 1.65 * 180 / 35.07) < 0.1)


def stopped_mid_step_checks(sg):
    print("\nstopped in the middle of a step: its shots are kept and loaded")
    tmp = tempfile.mkdtemp(prefix="rampol-stop-")
    scope, ell, bench = sim.make(sg)
    link, rot = hw.ScopeLink(scope, log=lambda *_: None), hw.Rotator(ell, log=lambda *_: None)
    shots_seen = []
    run = scan.ScanRun(tmp, "st", sg, link, rot, {1: ("PD", "PD")}, log=lambda *_: None,
                       clock=lambda: bench.clock,
                       cancelled=lambda: shots_seen.count("partial") >= 8 + 8 + 3)
    plan = dict(config.DEFAULTS["scan"], mode="single", shots=8, points=5000)
    run.new(plan, scan.build_steps(scan.angle_list(0, 90, 30), 0, 45))
    try:
        run.run(on_step=lambda s: shots_seen.append(s["status"]))
        check("the stop was honoured", False)
    except hw.Cancelled:
        pass
    steps = run.load()["steps"]
    status = [(s["status"], len(s.get("files", []))) for s in steps]
    check("two steps done, the third partial with the 3 shots it took",
          status[:3] == [("done", 8), ("done", 8), ("partial", 3)], status)
    d = an.load_scan(run.folder, sg.load_capture)
    part = [s for s in d.steps if s.get("partial")]
    check("the partial step is loaded, flagged, with its 3 shots",
          len(d.steps) == 3 and len(part) == 1 and part[0]["nb"] == 3,
          [(s["kind"], s["nb"], s.get("partial")) for s in d.steps])
    check("on_step reported every shot, and each finished step",
          shots_seen.count("partial") == 19 and shots_seen.count("done") == 2, shots_seen)


def few_angles_checks():
    print("\n3 angles over 45 deg (the 5 Oct test-2 scan) and a ramp turning the negative way")
    th = np.array([0.0, 22.5, 45.0])
    t = np.linspace(-12e-3, 38e-3, 2000)
    # raised-cosine 4.5 ms up, 0.5 ms hold, 4.5 ms down, to -180 deg
    up = 0.5 * (1 - np.cos(np.pi * np.clip(t, 0, 4.5e-3) / 4.5e-3))
    down = 0.5 * (1 - np.cos(np.pi * np.clip(9.5e-3 - t, 0, 4.5e-3) / 4.5e-3))
    psi = -90.0 - 180.0 * np.minimum(up, down)
    rng = np.random.default_rng(3)
    I = malus(th, psi, 5.2, np.full(len(t), 900.0)) + rng.normal(0, 1.1e-3, (3, len(t)))
    f = an.harmonic_fit(th, I, sem=np.full(3, 1.1e-3))
    check("no residual left: errors from the shot scatter, said so",
          f["dof"] == 0 and f["err_source"].startswith("shot scatter")
          and np.all(np.isfinite(f["sig_imin"])) and np.median(f["sig_imin"]) > 1e-4,
          f"{f['err_source']}, sig_Imin {np.median(f['sig_imin']) * 1e3:.2f} mV")
    check("angle coverage reported", abs(f["theta_span"] - 45.0) < 1e-6, f["theta_span"])
    rot = an.unwrap_psi(f["psi"]) - an.unwrap_psi(f["psi"])[0]
    kinds = [s["kind"] for s in an.segments(t, rot)]
    check("a ramp to -180 is still rest / up / hold / down / after",
          kinds == ["rest", "up", "hold", "down", "after"], kinds)
    from rampol import checks
    plan = dict(config.DEFAULTS["scan"], start=0.0, stop=45.0, step=22.5)
    sg_prof = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"]).scope_profiles.PROFILES["msox2014a"]
    found = checks.settings_checks({}, sg_prof, {}, plan)
    check("the pre-run check warns about too few, bunched angles",
          sum(1 for lv, m in found if lv == "WARN" and ("angles" in m)) == 2,
          [m for lv, m in found if lv == "WARN"])


def main():
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    few_angles_checks()
    stopped_mid_step_checks(sg)
    harmonic_checks()
    scan_checks(sg)
    no_light_checks(sg)
    spin_echo_checks(sg)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Analysis OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

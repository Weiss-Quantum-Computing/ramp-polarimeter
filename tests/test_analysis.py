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
    plan = dict(config.DEFAULTS["scan"])
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
    print("
no light (the 5 Oct dry run): nothing to report, and it says so")
    tmp = tempfile.mkdtemp(prefix="rampol-dark-")
    run, bench = sim_scan(tmp, sg, imax=0.0)
    d = an.load_scan(run.folder, sg.load_capture)
    pol = an.polarization(d)
    check("modulation flagged as unresolved", pol["mod_snr"] < 10, f"{pol['mod_snr']:.1f}")
    check("no dips found in noise", len(an.dip_er(pol)) == 0, len(an.dip_er(pol)))


def main():
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    harmonic_checks()
    scan_checks(sg)
    no_light_checks(sg)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Analysis OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

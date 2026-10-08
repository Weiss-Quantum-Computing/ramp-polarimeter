"""Bias points: the plan, the null fit, and a whole run on the simulated bench
(AWG plateaus into the bench model, the analyzer stepped around each null at
a sensitive V/div), against the bench's known extinction ratio and rotator
error.

    python tests/test_bias.py
"""
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import bias, config, hw, sim  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def plan_checks(eomilc):
    print("\nthe plan")
    check("bias list", bias.parse_biases("0:180:15") == [15.0 * i for i in range(13)]
          and bias.parse_biases("0, 30,90") == [0.0, 30.0, 90.0])
    check("up-down order", bias.order_biases([0, 30, 60], "updown") == [0, 30, 60, 30, 0])
    v = bias.awg_volts(90, 0.5)
    check("AWG volts for 90 deg, half each",
          abs(v["EO1"] - 0.5 * 9.1675) < 0.01 and abs(v["EO2"] - 0.5 * 8.672) < 0.01,
          f"EO1 {v['EO1']:.3f} V, EO2 {v['EO2']:.3f} V (V90 commanded 9.168 / 8.672)")
    t, u = bias.plateau(4.0, bias.PLAN)
    check("plateau ends at 0 V and holds the bias",
          u[0] == 0 and u[-1] == 0 and np.isclose(u.max(), 4.0),
          f"{len(u)} pts, {t[-1]*1e3:.2f} ms")
    try:
        bias.check_plateaus([200], dict(bias.PLAN), eomilc)
        check("past the AWG cap refused", False)
    except ValueError as e:
        check("past the AWG cap refused", "cap" in str(e), str(e)[:70])
    out = bias.check_plateaus(bias.parse_biases("0:180:30"), dict(bias.PLAN), eomilc)
    check("0-180 deg passes the Trek limit check", len(out) == 7,
          out[-1][2][:80] if eomilc else "(eomilc not loaded: cap only)")


def null_fit_checks():
    print("\nthe null fit")
    rng = np.random.default_rng(5)
    th = np.linspace(-3, 3, 9) + 12.345
    I = 1e-3 + 5.0 * np.sin(np.deg2rad(th - 12.345)) ** 2 + rng.normal(0, 2e-5, th.size)
    f = bias.fit_null(th, I, np.full(th.size, 2e-5))
    check("null angle, Imin, K recovered",
          abs(f["theta_n"] - 12.345) < 3 * f["sig_theta_n"] + 1e-4
          and abs(f["imin"] - 1e-3) < 3 * f["sig_imin"] + 1e-6 and abs(f["k"] - 5) < 0.02,
          f"theta_n {f['theta_n']:.4f} +- {f['sig_theta_n']*1e3:.2f} mdeg, Imin "
          f"{f['imin']*1e3:.4f} +- {f['sig_imin']*1e6:.1f} uV, K {f['k']:.4f}")
    e = bias.er_point(1e-5, 2e-5, 5.0)
    check("unresolved Imin -> a lower bound", e["er"] is None and e["er_lower"] > 1e5,
          f"ER > {e['er_lower']:.0f}")
    e = bias.er_point(1.5e-3, 1e-5, 5.0, 0.0, 4.9985)
    check("ER, Malus ratio, ellipticity", abs(e["er"] - 3333.3) < 1 and
          abs(e["malus_ratio"] - 1) < 1e-3 and abs(e["ellipticity_deg"] - 0.99) < 0.01,
          f"ER {e['er']:.0f}, K/(Imax-Imin) {e['malus_ratio']:.4f}, "
          f"ellipticity {e['ellipticity_deg']:.3f} deg")


def run_checks(sg, eomilc):
    print("\na run on the simulated bench (rotator error 2 deg, 90 deg period)")
    roles = {1: "PD", 2: "CmdX1", 3: "MonX1", 4: "MonX2"}
    bench = sim.Bench(pd_noise=0.2e-3, drift=0.0, rotator_err_deg=2.0,
                      er_rest=5000.0, er_mid=300.0, legs_ms=(0.0,))
    scope, ell, bench = sim.make(sg, roles=roles, bench=bench)
    scope.noise_per_div = 0.01
    link = hw.ScopeLink(scope, log=lambda s: None)
    rot = hw.Rotator(ell, log=lambda s: None)
    awg = sim.FakeAWG(bench)
    asked = []

    def ask(title, text):
        asked.append(title)
        if "Block the beam" in text:
            bench._imax_saved, bench.imax = bench.imax, 0.0
        elif "Unblock" in text:
            bench.imax = bench._imax_saved
        return True
    tb0 = scope.get(":TIMebase:SCALe")
    pd0 = link.channel_state([1])[1]
    logs = []
    run = bias.BiasRun(tempfile.mkdtemp(prefix="rampol-bias-"), "sim-bias", link, rot,
                       awg, {r: ch for ch, r in roles.items()},
                       plan={"biases": "0:90:30", "shots": 4, "points": 4000,
                             "upload_settle_s": 0.0}, log=logs.append, ask=ask,
                       eomilc=eomilc)
    pts = run.run()
    check("4 points measured", len(pts) == 4, ", ".join(f"{p['bias']:g}" for p in pts))
    er_true = lambda r: 1 / (1 / bench.er(r) + 1 / bench.er_pol)
    for p_ in pts:
        tr = er_true(p_["phi_mon"])
        ok = p_["er"] is not None and abs(p_["er"] / tr - 1) < 0.15
        check(f"ER at {p_['bias']:g} deg near the bench's {tr:.0f}", ok,
              f"measured {p_['er'] or 0:.0f}, Imin {p_['imin']*1e3:.3f} mV at "
              f"{p_['scan']['vdiv']*1e3:g} mV/div, Malus ratio {p_.get('malus_ratio', 0):.3f}")
    tf = bias.transfer(pts)
    want = [2.0 * math.sin(2 * math.pi * ph / 90) for ph in tf["phi_mon"]]
    dev = np.array(tf["rot_light"]) - np.array(tf["phi_mon"])
    dev -= dev[0]
    check("the static rotator error is seen",
          np.max(np.abs(dev - np.array(want))) < 0.15,
          "light - monitors " + ", ".join(f"{d:+.2f}" for d in dev)
          + " deg; bench " + ", ".join(f"{w:+.2f}" for w in want))
    check("monitors read the bias", abs(pts[-1]["phi_mon"] - 90) < 1.0,
          f"{pts[-1]['phi_mon']:.2f} deg at 90")
    check("outputs off, scope restored",
          not any(bench.awg_on.values()) and scope.get(":TIMebase:SCALe") == tb0
          and link.channel_state([1])[1] == pd0)
    check("asked before the outputs and for the dark",
          asked[:3] == ["Fixed rotations", "Dark", "Dark"], str(asked))
    man = bias.load(run.folder)
    check("manifest reloads", len(man["points"]) == 4 and man["transfer"] is not None)


def pairs_checks(sg, eomilc):
    print("\nX1 / X2 pairs on the simulated bench: a 2 x 2 grid, the null predicted, the "
          "azimuth tracked, a stop and a resume")
    from rampol import plan as planmod, lablog
    # the simulated bench turns the light WITH the monitors (sense +1); the
    # plan says -1 (the real bench) and the run must learn it from its points
    p = {"x1": "0, 45", "x2": "0, 45", "how": "grid", "shots": 4, "points": 4000,
         "upload_settle_s": 0.0, "track": True, "track_ms": 30.0, "predict_null": True,
         "sense": -1.0}
    check("the sense is learned from two measured nulls (+1 here), else the plan's",
          bias.sense_from([{"theta_n": 10.0, "bias": 0.0}, {"theta_n": 55.0, "bias": 45.0}]) == 1.0
          and bias.sense_from([{"theta_n": 10.0, "bias": 0.0}, {"theta_n": 145.0, "bias": 45.0}]) == -1.0
          and bias.sense_from([{"theta_n": 10.0, "bias": 0.0}], -1.0) == -1.0)
    ends = bias.ends_list(p)
    check("a grid is every X1 with every X2", ends == [(0.0, 0.0), (0.0, 45.0), (45.0, 0.0),
                                                      (45.0, 45.0)], ends)
    check("pairs take the lists together",
          bias.ends_list({"x1": "0, 30", "x2": "10", "how": "pairs"}) == [(0.0, 10.0), (30.0, 10.0)])
    check("a rotation list still splits between the crystals",
          bias.ends_list({"biases": "0, 60", "split": 0.25}) == [(0.0, 0.0), (15.0, 45.0)])
    v = bias.ends_volts(90.0, 0.0)
    check("volts for X1 90 / X2 0", abs(v["EO1"] - 9.1675) < 0.01 and v["EO2"] == 0.0, v)
    w = bias.plateau_wave((30.0, 60.0), dict(bias.PLAN))
    check("a pair's plateau carries its ends and the summed rotation",
          w.ends == {"EO1": 30.0, "EO2": 60.0} and w.rotation == 90.0 and "X1 30" in w.label)
    steps = planmod.fixed_rotations(p, None, "g")
    kinds = [s["kind"] for s in steps]
    check("the plan: 4 azimuth angles once, then null points, bright and 2 track angles "
          "per point", kinds.count("azimuth") == 4 and kinds.count("track") == 8
          and kinds.count("null") == 4 * bias.PLAN["null_points"] and kinds.count("bright") == 4,
          {k: kinds.count(k) for k in set(kinds)})
    steps = planmod.fixed_rotations(dict(p, predict_null=False), None, "g")
    check("without prediction every point has its 4 angles",
          [s["kind"] for s in steps].count("azimuth") == 16)
    roles = {1: "PD", 2: "CmdX1", 3: "MonX1", 4: "MonX2"}
    bench = sim.Bench(pd_noise=0.2e-3, drift=0.0, rotator_err_deg=2.0,
                      er_rest=5000.0, er_mid=300.0, legs_ms=(0.0,))
    scope, ell, bench = sim.make(sg, roles=roles, bench=bench)
    scope.noise_per_div = 0.01
    link = hw.ScopeLink(scope, log=lambda s: None)
    rot = hw.Rotator(ell, log=lambda s: None)
    awg = sim.FakeAWG(bench)
    asked = []

    def ask(title, text):
        asked.append(title)
        if "Block the beam" in text:
            bench._imax_saved, bench.imax = bench.imax, 0.0
        elif "Unblock" in text:
            bench.imax = bench._imax_saved
        return True
    folder = tempfile.mkdtemp(prefix="rampol-pairs-")
    n_pts = []
    logs = []
    run = bias.BiasRun(folder, "grid", link, rot, awg, {r: ch for ch, r in roles.items()},
                       plan=p, log=logs.append, ask=ask, eomilc=eomilc,
                       cancelled=lambda: len(n_pts) >= 2,
                       on_point=lambda pt: n_pts.append(pt))
    try:
        run.run()
    except hw.Cancelled:
        pass
    man = bias.load(os.path.join(folder, "grid"))
    check("stopped after 2 points: on record with the darks and null settings, not finished",
          len(man["points"]) == 2 and man.get("dark") and man.get("null_settings")
          and not man.get("finished"), (len(man["points"]), bool(man.get("dark"))))
    asked2 = []

    def ask2(title, text):
        asked2.append(title)
        return True
    run2 = bias.BiasRun(folder, "grid", link, rot, awg, {r: ch for ch, r in roles.items()},
                        plan=p, log=logs.append, ask=ask2, eomilc=eomilc, resume=True)
    pts = run2.run()
    check("resumed: the other 2 points measured, no dark asked for again, finished",
          len(pts) == 4 and asked2 == ["Fixed rotations"] and [p_["i"] for p_ in pts] == [0, 1, 2, 3],
          (len(pts), asked2))
    man = bias.load(os.path.join(folder, "grid"))
    check("points carry X1 / X2 and the summed rotation",
          [(p_["x1"], p_["x2"], p_["bias"]) for p_ in man["points"]]
          == [(0.0, 0.0, 0.0), (0.0, 45.0, 45.0), (45.0, 0.0, 45.0), (45.0, 45.0, 90.0)])
    er_true = lambda r_: 1 / (1 / bench.er(r_) + 1 / bench.er_pol)
    for p_ in man["points"]:
        tr = er_true(p_["phi_mon"])
        ok = p_["er"] is not None and abs(p_["er"] / tr - 1) < 0.15
        check(f"ER at X1 {p_['x1']:g} / X2 {p_['x2']:g} near the bench's {tr:.0f}", ok,
              f"measured {p_['er'] or 0:.0f}, Imin {p_['imin']*1e3:.3f} mV at "
              f"{p_['scan']['vdiv']*1e3:g} mV/div"
              + (", null predicted" if p_.get("null_predicted") else ""))
    check("point 2 predicted with the plan's wrong sense misses and falls back to 4 angles; "
          "from then on the learned sense predicts the null",
          not man["points"][0]["null_predicted"] and not man["points"][1]["null_predicted"]
          and all(p_["null_predicted"] and p_["null_converged"] for p_ in man["points"][2:])
          and all(p_["null_converged"] for p_ in man["points"])
          and [p_["sense"] for p_ in man["points"]] == [-1.0, -1.0, 1.0, 1.0],
          [(p_["null_predicted"], p_["null_converged"], p_["sense"]) for p_ in man["points"]])
    tracks = bias.load_tracks(os.path.join(folder, "grid"))
    tk = [p_["track"] for p_ in man["points"]]
    check("the azimuth was tracked at every point: traces on disk, a creep slope in the "
          "manifest, no relaxation time claimed on a bench without drift",
          len(tracks) == 4 and all("hold_slope_mdeg_ms" in x and "tau_ms" not in x for x in tk)
          and all(len(t_["lm"]) > 100 for t_ in tracks.values()),
          [(round(x.get("hold_slope_mdeg_ms", 0), 2), x.get("after_from")) for x in tk])
    check("holds past 25 deg carry the rest pair for the tail after the fall",
          [("lm_rest" in t_) for _i, t_ in sorted(tracks.items())] == [False, True, True, True]
          and tk[0]["after_from"] == "hold pair" and tk[3]["after_from"] == "rest pair")
    # the bench has no slow drift and its monitors read the drive: light -
    # monitors stays within the shot noise through the hold, and after the
    # fall where the pair in use can see it
    w0, w1 = man["window_s"]
    worst_hold = max(float(np.max(np.abs(t_["lm"][(t_["t"] > w0) & (t_["t"] < w1)])))
                     for t_ in tracks.values())
    t_f = man["t_fall_s"]
    worst_after = max(float(np.max(np.abs((t_.get("lm_rest", t_["lm"]))[t_["t"] > t_f + 2e-3])))
                      for t_ in tracks.values())
    check("tracked light - monitors stays near zero on a drift-free bench (hold and tail < 0.3 deg)",
          worst_hold < 0.3 and worst_after < 0.3, f"{worst_hold*1e3:.0f} / {worst_after*1e3:.0f} mdeg worst")
    row = lablog.bias_row(man)
    check("the lab-log row names the point of the lowest ER and the tracking",
          row["direct_er_min_at"].startswith("bias X1") and "tracked at 4" in row["result"],
          (row["direct_er_min_at"], row["result"]))


def main():
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    try:
        eomilc = hw.load_eomilc(config.DEFAULTS["eomilc_path"])
    except Exception as exc:
        print(f"(EOM-ILC not loaded: {exc})")
        eomilc = None
    plan_checks(eomilc)
    null_fit_checks()
    run_checks(sg, eomilc)
    pairs_checks(sg, eomilc)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Bias OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

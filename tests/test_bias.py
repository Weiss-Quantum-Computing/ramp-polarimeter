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
          asked[:3] == ["Bias points", "Dark", "Dark"], str(asked))
    man = bias.load(run.folder)
    check("manifest reloads", len(man["points"]) == 4 and man["transfer"] is not None)


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
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Bias OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

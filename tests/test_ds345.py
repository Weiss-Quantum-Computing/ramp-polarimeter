"""The DS345 light gate (rampol/ds345.py) against the simulator: the gate,
the Hi-Z scaling, the limits, the session's order of commands, the dry run
(and that it catches a 50 Ohm load), and the light it lets through on the
simulated bench. No hardware.

    python tests/test_ds345.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import config, ds345 as ds, hw, sim  # noqa: E402

FAILS = []
ROLES = {1: "PD", 3: "MonX1", 4: "MonX2"}


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


def gate_checks():
    print("\nthe gate and its limits (SRS manual: 8..16,300 points, 40 MHz / N, 50 Ohm source)")
    n, dt = ds.fit_clock(0.300)
    check("a 300 ms record fits 16,300 points by its clock divider",
          0.300 / dt + 1 <= ds.MAX_PTS and abs(dt - n / 40e6) < 1e-15, f"N {n}, {dt*1e6:.3f} us")
    g = ds.build([(2.2e-3, 11.8e-3)], idle="off", on_v=1.0, off_v=0.0, edge_us=20)
    check("starts and ends at idle, the other level inside the window",
          g.v[0] == g.v[-1] == 0.0 and abs(g.v[int(5e-3 / g.dt)] - 1.0) < 1e-12)
    check("light() follows the gate, idle before the trigger",
          np.allclose(g.light(np.array([-1e-3, 5e-3, 30e-3])), [0, 1, 0]))
    offs, ampl, codes = ds.program(g)
    check("Hi-Z: 0..1 V at the load is programmed 0..0.5 V (OFFS 0.25, AMPL 0.5 Vpp)",
          abs(offs - 0.25) < 1e-12 and abs(ampl - 0.5) < 1e-12
          and codes.min() == -1 and codes.max() == 1, (offs, ampl))
    big = ds.build([(1e-3, 2e-3)], on_v=10.5, off_v=0.0)
    check("past +-10 V at a Hi-Z load is refused", "past the DS345" in (raises(ds.program, big) or ""))
    found = ds.check(g, 0.0, 0.5)
    check("a gate outside the modulator's range is a FAIL",
          ds.worst(found) == "FAIL" and any("outside" in m for _l, m in found))
    long_ = ds.build([(1e-3, 280e-3)], idle="on")
    check("a gate still playing at the next trigger is a FAIL (the DS345 ignores it)",
          ds.worst(ds.check(long_, 0.0, 1.0, trig_hz=3.7)) == "FAIL")
    check("a window before the trigger is refused",
          raises(ds.build, [(-1e-3, 2e-3)]) is not None)
    check("parse_windows reads ms", np.allclose(ds.parse_windows("20-30, 2.2-11.8"),
                                                [(2.2e-3, 11.8e-3), (20e-3, 30e-3)]))


def session_checks(sg):
    print("\nthe session on the simulated DS345 and bench")
    bench = sim.Bench(seed=3)
    scope, _ell, bench = sim.make(sg, roles=ROLES, bench=bench)
    link = hw.ScopeLink(scope, log=lambda *_: None)
    dev = sim.FakeDS345(bench)
    s = ds.Session(dev, log=lambda *_: None)
    s.park(1.0)
    check("park: DC at the load, burst off", np.allclose(dev.out_v(np.array([0.0, 5e-3])), 1.0)
          and dev.st["MENA"] == 0)
    g = ds.build([(2.2e-3, 11.8e-3)], idle="off")
    s.load(g)
    w = dev.writes
    i_mena, i_func = w.index("MENA 1"), max(i for i, x in enumerate(w) if x == "FUNC 5")
    check("loaded: burst armed on the external trigger before ARB is selected (never "
          "free-running)", i_mena < i_func and "TSRC 2" in w and "BCNT 1" in w, w[-8:])
    t = np.array([-1e-3, 1e-3, 5e-3, 13e-3, 100e-3])
    check("plays the gate after the trigger", np.allclose(dev.out_v(t), [0, 0, 1, 0, 0], atol=1e-3))
    # the light it lets through
    sig_on = bench.signals(np.array([5e-3]), 1)["PD"][0]
    sig_off = bench.signals(np.array([-1e-3]), 1)["PD"][0]
    check("the bench's light follows the gate (dark outside the window)",
          sig_off < bench.dark + 0.1 and sig_on > sig_off + 0.1, (sig_off, sig_on))
    bench.ds_wiring, bench.ds_scope_ch = "scope", 2      # the output teed to scope CH2
    rep = ds.dry_run(s, link, g, 2, shots=2, points=20000, log=lambda *_: None)
    r = rep["result"]
    check("dry run on a Hi-Z scope channel: gain 1, idle right, passes",
          rep["ok"] and abs(r["gain"] - 1) < 0.02, (r["gain"], r["delay_us"], rep["problems"]))
    check("the dry run's scope channel is put back", link.channel_state([2])[2] is not None)
    dev.load_ohms = 50.0
    rep = ds.dry_run(s, link, g, 2, shots=2, points=20000, log=lambda *_: None)
    check("a 50 Ohm load reads gain 0.5 and is named", not rep["ok"] and
          any("50 Ohm" in p for p in rep["problems"]), rep["problems"])
    dev.load_ohms = None
    bench.ds_wiring = "modulator"


def main():
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    gate_checks()
    session_checks(sg)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("DS345 OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The ILC-target comparison (rampol.ilc_target.compare) end to end on the
simulated bench: the AWG plays an ILC record into the Treks, a ramp scan is
taken and written, and compare() reads it back against the record's target.

    python tests/test_ilc_target.py

One-transport record: the numbers must stay what they were before records
could hold several transports (13 Oct 2026). Two-transport record (transport
1, idle, transport 2 16.667 ms later in ONE record): each transport located
on its own, identity timing for each, per-transport numbers, one correction
per crystal over the whole record.
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import config, hw, scan, sim  # noqa: E402
from rampol import ilc_target as it  # noqa: E402

FAILS = []
DT = 2e-6
RISE, HOLD, PEAK = 2.5e-3, 1.0e-3, 5.0


def check(label, ok, detail=""):
    detail = str(detail)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def transport(t, start):
    """A cosine ramp 0 -> PEAK -> 0 (monitor volts) starting at `start`."""
    x = t - start
    up = 0.5 * (1 - np.cos(np.pi * np.clip(x / RISE, 0, 1)))
    down = 0.5 * (1 - np.cos(np.pi * np.clip((2 * RISE + HOLD - x) / RISE, 0, 1)))
    return PEAK * np.minimum(up, down)


def record(starts, length):
    tt = np.arange(0, length, DT)
    return tt, np.max([transport(tt, s) for s in starts], axis=0)


def expected_landmarks(start):
    """The 0.1 % / 99.9 % crossings of one cosine transport."""
    a = np.arccos(1 - 2e-3) / np.pi * RISE
    return np.array([start + a, start + RISE - a, start + RISE + HOLD + a,
                     start + 2 * RISE + HOLD - a])


def bench_scan(tmp, sg, tt, T, play_t, play_T, name):
    """ILC states for EO1/EO2 (target T on the record tt, drive = T over the
    AWG gain), the bench's AWG playing play_T (on its own grid from the
    trigger), and a ramp scan of it. Returns (scan folder, x1, x2)."""
    bench = sim.Bench(seed=7)
    scope, ell, bench = sim.make(sg, bench=bench,
                                 roles={1: "CmdX1", 2: "PD", 3: "MonX1", 4: "MonX2"})
    for k, v in config.PRESETS[config.SPIN_ECHO]["scope"].items():
        scope.put(k, v)
    states = []
    for ch, eo in ((1, "EO1"), (2, "EO2")):
        p = os.path.join(tmp, f"drive_{name}X{ch}.state.npz")
        np.savez(p, t=tt, target=T, u=T / bench.awg_gain[ch], dt=DT,
                 name=f"{name}X{ch}", channel=eo, iteration=1)
        states.append(p)
        bench.awg_drive[ch] = (len(play_t) * DT, play_T / bench.awg_gain[ch])
        bench.awg_on[ch] = True
    link, rot = hw.ScopeLink(scope, log=lambda *_: None), hw.Rotator(ell, log=lambda *_: None)
    chans = {1: ("CmdX1", "Trek X1 command"), 2: ("PD", "Analyzer PD"), 3: ("MonX1", "X1"),
             4: ("MonX2", "X2")}
    run = scan.ScanRun(tmp, name, sg, link, rot, chans, log=lambda *_: None,
                       clock=lambda: bench.clock)
    plan = dict(config.DEFAULTS["scan"], mode="single", shots=4, points=20000)
    run.new(plan, [{"kind": "dark", "target": 0}]
            + scan.build_steps(scan.angle_list(0, 170, 10), 6, 45))
    imax, bench.imax = bench.imax, 0.0
    run.run(kinds={"dark"})
    bench.imax = imax
    run.run()
    return run.folder, states[0], states[1]


def run_compare(folder, x1, x2, **kw):
    lines = []
    s = it.compare(folder, x1, x2, scope_grab_path=config.DEFAULTS["scope_grab_path"],
                   eomilc_path=config.DEFAULTS["eomilc_path"], log=lines.append, **kw)
    return s, lines


# compare()'s numbers on the one-transport record below, from the code before
# records could hold several transports (4ad27fd, 13 Oct 2026)
BASELINE = {
    "source_landmarks_ms": [0.252, 2.65, 3.75, 6.148],
    "landmarks_ms": [[0.25200034495867757, 2.6500006219902805, 3.7499984077564434,
                      6.148002845736654],
                     [16.91899985372367, 19.31700013075527, 20.416997916521435,
                      22.815002354501644]],
    "gain": [0.9200000557200727, 0.9200003955012487],
    "rms_mV": [0.12243603707190207, 0.11964401327440434],
    "d_light_rms_mdeg": [18.824863219871578, 16.82424002676028],
    "d_lm_rms_mdeg": [19.04697984444167, 16.945689356250192],
    "d_lm_delay_us": [0.0074344184289593445, 0.0007278528543910107],
    "corr_peak_mdeg": 5.086575863475585, "corr_rms_mdeg_active": 2.6934131216004125,
    "peak_V": {"EO1": 0.1449193722258987, "EO2": 0.1451765268945522},
}


def close(a, b, rel=1e-6, abs_=1e-9):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float))
                       <= abs_ + rel * np.abs(np.asarray(b, float))))


def one_transport(sg):
    print("\none transport in the record, played twice (legs 16.667 ms apart)")
    tmp = tempfile.mkdtemp(prefix="rampol-ilc1-")
    tt, T = record([0.2e-3], 12e-3)
    pt, pT = record([0.2e-3, 16.867e-3], 25e-3)
    folder, x1, x2 = bench_scan(tmp, sg, tt, T, pt, pT, "TT1")
    s, _ = run_compare(folder, x1, x2)
    nums = {"source_landmarks_ms": s["source_landmarks_ms"],
            "timing": [{k: tm[k] for k in ("landmarks_ms", "gain", "offset_V", "rms_mV")}
                       for tm in s["timing"]],
            "legs": [{k: L[k] for k in ("d_light_fit", "d_mon_fit", "d_lm_fit")}
                     for L in s["legs"]],
            "correction": {k: s["correction"][k] for k in ("peak_mdeg", "rms_mdeg_active")},
            "per_crystal": s["correction"]["per_crystal"]}
    if "--baseline" in sys.argv:
        print(json.dumps(nums, indent=1))
    S = it.landmarks(tt, T)
    tr = it.transports(tt, T)
    check("landmarks() is transports()[0] for one transport",
          len(tr) == 1 and np.array_equal(S, tr[0]), np.round(S * 1e3, 4))
    B = BASELINE
    check("source landmarks as before", close(s["source_landmarks_ms"], B["source_landmarks_ms"]),
          s["source_landmarks_ms"])
    check("time map of both legs as before",
          len(s["timing"]) == 2
          and close([tm["landmarks_ms"] for tm in s["timing"]], B["landmarks_ms"], abs_=1e-6)
          and close([tm["gain"] for tm in s["timing"]], B["gain"])
          and close([tm["rms_mV"] for tm in s["timing"]], B["rms_mV"]),
          [np.round(tm["landmarks_ms"], 4).tolist() for tm in s["timing"]])
    check("per-leg decomposition as before",
          close([L["d_light_fit"]["rms_mdeg"] for L in s["legs"]], B["d_light_rms_mdeg"])
          and close([L["d_lm_fit"]["rms_mdeg"] for L in s["legs"]], B["d_lm_rms_mdeg"])
          and close([L["d_lm_fit"]["delay_us"] for L in s["legs"]], B["d_lm_delay_us"],
                    abs_=1e-6),
          [round(L["d_lm_fit"]["rms_mdeg"], 3) for L in s["legs"]])
    check("correction as before",
          close(s["correction"]["peak_mdeg"], B["corr_peak_mdeg"])
          and close(s["correction"]["rms_mdeg_active"], B["corr_rms_mdeg_active"])
          and all(close(s["correction"]["per_crystal"][ch]["peak_V"], v)
                  for ch, v in B["peak_V"].items()),
          f"{s['correction']['peak_mdeg']:.4f} mdeg peak")
    check("a one-element transports list, in the summary and in each leg",
          len(s["transports"]) == 1 and s["transports"][0]["index"] == 1
          and all(len(L["transports"]) == 1 for L in s["legs"])
          and close(s["transports"][0]["landmarks_ms"], s["timing"][0]["landmarks_ms"])
          and close(s["legs"][1]["transports"][0]["landmarks_ms"],
                    s["timing"][1]["landmarks_ms"]),
          [L["transports"][0]["landmarks_ms"][0] for L in s["legs"]])
    keys = ("top_lm_mdeg", "top_lt_mdeg", "after_fall_lm_mdeg", "after_fall_lt_mdeg",
            "lt_rms_mdeg", "lm_rms_mdeg")
    check("its numbers finite", all(np.isfinite(L["transports"][0][k])
                                    for L in s["legs"] for k in keys),
          {k: round(s["transports"][0][k], 1) for k in keys})


def two_transports(sg):
    print("\ntwo transports in one record (starts 0.2 and 16.867 ms), played as recorded")
    tmp = tempfile.mkdtemp(prefix="rampol-ilc2-")
    starts = [0.2e-3, 16.867e-3]
    tt, T = record(starts, 25e-3)
    tr = it.transports(tt, T)
    exp = [expected_landmarks(s_) for s_ in starts]
    err = max(np.max(np.abs(a - b)) for a, b in zip(tr, exp)) if len(tr) == 2 else np.inf
    check("transports(): two, each at its 0.1 % / 99.9 % crossings (within 50 us)",
          len(tr) == 2 and err < 50e-6,
          [np.round(a * 1e3, 4).tolist() for a in tr])
    env = it.landmarks(tt, T)
    check("landmarks(): their envelope (rise and hold start of the first, hold end "
          "and fall of the second)",
          np.allclose(env, [tr[0][0], tr[0][1], tr[1][2], tr[1][3]]), np.round(env * 1e3, 4))
    segs = it.undriven(np.arange(-12e-3, 38e-3, 5e-6), [np.concatenate(tr)])
    check("undriven(): before transport 1 and between the two, none inside either",
          len(segs) == 2, len(segs))
    folder, x1, x2 = bench_scan(tmp, sg, tt, T, tt, T, "TT2")
    s, lines = run_compare(folder, x1, x2)
    P = s["transports"]
    check("summary: two transports, leg 1 only (leg 2 would fall inside the record)",
          len(P) == 2 and len(s["legs"]) == 1 and len(s["legs"][0]["transports"]) == 2
          and [p["index"] for p in P] == [1, 2],
          (len(P), len(s["legs"])))
    if len(P) != 2:
        return
    lm = [np.asarray(p["landmarks_ms"]) * 1e-3 for p in P]
    src = [np.asarray(x) * 1e-3 for x in s["source_transports_ms"]]
    d_lm = max(np.max(np.abs(a - b)) for a, b in zip(lm, exp))
    check("their landmarks in the experiment within 50 us of the record's crossings",
          d_lm < 50e-6, [np.round(a * 1e3, 4).tolist() for a in lm])
    d_id = max(np.max(np.abs(a - b)) for a, b in zip(lm, src))
    check("identity timing for each transport (within 20 us)", d_id < 20e-6,
          f"{d_id * 1e6:.2f} us worst")
    tm = s["timing"][0]
    check("time map: the command's gain, small rms",
          abs(tm["gain"] - 0.92) < 0.01 and tm["rms_mV"] < 5,
          f"gain {tm['gain']:.4f}, {tm['rms_mV']:.2f} mV rms")
    keys = ("top_lm_mdeg", "top_lt_mdeg", "after_fall_lm_mdeg", "after_fall_lt_mdeg",
            "lt_rms_mdeg", "lm_rms_mdeg")
    check("finite numbers per transport",
          all(np.isfinite(p[k]) for p in P for k in keys),
          [{k: round(p[k], 1) for k in keys} for p in P])
    check("light - monitors small on both (the bench's light follows its monitors)",
          all(abs(p["top_lm_mdeg"]) < 50 and p["lm_rms_mdeg"] < 30 for p in P),
          [round(p["lm_rms_mdeg"], 1) for p in P])
    corr = [f for f in s["correction"]["files"] if os.path.basename(f).startswith("corr_")]
    check("one correction file per crystal", len(corr) == 2
          and set(s["correction"]["per_crystal"]) == {"EO1", "EO2"}, corr)
    from eomilc import corrections as corrmod   # compare() put EOM-ILC on the path
    ok, why = True, []
    for f in corr:
        c = corrmod.read_correction(f)
        tc, dl = c.t, c.delta
        spans = len(tc) == len(tt) and abs(tc[0] - tt[0]) < 1e-9 and abs(tc[-1] - tt[-1]) < 1e-9
        on = [np.max(np.abs(dl[(tc > a[1]) & (tc < a[2])])) for a in tr]
        idle = np.max(np.abs(dl[(tc > tr[0][3] + 1e-3) & (tc < tr[1][0] - 1e-3)]))
        ok &= bool(spans and all(v > 0 for v in on) and idle == 0)
        why.append((os.path.basename(f), len(tc), [f"{v:.3g}" for v in on], idle))
    check("each spans the whole record, nonzero in both holds, zero in the idle", ok, why)
    figs = [f for f in os.listdir(s["out"]) if f.startswith("fig") and f.endswith(".png")]
    check("figures written", len(figs) == 5, figs)
    z = np.load(os.path.join(s["out"], "target_compare.npz"))
    check("knots in target_compare.npz", z["leg1_knots_s"].shape == (8,)
          and z["source_knots_s"].shape == (8,))


def time_map_checks():
    print("\ntime map of two transports, the second played 0.4 ms late and compressed")
    tt, T = record([0.2e-3, 16.867e-3], 25e-3)
    S_tr = it.transports(tt, T)
    u = 1.79 * T + 0.02
    E_true = [S_tr[0] * 1.0,
              np.array([17.267e-3, 17.267e-3 + 2.3e-3, 17.267e-3 + 2.3e-3 + 0.7e-3,
                        17.267e-3 + 2 * 2.3e-3 + 0.7e-3])]
    t = np.arange(-12e-3, 38e-3, 2.5e-6)
    cmd = 0.92 * np.interp(it.warp(S_tr, E_true, t), tt, u) - 0.05
    cmd = cmd + np.random.default_rng(4).normal(0, 0.01, len(t))
    E_tr, g, o, rms = it.fit_time_map(t, cmd, tt, u, S_tr, 0.3e-3)
    err = max(np.max(np.abs(a - b)) for a, b in zip(E_tr, E_true))
    check("each transport's landmarks recovered to 2 us", len(E_tr) == 2 and err < 2e-6,
          [np.round((a - b) * 1e6, 2).tolist() for a, b in zip(E_tr, E_true)])
    check("gain and offset of the whole map", abs(g - 0.92) < 1e-3 and abs(o + 0.05) < 2e-3,
          f"gain {g:.4f}, offset {o:+.4f} V, rms {rms * 1e3:.1f} mV")


def main():
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    time_map_checks()
    one_transport(sg)
    two_transports(sg)
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("ILC target OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

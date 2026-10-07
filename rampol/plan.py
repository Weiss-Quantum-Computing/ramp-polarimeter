"""What a measurement will do, step by step, before it does it: the analyzer
angle, what the AWG plays, and the shots of every step, in the order they
are taken, with an estimate of when. The Plan tab draws it.

A step is a dict: kind (dark | background | scan | ref | azimuth | null |
bright), angle (deg; None for a dark or background), rel (True when the
angle is counted from the crossed angle found at that rotation - Fixed
rotations finds it as it goes, so it is not known beforehand), x1 / x2 (the
rotation the AWG puts on each crystal, deg; None when this window's AWG does
not drive), shots, scan (which scan it belongs to), note, then t0 and dur
(s) from timeline().
"""
import math

from . import scan as scanmod

# Fitted to the capture times of 7 Oct 2026 (two 5-ramp AWG sequences at a
# 0.27 s bench trigger, 550 shots each, and three spin-echo scans). A single
# shot waits for the first trigger after the scope is re-armed, so every
# shot is a whole number of trigger periods: readout + the screen's span,
# rounded UP to periods - 0.81 s (3 periods) at a 15 ms screen and 1.08 s (4)
# at 270 ms, where "one per 0.6 s" had said 0.6 s. Whatever happens between
# steps (analyzer, AWG) runs during that wait, so at a 5.6 s trigger it costs
# nothing and at 0.27 s it costs whole periods.
READOUT_S = 0.6             # re-arm + read out one shot (20000 points, 4 channels)
MOVE_S = 0.75               # analyzer step (ELL14, 22.5 deg + the 3 deg backoff)
AWG_LOAD_S = 0.6            # select a stored waveform + the step's bookkeeping


def periods(t, rep):
    """`t` s rounded up to whole trigger periods of `rep` s (at least one)."""
    return max(1, math.ceil(t / rep - 1e-6)) * rep


def shot_period(s, span_s=0.0):
    """Seconds between single shots: readout + the screen span, in periods."""
    rep = max(float(s.get("rep_s", 0.27)), 0.01)
    return periods(READOUT_S + float(span_s or 0), rep)


def _offsets(dark_mode, bg_mode, stray_vdiv, scan_name, drive):
    out = []
    for kind, mode in (("dark", dark_mode), ("background", bg_mode)):
        if mode == "measure":
            out.append({"kind": kind, "angle": None, "scan": scan_name, "x1": None, "x2": None,
                        "note": "PD covered" if kind == "dark" else "beam blocked"})
            if stray_vdiv and dark_mode == "measure" and bg_mode == "measure":
                out.append(dict(out[-1], note=out[-1]["note"]
                                + f", again at {stray_vdiv * 1e3:g} mV/div (stray light)"))
    return out


def ramp_scan(s, dark_mode="none", bg_mode="measure", stray_vdiv=None, drive=None,
              name="scan"):
    """A ramp scan's steps (the Ramp scan tab's settings `s`): the dark /
    background first, then the angles with the reference returns. drive:
    (x1, x2) deg when this window's AWG plays a ramp, else None."""
    angles = scanmod.ordered(scanmod.angle_list(s["start"], s["stop"], s["step"]), s["order"])
    steps = _offsets(dark_mode, bg_mode, stray_vdiv, name, drive)
    for st in scanmod.build_steps(angles, int(s.get("ref_every", 0)), s.get("ref_angle", 45.0)):
        steps.append({"kind": st["kind"], "angle": float(st["target"]), "scan": name,
                      "x1": drive[0] if drive else None, "x2": drive[1] if drive else None,
                      "shots": int(s["shots"])})
    for st in steps:
        st.setdefault("shots", int(s["shots"]))
        st.setdefault("rel", False)
    return steps


def sequence(s, ends, order, names, dark_mode="none", bg_mode="measure", stray_vdiv=None):
    """The AWG sequence's steps: one ramp scan per (X1, X2) end point at the
    Ramp scan tab's angles, interleaved (every ramp at one angle before the
    analyzer moves) or one setting at a time - as _run_seq takes them. The
    dark / background is taken once, in the first scan."""
    per = [ramp_scan(s, "none", "none", None, e, n) for e, n in zip(ends, names)]
    n = len(per[0])
    if str(order).startswith("one"):
        idx = [(k, i) for k in range(len(per)) for i in range(n)]
    else:
        idx = [(k, i) for i in range(n) for k in range(len(per))]
    steps = _offsets(dark_mode, bg_mode, stray_vdiv, names[0], None)
    for st in steps:
        st.update(shots=int(s["shots"]), rel=False)
    return steps + [per[k][i] for k, i in idx]


def fixed_rotations(p, biases, name="bias", ladder=4):
    """A Fixed rotations run's steps: the dark at every V/div of the null
    ladder (its length is only known once Imax is read; `ladder` is a
    guess), then per rotation 4 angles for the azimuth, `null_points`
    across +-`null_half_deg` around the crossed angle found, and the bright
    angle (crossed + 90). Angles are counted from that crossed angle (rel)."""
    from . import bias as biasmod
    split = float(p["split"])
    shots = int(p["shots"])
    steps = [{"kind": "dark", "angle": None, "rel": True, "x1": None, "x2": None,
              "shots": shots, "scan": name,
              "note": f"beam blocked, at ~{ladder + 1} V/div settings"}]
    order = biasmod.order_biases(biases, p.get("order", "up"))
    half, npts = float(p["null_half_deg"]), int(p["null_points"])
    offs = [(-half + 2 * half * k / (npts - 1)) if npts > 1 else 0.0 for k in range(npts)]
    for b in order:
        x1, x2 = b * split, b * (1 - split)
        common = {"x1": x1, "x2": x2, "shots": shots, "scan": name, "rel": True,
                  "note": f"rotation {b:g} deg"}
        for a in (-90.0, -45.0, 0.0, 45.0):
            steps.append(dict(common, kind="azimuth", angle=a))
        for o in offs:
            steps.append(dict(common, kind="null", angle=o))
        steps.append(dict(common, kind="bright", angle=90.0))
    return steps


def timeline(steps, s, settle_s=1.0, span_s=None):
    """Fill in t0 and dur (s): a move when the angle changes, a load and
    `settle_s` when what the AWG plays changes, then the shots - all on the
    trigger's periods (module constants). span_s: the scope screen's width
    (default s['span_s'], else 0). Returns the total (s)."""
    rep = max(float(s.get("rep_s", 0.27)), 0.01)
    span = float(s.get("span_s", 0.0) if span_s is None else span_s or 0.0)
    per = shot_period(s, span)
    avg = s.get("mode") == "average"
    t, last_a, last_d = 0.0, object(), None
    for st in steps:
        over = 0.0
        if st["angle"] is not None and st["angle"] != last_a:
            over += MOVE_S
            last_a = st["angle"]
        d = (st.get("x1"), st.get("x2"))
        if st["angle"] is not None and d != last_d and d != (None, None):
            over += AWG_LOAD_S + settle_s
            last_d = d
        n = int(st.get("shots", 1))
        if avg:
            dur = over + n * rep + int(s.get("blocks", 4)) * 0.8
        else:
            # the first shot waits out the step's own overhead too
            dur = periods(over + READOUT_S + span, rep) + (n - 1) * per
        st["t0"], st["dur"] = t, dur
        t += dur
    return t


def table(steps, limit=40):
    """Lines for the log: one per step."""
    out = [f"{'#':>4} {'t (min)':>8}  {'kind':10s} {'analyzer':>10} {'X1':>7} {'X2':>7} "
           f"{'shots':>5}  scan"]
    for i, st in enumerate(steps[:limit]):
        a = ("-" if st["angle"] is None else
             (f"{st['angle']:+.2f}*" if st.get("rel") else f"{st['angle']:.2f}"))
        x1 = "-" if st.get("x1") is None else f"{st['x1']:g}"
        x2 = "-" if st.get("x2") is None else f"{st['x2']:g}"
        out.append(f"{i + 1:4d} {st.get('t0', 0) / 60:8.2f}  {st['kind']:10s} {a:>10} {x1:>7} "
                   f"{x2:>7} {st.get('shots', ''):>5}  {st.get('scan', '')}"
                   + (f"  ({st['note']})" if st.get("note") else ""))
    if len(steps) > limit:
        out.append(f"  ... {len(steps) - limit} more")
    return out

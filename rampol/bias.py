"""Static bias points: the extinction ratio where the scope can see the null,
and the rotation-vs-voltage curve.

WHY. A ramp scan takes every analyzer angle at the V/div the brightest one
needs (1 V/div for ~5 V of light), so near a null the scope reads ~1 code: on
16-ms-spin-echo-test-4 the rest and hold minima were 2.6-3 mV, about one step
at 1 V/div, i.e. ER ~1800 is the SCOPE's floor there, not the light's. Mid-ramp
(ER 18-31) the minima were 166-281 mV and well resolved.

HOW. The AWG holds the EOMs at a fixed bias for several ms (a plateau with
raised-cosine edges, played on the bench trigger like the ILC's bursts), so the
light is static long enough to:
  1. find the polarization: 4 analyzer angles at the normal V/div, Malus fit
     on the plateau window -> azimuth psi, Imax;
  2. sit the scope at a SENSITIVE V/div and step the analyzer by small angles
     around the crossed position: I = Imin + K sin^2(theta - theta_n) -> the
     null angle theta_n and Imin directly;
  3. measure Imax at theta_n + 90 at the normal V/div.
ER = Imax / Imin with both measured (dark-subtracted at the V/div each was
taken at); K / (Imax - Imin) checks the Malus shape (1 for pure ellipticity
or depolarization; not 1 = something else, e.g. a beam that moves with the
analyzer). Over a list of biases this is ER(rotation) where the light can be
seen, and theta_n - 90 is the static polarization azimuth: psi(V), the
rotator's static transfer curve. Compared with the monitors' prediction it
shows whether the ramps' light-minus-monitor deviation (+-2-3 deg, ~90 deg
period on test-4) is a static property of the optics (a misaligned QWP) or
something only the ramps do.

The plateau's edges sweep the light through bright states, which overdrives
the scope at the null V/div: the MSO-X front end reads +8 % of the overdrive
0.3-1.3 ms later and ~2 % at 1.3-3.3 ms (2 Oct 2026). The analysis window
therefore starts `settle_ms` (4 ms) into the hold.

Safety: every plateau is checked with EOM-ILC's own limit check (AWG rail,
Trek input, HV, slew, current) before anything is uploaded; outputs are
switched on only after the user agrees and are OFF at the end however the run
ends.
"""
import datetime
import json
import math
import os
import time

import numpy as np

FORMAT = "rampol-bias/1"

# AWG -> monitor gain (monitor V per AWG V) and the monitor voltage for a 90
# deg rotation, per crystal: EOM-ILC's eomilc.config (optical calibration of
# 31 Aug - 1 Sep 2026), copied as fallbacks for when it cannot be loaded.
CHAN = {"EO1": {"gain": 0.5594, "v90": 5.1283, "awg": 1, "mon": "MonX1"},
        "EO2": {"gain": 0.5924, "v90": 5.1374, "awg": 2, "mon": "MonX2"}}

PLAN = {
    "biases": "0:180:15",       # target rotation, deg (the ILC target's sense)
    "order": "up",              # up | updown (the second pass reads hysteresis)
    "split": 0.5,               # fraction of each rotation put on EO1
    # X1 / X2 held separately (deg each; start:stop:step or a list; 'pairs'
    # takes them together, 'grid' every X1 with every X2). When x1 is given
    # these replace biases x split - the grid of 7 Oct 2026's asking.
    "x1": "", "x2": "", "how": "pairs",
    # the null's angle predicted from the previous point (crossed moves by
    # sense x the rotation change) instead of 4 coarse angles each time;
    # the 4 angles are taken when the prediction misses
    "predict_null": True, "sense": -1.0,
    # the azimuth against time at the two slope angles (null +- 45 deg):
    # (I+ - I-) / (I+ + I- - 2 Imin) = -sin 2 d(null), intensity cancelled,
    # read through the hold and `track_ms` after the fall (the slow
    # relaxation, tau ~70 ms on 7 Oct 2026)
    "track": True, "track_ms": 150.0,
    "lead_ms": 0.5, "rise_ms": 1.0, "hold_ms": 8.0, "tail_ms": 0.5,
    "dt_us": 2.0,
    "settle_ms": 4.0,           # into the hold before the window opens
    "null_half_deg": 3.0, "null_points": 9,
    "shots": 8, "dither_codes": 3, "points": 20000, "wait_s": 10.0,
    "awg_max": 9.6,             # V at the AWG, per channel
    "full_scale": 10.0,         # the fixed upload mapping (AMP = 2 x this)
    "upload_settle_s": 1.0,
    # the AWG level between bursts per channel (V): EOM-ILC's learned idle
    # trim parks the chain at 0 V (file zero parks it at -9 / -41 V)
    "idle": {"EO1": 0.0, "EO2": 0.0},
    "end": "off",               # off | park (idle waveform, outputs left ON)
}


# ------------------------------------------------------------------- plan
def parse_biases(text):
    """'0:180:15' (start:stop:step, stop included) or '0, 30, 90' -> list."""
    text = str(text).strip()
    if ":" in text:
        a, b, c = (float(x) for x in text.split(":"))
        if c <= 0:
            raise ValueError("bias step must be > 0")
        n = int(math.floor((b - a) / c + 1e-9)) + 1
        return [round(a + i * c, 6) for i in range(max(n, 0))]
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def order_biases(biases, order):
    if order == "updown":
        return list(biases) + list(reversed(biases))[1:]
    return list(biases)


def awg_volts(bias_deg, split, chan=CHAN):
    """{EO1: u1, EO2: u2}: AWG volts that put bias_deg of rotation on the pair,
    split[EO1] of it on EO1. Rotation per crystal = 90 x monitor / V90."""
    out = {}
    for name, frac in (("EO1", float(split)), ("EO2", 1.0 - float(split))):
        c = chan[name]
        out[name] = frac * bias_deg / 90.0 * c["v90"] / c["gain"]
    return out


def ends_volts(e1, e2, chan=CHAN):
    """{EO1, EO2}: AWG volts for e1 deg on X1 and e2 deg on X2."""
    return {"EO1": awg_volts(e1, 1.0, chan)["EO1"], "EO2": awg_volts(e2, 0.0, chan)["EO2"]}


def as_ends(item, split=0.5):
    """(x1, x2) deg for a point: a rotation (split between the crystals) or
    an (x1, x2) pair / {EO1, EO2} dict as given."""
    if isinstance(item, dict):
        return float(item.get("EO1", item.get("X1", 0.0))), float(item.get("EO2", item.get("X2", 0.0)))
    if isinstance(item, (tuple, list)):
        return float(item[0]), float(item[1])
    b = float(item)
    return b * float(split), b * (1.0 - float(split))


def ends_list(p):
    """The (x1, x2) points of a plan, in measuring order: p['x1'] / p['x2']
    (pairs or grid, as awg.parse_ends) when x1 is given, else the rotation
    list split between the crystals. 'updown' appends the reverse pass."""
    p = dict(PLAN, **p)
    if str(p.get("x1", "")).strip():
        from . import awg as awgmod
        pts = [(float(a), float(b)) for a, b in
               awgmod.parse_ends(p["x1"], p.get("x2") or "0", p.get("how", "pairs"))]
    else:
        pts = [as_ends(b, p["split"]) for b in parse_biases(p["biases"])]
    return order_biases(pts, p["order"])


def uses_pairs(p):
    return bool(str(dict(PLAN, **p).get("x1", "")).strip())


def plateau(amp, p, idle=0.0):
    """(t, u): idle lead, raised-cosine rise to idle + amp, hold, raised-
    cosine fall, idle tail, on p['dt_us'] - one sample longer than the
    segments, EOM-ILC's convention (11 ms at 2 us = 5501 points, FRQ
    90.893 Hz), so the ILC's FRQ check passes after a bias run. Both ends
    exactly idle (the AWG holds the first sample between bursts)."""
    dt = p["dt_us"] * 1e-6
    seg = [p["lead_ms"], p["rise_ms"], p["hold_ms"], p["rise_ms"], p["tail_ms"]]
    n = [int(round(x * 1e-3 / dt)) for x in seg]
    lead, rise, hold, fall, tail = n
    up = 0.5 - 0.5 * np.cos(np.pi * np.arange(rise) / rise)
    u = np.r_[np.zeros(lead), amp * up, np.full(hold, amp), amp * up[::-1],
              np.zeros(tail + 1)]
    u[[0, -1]] = 0.0
    return np.arange(len(u)) * dt, u + float(idle)


def eta(t0, done, total):
    """', ~N s left' from the pace so far (time.time() at t0, `done` of
    `total` steps finished); '' before the first step is done."""
    if done <= 0 or total <= done:
        return ""
    left = (time.time() - t0) / done * (total - done)
    return f", ~{left / 60:.1f} min left" if left >= 90 else f", ~{left:.0f} s left"


def timebase_around(win, margin=0.1):
    """(s/div, position) for REFerence LEFT that put the window `win` (s
    from the trigger) on screen with `margin` of its length either side, on
    a 1-2-5 step. A window before the trigger keeps the trigger on screen:
    the scope cannot delay the record back past it."""
    lo, hi = float(win[0]), float(win[1])
    if lo < 0:
        hi = max(hi, 0.0)
    span = (hi - lo) * (1 + 2 * margin)
    scale = _nice_up(max(span, 1e-6) / 10)
    t0 = 0.5 * (lo + hi) - 5 * scale
    return scale, t0 + scale          # LEFT: the record starts one division before


class window_timebase:
    """The timebase zoomed onto a measurement window for the duration, and
    put back after: a 50 ms spin-echo record read for a 1 ms window spends
    its points and its readout on what is thrown away.

        with window_timebase(link, (t0, t1), log):
            ... find_extremum(..., window_s=(t0, t1)) ...
    """

    KEYS = (":TIMebase:SCALe", ":TIMebase:POSition", ":TIMebase:REFerence")

    def __init__(self, link, win, log=print, margin=0.1):
        self.link, self.win, self.log, self.margin = link, win, log, margin
        self.saved = {}

    def __enter__(self):
        sc = self.link.scope
        self.saved = {k: sc.get(k) for k in self.KEYS}
        scale, pos = timebase_around(self.win, self.margin)
        sc.put(":TIMebase:REFerence", "LEFT")
        sc.put(":TIMebase:SCALe", f"{scale:.6g}")
        sc.put(":TIMebase:POSition", f"{pos:.6g}")
        t0 = pos - scale
        self.log(f"  timebase {scale*1e3:g} ms/div around the window "
                 f"{self.win[0]*1e3:.2f}..{self.win[1]*1e3:.2f} ms (record "
                 f"{t0*1e3:.2f}..{(t0 + 10*scale)*1e3:.2f} ms)")
        return t0, t0 + 10 * scale

    def __exit__(self, *exc):
        sc = self.link.scope
        for k, v in self.saved.items():
            if v is not None:
                try:
                    sc.put(k, v)
                except Exception as e:
                    self.log(f"  could not restore {k} ({e})")
        return False


def plateau_wave(item, p):
    """A point's plateau on both channels, as an awg.Wave - what a bias run
    plays, and what its dry run checks. `item`: a rotation (deg, split as
    the plan says) or an (x1, x2) pair."""
    from . import awg as awgmod
    p = dict(PLAN, **p)
    idle = p.get("idle") or {}
    e1, e2 = as_ends(item, p["split"])
    volts = ends_volts(e1, e2)
    u, t = {}, None
    for name in CHAN:
        t, u[name] = plateau(volts[name], p, float(idle.get(name, 0.0)))
    label = (f"bias X1 {e1:g} / X2 {e2:g} deg" if isinstance(item, (tuple, list, dict))
             else f"bias {float(item):g} deg")
    w = awgmod.Wave(t, u, p["dt_us"] * 1e-6, label,
                    hold=((p["lead_ms"] + p["rise_ms"]) * 1e-3,
                          (p["lead_ms"] + p["rise_ms"] + p["hold_ms"]) * 1e-3),
                    rotation=e1 + e2, source="bias")
    w.ends = {"EO1": e1, "EO2": e2}
    return w


def windows(p):
    """(hold window, idle window) in seconds after the trigger."""
    t_hold = (p["lead_ms"] + p["rise_ms"]) * 1e-3
    w = (t_hold + p["settle_ms"] * 1e-3, t_hold + (p["hold_ms"] - 0.2) * 1e-3)
    idle = (-1.0, (p["lead_ms"] - 0.05) * 1e-3)
    if w[1] - w[0] < 0.5e-3:
        raise ValueError(f"the hold ({p['hold_ms']} ms) leaves under 0.5 ms after "
                         f"the {p['settle_ms']} ms settle")
    return w, idle


def check_plateaus(items, p, eomilc=None):
    """[(item, {ch: u peak}, report text)] and raises ValueError on the first
    that fails the AWG cap or (with eomilc) the Trek chain's limit check.
    `items`: rotations (deg) or (x1, x2) pairs."""
    out = []
    dt = p["dt_us"] * 1e-6
    if not 0.0 <= float(p["split"]) <= 1.0:
        raise ValueError(f"split {p['split']} must be between 0 and 1")
    idle = p.get("idle") or {}
    seen = []
    for b in items:
        e1, e2 = as_ends(b, p["split"])
        if (e1, e2) in seen:
            continue
        seen.append((e1, e2))
        what = f"bias {b:g} deg" if not isinstance(b, (tuple, list, dict)) else \
            f"bias X1 {e1:g} / X2 {e2:g} deg"
        if e1 < 0 or e2 < 0:
            raise ValueError(f"{what}: negative biases are not driven "
                             f"(the ramps' drives are unipolar)")
        v = ends_volts(e1, e2)
        msgs = []
        for name, amp in v.items():
            i0 = float(idle.get(name, 0.0))
            if abs(i0) > 0.1:
                raise ValueError(f"{name} idle {i0*1e3:+.0f} mV: past the 100 mV idle cap")
            if abs(amp + i0) > p["awg_max"]:
                raise ValueError(
                    f"{what} needs {amp + i0:.2f} V on {name} at the AWG, past "
                    f"the {p['awg_max']:g} V cap - lower the bias or change the split")
            if eomilc is not None and abs(amp) > 0:
                from eomilc.config import CHANNELS
                from eomilc.ilc import check_limits
                ch = CHANNELS[name]
                _, u = plateau(amp, p, i0)
                rep = check_limits(u, u * CHAN[name]["gain"], dt, ch, ch.limits)
                if not rep.ok:
                    raise ValueError(f"{what}, {name}: {rep}")
                msgs.append(f"{name} {rep.messages[-1]}")
        out.append((b, v, "; ".join(msgs)))
    return out


# --------------------------------------------------------------- analysis
def malus4(theta_deg, I):
    """I = a0 + c2 cos 2theta + s2 sin 2theta through the points: (psi deg,
    Imax, Imin) with psi the transmission maximum."""
    r = np.deg2rad(np.asarray(theta_deg, float))
    A = np.column_stack([np.ones_like(r), np.cos(2 * r), np.sin(2 * r)])
    a0, c2, s2 = np.linalg.lstsq(A, np.asarray(I, float), rcond=None)[0]
    B = math.hypot(c2, s2)
    return math.degrees(0.5 * math.atan2(s2, c2)), a0 + B, a0 - B


def fit_null(theta_deg, I, sem=None):
    """I = imin + k sin^2(theta - theta_n) through a scan around the null.
    Returns dict(theta_n, imin, k, sig_theta_n, sig_imin, sig_k, rms, chi2,
    inside: theta_n within the scanned range)."""
    from scipy.optimize import least_squares
    th = np.asarray(theta_deg, float)
    I = np.asarray(I, float)
    w = 1.0 / np.maximum(np.asarray(sem, float), 1e-12) if sem is not None else np.ones_like(I)
    # start: a parabola in degrees
    c = np.polyfit(th - th.mean(), I, 2)
    tn0 = th.mean() - c[1] / (2 * c[0]) if c[0] > 0 else th[np.argmin(I)]
    tn0 = float(np.clip(tn0, th.min(), th.max()))
    k0 = max(c[0] * (180 / np.pi) ** 2, 1e-9)
    # the floor starts at the lowest reading, NOT clipped at 0: Find max fits
    # -I, whose floor is near -Imax, and a start at 0 with a 1e-6 step scale
    # collapsed the fit (k -> 0, the angle anywhere) in 9 of 40 simulated
    # runs (6 Oct 2026)
    i0 = float(np.min(I))
    i_scale = max(abs(i0), float(np.ptp(I)), 1e-6)

    def resid(p):
        tn, imin, k = p
        return (imin + k * np.sin(np.deg2rad(th - tn)) ** 2 - I) * w
    r = least_squares(resid, [tn0, i0, k0], x_scale=[0.1, i_scale, k0])
    J = r.jac
    dof = max(len(th) - 3, 1)
    s2 = float(r.fun @ r.fun) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * (s2 if sem is None else max(s2, 1.0))
        sig = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sig = np.full(3, np.nan)
    tn, imin, k = r.x
    model = imin + k * np.sin(np.deg2rad(th - tn)) ** 2
    return {"theta_n": float(tn), "imin": float(imin), "k": float(k),
            "sig_theta_n": float(sig[0]), "sig_imin": float(sig[1]),
            "sig_k": float(sig[2]), "rms": float(np.std(I - model)),
            "chi2": float(s2), "inside": bool(th.min() <= tn <= th.max())}


def er_point(imin, sig_imin, imax, sig_imax=0.0, k=None):
    """ER = Imax/Imin with its error; a lower bound when Imin is not resolved
    (< 2 sigma): ER > Imax/(2 sigma). Also the Malus check K/(Imax-Imin) and the
    ellipticity a fully polarized beam with this ER would have."""
    out = {"imin": imin, "sig_imin": sig_imin, "imax": imax, "sig_imax": sig_imax}
    if sig_imin > 0 and imin < 2 * sig_imin:
        out.update(er=None, er_lower=imax / (2 * sig_imin), sig_er=None)
    else:
        er = imax / imin
        rel = math.hypot(sig_imin / imin, (sig_imax / imax) if imax else 0.0)
        out.update(er=er, er_lower=None, sig_er=er * rel)
    if k is not None and imax > imin:
        out["malus_ratio"] = k / (imax - imin)
    e = out["er"] or out["er_lower"]
    out["ellipticity_deg"] = math.degrees(math.atan(1 / math.sqrt(e))) if e else None
    return out


def transfer(points, ref=0):
    """The static transfer curve from finished points: psi (deg, unwrapped,
    from the null angle) against the rotation the monitors predict, both
    relative to the point with bias `ref`. Fits psi = g x phi_mon + c; returns
    dict(phi_mon, rot_light, resid, gain, offset, v90_scale, up/down split)."""
    pts = [p for p in points if p.get("theta_n") is not None]
    if len(pts) < 2:
        return None
    psi = np.unwrap(np.deg2rad(2 * np.array([p["theta_n"] - 90 for p in pts]))) / 2
    psi = np.degrees(psi)
    phi = np.array([p["phi_mon"] for p in pts])
    b = np.array([p["bias"] for p in pts])
    i0 = int(np.argmin(np.abs(b - ref)))
    rot = psi - psi[i0]
    phi = phi - phi[i0]
    # the light's rotation runs against the analyzer's sense on this bench
    # (test-4: negative); fit the sign rather than assume it
    sgn = 1.0 if np.dot(rot, phi) >= 0 else -1.0
    rot = sgn * rot
    A = np.column_stack([phi, np.ones_like(phi)])
    (g, c), *_ = np.linalg.lstsq(A, rot, rcond=None)
    res = rot - (g * phi + c)
    out = {"bias": b.tolist(), "phi_mon": phi.tolist(), "rot_light": rot.tolist(),
           "resid": res.tolist(), "gain": float(g), "offset": float(c), "sign": sgn,
           "rms_resid": float(np.std(res)),
           "dir": [p.get("dir", "up") for p in pts]}
    return out


def _expfit(tt, y):
    """y = A exp(-(tt - tt[0]) / tau) + c by a grid over tau (ms), linear in
    A and c. Returns (tau, A, c)."""
    best = None
    for tau in np.geomspace(3.0, 500.0, 120):
        M = np.column_stack([np.exp(-(tt - tt[0]) / tau), np.ones_like(tt)])
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        ss = float(np.sum((y - M @ coef) ** 2))
        if best is None or ss < best[0]:
            best = (ss, tau, float(coef[0]), float(coef[1]))
    return best[1:]


def track_azimuth(t, I_plus, I_minus, imin, mons, ref, sense=-1.0, chan=CHAN):
    """The azimuth against time from two slope angles (a null +- 45 deg):
    with I(theta) = Imin + K sin^2(theta - theta_n) and the null moved by d,
    I+ - I- = -K sin 2d and I+ + I- = 2 Imin + K, so
    d = -asin((I+ - I-) / (I+ + I- - 2 Imin)) / 2 - the intensity cancels.
    Valid while the light is within 45 deg of that null, so a hold past 45
    deg needs one pair at the hold's null (the hold) and one at the rest
    null (the lead and the tail after the fall). `ref` (t0, t1) is the
    window both the azimuth and the monitors' rotation (sum of 90 x V / V90)
    are referenced to. Returns dict of arrays: dpsi, mon_rot, lm (= dpsi -
    sense x rotation, deg), k (V), a0_rel, and 'valid' (|ratio| < 0.95)."""
    t = np.asarray(t, float)
    ip, im = np.asarray(I_plus, float), np.asarray(I_minus, float)
    k = ip + im - 2.0 * float(imin)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(k > 0, (ip - im) / np.where(k > 0, k, 1.0), 0.0)
    dpsi = -0.5 * np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    m = (t >= ref[0]) & (t <= ref[1])
    if m.sum() < 5:
        m = np.ones(len(t), bool)
    rot = np.zeros_like(t)
    for name, c in chan.items():
        r = c["mon"]
        if r in mons:
            mm = np.asarray(mons[r], float)
            rot += 90.0 * (mm - mm[m].mean()) / c["v90"]
    dpsi = dpsi - dpsi[m].mean()
    lm = dpsi - float(sense) * rot
    a0 = ip + im
    return {"dpsi": dpsi, "mon_rot": rot, "lm": lm - lm[m].mean(), "k": k,
            "a0_rel": a0 / a0[m].mean() - 1.0, "valid": np.abs(ratio) < 0.95}


def track_summary(t, lm_hold, w, t_fall, lm_rest=None, valid_rest=None):
    """Numbers from light minus monitors (deg) against t (s): the creep
    slope through the hold window `w` (mdeg/ms) from the hold pair; from
    the rest pair (else the hold pair) the value 1 ms after the fall ends
    at `t_fall`, the extreme after it (mdeg, at ms), and the relaxation time
    constant (ms) from 1.5 ms after the fall to the end with the level it
    relaxes to - the time constant only when the fitted amplitude stands
    above the trace's own scatter."""
    t = np.asarray(t, float)
    out = {}
    m = (t >= w[0]) & (t <= w[1])
    if m.sum() >= 5:
        pf = np.polyfit(t[m] * 1e3, lm_hold[m] * 1e3, 1)
        out["hold_slope_mdeg_ms"] = float(pf[0])
        out["hold_scatter_mdeg"] = float(np.std(lm_hold[m] - np.polyval(pf, t[m] * 1e3) / 1e3) * 1e3)
    lm = lm_rest if lm_rest is not None else lm_hold
    out["after_from"] = "rest pair" if lm_rest is not None else "hold pair"
    a = t >= t_fall + 1.0e-3
    if valid_rest is not None:
        a &= valid_rest
    if a.sum() >= 20:
        out["after_1ms_mdeg"] = float(np.interp(t_fall + 1.0e-3, t, lm) * 1e3)
        j = int(np.argmax(np.abs(lm[a])))
        out["after_extreme_mdeg"] = float(lm[a][j] * 1e3)
        out["after_extreme_ms"] = float((t[a][j] - t_fall) * 1e3)
        b = a & (t >= t_fall + 1.5e-3)
        if b.sum() >= 30 and (t[b][-1] - t[b][0]) > 10e-3:
            tau, A_, c_ = _expfit(t[b] * 1e3, lm[b] * 1e3)
            noise = float(np.std(np.diff(lm[b] * 1e3))) / math.sqrt(2)
            out["after_scatter_mdeg"] = noise
            if abs(A_) > 3 * noise:
                out.update(tau_ms=float(tau), tau_amp_mdeg=float(A_),
                           tau_level_mdeg=float(c_))
    return out


def sense_from(points, default=-1.0):
    """The light's sense in the analyzer frame learned from measured
    points: the null moved by (theta_n - theta_n_prev) for a rotation
    change (bias - bias_prev); the sign of their ratio over the last pair
    at least 10 deg apart, when the ratio is near +-1 (the sim's bench runs
    +1, the real one -1 on 7 Oct 2026). Else `default`."""
    pts = [p for p in points if p.get("theta_n") is not None]
    for a, b in zip(pts[-2::-1], pts[::-1]):
        db = b["bias"] - a["bias"]
        if abs(db) < 10:
            continue
        dth = (b["theta_n"] - a["theta_n"] + 90.0) % 180.0 - 90.0
        ratio = dth / db
        if 0.5 < abs(ratio) < 1.5:
            return 1.0 if ratio > 0 else -1.0
        break
    return float(default)


def load_tracks(folder, points=None):
    """{point index: dict(t, dpsi, mon_rot, lm, a0_rel)} from the point_NN.npz
    files of a run that tracked the azimuth."""
    out = {}
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("point_") and fn.endswith(".npz")):
            continue
        i = int(fn[6:8])
        if points is not None and i not in points:
            continue
        with np.load(os.path.join(folder, fn)) as z:
            if "track_t" in z.files:
                out[i] = {k[6:]: z[k] for k in z.files if k.startswith("track_")}
    return out


# ------------------------------------------------------------- the run
class BiasRun:
    """One bias-point measurement on the bench. `link` is hw.ScopeLink, `rot`
    hw.Rotator, `awg` a BK4063B (or the simulator's), `roles` {role: channel}.
    `ask(title, text)` -> bool comes from the GUI thread (blocking is fine: it
    is called on the worker). `on_point(point)` gets every finished point."""

    def __init__(self, folder, name, link, rot, awg, roles, plan=None, log=print,
                 cancelled=None, ask=None, on_point=None, progress=None,
                 eomilc=None, ilc_bench=None, provenance=None, session=None,
                 resume=False):
        self.folder = os.path.join(folder, name)
        self.provenance = provenance
        # resume: a run folder with the same points whose darks are on
        # record continues from its first unmeasured point (a 361-point grid
        # is hours; a stop or an error must not cost the darks and the rest)
        self.resume = resume
        self.name = name
        self.link, self.rot, self.awg = link, rot, awg
        self.roles = dict(roles)
        self.p = dict(PLAN, **(plan or {}))
        self.log = log
        self.cancelled = cancelled or (lambda: False)
        self.ask = ask or (lambda title, text: True)
        self.on_point = on_point
        self.progress = progress or (lambda done, total, text: None)
        self.eomilc = eomilc
        self.ib = ilc_bench
        from . import awg as awgmod
        # the window passes its session (with its never-float and dry-run
        # policy); a script calling this directly gets a plain one
        self.sess = session or awgmod.Session(
            awg, ilc_bench, log=log, never_float=self.p.get("end") == "park",
            require_dry_run=False)
        self.points = []
        self.dark = {}
        self.manifest = None

    # -- helpers ----------------------------------------------------------
    def _check(self):
        if self.cancelled():
            from .hw import Cancelled
            raise Cancelled()

    def _chans(self):
        # Find and the Malus scan look at the PD alone: reading the monitors
        # as well tripled every readout for nothing
        want = ["PD"] if getattr(self, "pd_only", False) else \
            ["PD", "MonX1", "MonX2", "CmdX1", "CmdX2"]
        return [self.roles[r] for r in want if r in self.roles]

    def _acquire(self, angle, pd_setting):
        """Analyzer to `angle` (from below), PD channel to (V/div, offset),
        `shots` single HRES shots; returns {role: (mean trace, per-shot
        window means)} and t."""
        self._check()
        if angle is not None:
            self.rot.approach(angle)
        pd = self.roles["PD"]
        self.link.set_channel(pd, *pd_setting)
        chans = self._chans()
        p = self.p
        acc = {ch: [] for ch in chans}
        tt = {}

        def on_block(k, recs, hits):
            for ch, rec in recs.items():
                acc[ch].append(rec.v())
                tt[ch] = rec.t()
        self.link.acquire_blocks(chans, "single", int(p["shots"]), int(p["shots"]),
                                 dither_codes=int(p["dither_codes"]),
                                 points=int(p["points"]), wait_s=float(p["wait_s"]),
                                 cancelled=self.cancelled, on_block=on_block)
        t = tt[pd]
        inv = {ch: r for r, ch in self.roles.items()}
        out = {}
        for ch in chans:
            n = min(len(x) for x in acc[ch])
            out[inv[ch]] = np.array([x[:n] for x in acc[ch]])
        return t[:out["PD"].shape[1]], out

    def _window(self, t, stack, w):
        m = (t >= w[0]) & (t <= w[1])
        if m.sum() < 5:
            raise RuntimeError(f"the scope record ({t[0]*1e3:.2f}..{t[-1]*1e3:.2f} "
                               f"ms) misses the hold window {w[0]*1e3:.2f}.."
                               f"{w[1]*1e3:.2f} ms - set the timebase to cover the "
                               f"plateau (the run does this itself unless told not to)")
        per = stack[:, m].mean(axis=1)
        return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else 0.0

    def _clipped(self, stack, setting, t, w):
        return self._clip_side(stack, setting, t, w) is not None

    def _clip_side(self, stack, setting, t, w):
        """'high' or 'low' when any shot leaves the screen in the window,
        else None (the screen is +-4 div about the offset)."""
        sc, off = setting
        m = (t >= w[0]) & (t <= w[1])
        x = stack[:, m]
        if (x > off + 3.9 * sc).any():
            return "high"
        if (x < off - 3.9 * sc).any():
            return "low"
        return None

    # -- the AWG -----------------------------------------------------------
    def _wave(self, bias_deg):
        return plateau_wave(bias_deg, self.p)

    def _play(self, bias_deg):
        """Load a bias's plateau. Under the 'off' policy the outputs go off
        for the change and back on (a selection switched mid-burst steps
        the drive by the bias difference, which no limit check sees)."""
        keep = self.p.get("end") == "park"
        was_on = bool(self.sess.owned)
        self.sess.load(self._wave(bias_deg), keep_on=keep)
        if was_on and not all(self.sess.outputs().values()):
            self.sess.on()

    # -- the run -----------------------------------------------------------
    def _resume_state(self, ends):
        """The earlier run's points, darks and null settings when this folder
        holds a run of the same points with its darks on record; else None."""
        path = os.path.join(self.folder, "bias.json")
        if not self.resume or not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
        same = [tuple(x) for x in man.get("ends", [])] == [tuple(x) for x in ends]
        if man.get("format") != FORMAT or not same or not man.get("dark") \
                or not man.get("null_settings"):
            self.log(f"  {self.name}: not resumable (another plan, or no darks on record) "
                     f"- starting over")
            return None
        return man

    def run(self):
        p = self.p
        if "PD" not in self.roles:
            raise RuntimeError("no channel has the PD role")
        ends = ends_list(p)
        biases = [e1 + e2 for e1, e2 in ends]
        t_run = time.time()
        checks = check_plateaus(ends, p, self.eomilc)
        t_rec, _ = plateau(0.0, p)
        period = float(t_rec[-1] + p["dt_us"] * 1e-6)
        w, idle = windows(p)
        t_fall = (p["lead_ms"] + 2 * p["rise_ms"] + p["hold_ms"]) * 1e-3
        track = bool(p.get("track")) and float(p.get("track_ms") or 0) > 0
        span = period + (float(p["track_ms"]) * 1e-3 if track else 0.0)
        os.makedirs(self.folder, exist_ok=True)
        old = self._resume_state(ends)
        if old is not None:
            self.manifest = old
            self.manifest["resumed"] = (self.manifest.get("resumed") or []) + [
                datetime.datetime.now().isoformat(timespec="seconds")]
            self.manifest.pop("finished", None)
            self.points = list(old.get("points", []))
            self.dark = {k: tuple(v) for k, v in old["dark"].items()}
            null_sets = [tuple(x) for x in old["null_settings"]]
            coarse = tuple(old["coarse"])
            self.log(f"Bias run {self.name} resumed: {len(self.points)} of {len(ends)} points "
                     f"on record, darks from {old.get('created', '')}")
        else:
            self.manifest = {"format": FORMAT, "name": self.name, "plan": p,
                             "created": datetime.datetime.now().isoformat(timespec="seconds"),
                             "roles": self.roles, "window_s": w, "biases": biases,
                             "ends": [list(e) for e in ends], "t_fall_s": t_fall,
                             "limit_checks": [c[2] for c in checks], "points": [],
                             "provenance": self.provenance}
            null_sets, coarse = None, None
            lender = p.get("darks_from")
            if lender:
                # another run's darks (the same bench, the same evening: one
                # beam block serves the night), with its V/div settings
                lp = os.path.join(os.path.dirname(self.folder), str(lender), "bias.json")
                with open(lp, encoding="utf-8") as fh:
                    lm = json.load(fh)
                if not (lm.get("dark") and lm.get("null_settings") and lm.get("coarse")):
                    raise ValueError(f"{lender}: no darks and null settings on record to lend")
                self.dark = {k: tuple(v) for k, v in lm["dark"].items()}
                null_sets = [tuple(x) for x in lm["null_settings"]]
                coarse = tuple(lm["coarse"])
                self.manifest["dark"] = dict(lm["dark"])
                self.manifest["null_settings"] = list(lm["null_settings"])
                self.manifest["coarse"] = list(lm["coarse"])
                self.manifest["darks_from"] = {"run": str(lender),
                                               "measured": lm.get("created", "")}
                self.log(f"  darks from {lender} ({lm.get('created', '')}): "
                         + ", ".join(f"{k}: {v[0]*1e3:+.3f} mV" for k, v in self.dark.items()))
        self._save()
        self.log(f"Bias run {self.name}: {len(ends)} points, record "
                 f"{period*1e3:.2f} ms, window {w[0]*1e3:.2f}-{w[1]*1e3:.2f} ms"
                 + (f", azimuth tracked to {span*1e3:.0f} ms" if track else ""))
        sc = self.link.scope
        saved_tb = {k: sc.get(k) for k in (":TIMebase:SCALe", ":TIMebase:POSition",
                                           ":TIMebase:REFerence")}
        pd = self.roles["PD"]
        saved_pd = self.link.channel_state([pd])[pd]
        if coarse is None:
            coarse = saved_pd
        on = False
        try:
            # the scope covers the record (and the tracked tail): 10 divisions
            # from the lead's start
            div = _nice_up(span * 1.05 / 10)
            sc.put(":TIMebase:REFerence", "LEFT")
            sc.put(":TIMebase:SCALe", f"{div:.6g}")
            sc.put(":TIMebase:POSition", f"{-0.2e-3 + div:.6g}")   # LEFT: start 1 div before
            self.log(f"  scope {div*1e3:g} ms/div from -0.2 ms; PD coarse "
                     f"{coarse[0]:g} V/div offset {coarse[1]:g} V")
            on = True                  # from here every exit ends the AWG
            first = len(self.points)
            if first >= len(ends):
                self.log("  nothing left to measure")
                return self.points
            self._play(ends[first])
            end = "parked at idle" if p.get("end") == "park" else "OFF"
            if not self.ask("Fixed rotations", "The AWG holds the plateaus. Switch "
                            f"both outputs ON now? (At the end they go {end}.)"):
                raise RuntimeError("outputs left off - nothing measured")
            self.sess.on()
            time.sleep(p["upload_settle_s"])

            if null_sets is None:
                # Imax at the first bias from 4 angles (one angle can sit at
                # the null), then the dark at every V/div used
                th4 = (0.0, 45.0, 90.0, 135.0)
                I4 = []
                for a in th4:
                    t, s = self._acquire(a, coarse)
                    I4.append(self._window(t, s["PD"], w)[0])
                imax_est = max(malus4(th4, I4)[1], 1e-3)
                ladder = _ladder(imax_est, p["null_half_deg"], coarse[0])
                self.log(f"  PD up to {imax_est:.3f} V; null V/div ladder "
                         + ", ".join(f"{x*1e3:g}m" for x in ladder))
                settings = [coarse]
                if not self.ask("Dark", "Block the beam before the analyzer, then OK. "
                                f"(Darks at {len(ladder) + 1} V/div settings, "
                                f"~{(len(ladder) + 1) * p['shots'] / 3.7:.0f} s.)"):
                    raise RuntimeError("no dark - Imin cannot be dark-corrected")
                t, s = self._acquire(None, coarse)
                d0, d0s = self._window(t, s["PD"], w)
                self.dark[_key(coarse)] = (d0, d0s)
                for vd in ladder:
                    st_ = (vd, d0 + 3 * vd)          # 0 V of light 3 div below centre
                    t, s = self._acquire(None, st_)
                    self.dark[_key(st_)] = self._window(t, s["PD"], w)
                    settings.append(st_)
                self.log("  dark: " + ", ".join(f"{k}: {v[0]*1e3:+.3f} mV"
                                                for k, v in self.dark.items()))
                if not self.ask("Dark", "Dark done. Unblock the beam, then OK."):
                    raise RuntimeError("stopped after the dark")
                null_sets = settings[1:]
                # on record at once: a resume needs them more than anything
                self.manifest["dark"] = {k: list(v) for k, v in self.dark.items()}
                self.manifest["null_settings"] = [list(s_) for s_ in null_sets]
                self.manifest["coarse"] = list(coarse)
                self._save()

            prev = self.points[-1] if self.points else None
            for i in range(first, len(ends)):
                e = ends[i]
                b = biases[i]
                self._check()
                d = "up" if (p["order"] != "updown" or i < len(ends) // 2 + 1) else "down"
                self.progress(i, len(ends), f"X1 {e[0]:g} / X2 {e[1]:g} deg ({i + 1}/{len(ends)}"
                              f"{eta(t_run, i - first, len(ends) - first)})")
                if i > first:
                    self._play(e)
                    time.sleep(p["upload_settle_s"])
                pt = self._point(i, e, d, coarse, null_sets, w, idle, prev,
                                 track=track, t_fall=t_fall)
                prev = pt
                self.points.append(pt)
                self.manifest["points"].append(pt)
                self._save()
                if self.on_point:
                    self.on_point(pt)
            self.progress(len(ends), len(ends), "bias points done")
        finally:
            if on:
                self.sess.end(p.get("end", "off"))
                self.sess.forget()
            try:
                self.link.set_channel(pd, *saved_pd)
                for k, v in saved_tb.items():
                    if v is not None:
                        sc.put(k, v)
            except Exception as exc:
                self.log(f"  could not restore the scope ({exc})")
            if self.manifest is not None:
                self.manifest["transfer"] = transfer(self.points)
                self.manifest["dark"] = {k: list(v) for k, v in self.dark.items()}
                if len(self.points) == len(ends):
                    self.manifest["finished"] = datetime.datetime.now().isoformat(
                        timespec="seconds")
                self._save()
        return self.points

    def _coarse_azimuth(self, base, coarse, w, dark_c, mons, traces):
        """4 angles at the coarse V/div, Malus through them: (psi, imax, imin)."""
        th4 = [(base + a) % 180 for a in (0, 45, 90, 135)]
        I4 = []
        for a in th4:
            t, s = self._acquire(a, coarse)
            I4.append(self._window(t, s["PD"], w)[0] - dark_c)
            for r in ("MonX1", "MonX2"):
                if r in s:
                    mons.setdefault(r, []).append(s[r])
            traces[f"coarse_{a:.2f}"] = s["PD"].mean(axis=0)
        return malus4(th4, I4)

    def _null_scan(self, b, theta0, est, coarse, null_sets, w, traces):
        """The analyzer stepped across the null at the most sensitive V/div
        that holds the scan; re-centred when the fit lands off centre.
        Returns (fit, scan dict, setting, converged)."""
        p = self.p
        setting = next((s_ for s_ in null_sets if est < 5.5 * s_[0]), coarse)
        fit, scan, recentred = None, None, 0
        while True:
            offs = np.linspace(-p["null_half_deg"], p["null_half_deg"], int(p["null_points"]))
            th = theta0 + offs
            I, S, clipped = [], [], False
            for a in th:
                t, s = self._acquire(a % 180, setting)
                if setting != coarse and self._clipped(s["PD"], setting, t, w):
                    clipped = True
                    break
                m, se = self._window(t, s["PD"], w)
                I.append(m - self.dark[_key(setting)][0])
                S.append(math.hypot(se, self.dark[_key(setting)][1]))
                traces[f"null_{a:.3f}"] = s["PD"].mean(axis=0)
            if clipped:
                bigger = [s_ for s_ in null_sets if s_[0] > setting[0]]
                setting = bigger[0] if bigger else coarse
                self.log(f"  {b}: off screen at the null V/div, now "
                         f"{setting[0]*1e3:g} mV/div")
                continue
            fit = fit_null(th, I, S)
            scan = {"theta": th.tolist(), "I": I, "sem": S, "vdiv": setting[0]}
            ok = fit["inside"] and abs(fit["theta_n"] - theta0) < 0.6 * p["null_half_deg"]
            if ok or recentred >= 2:
                return fit, scan, setting, ok
            theta0 = fit["theta_n"]
            recentred += 1
            self.log(f"  {b}: null at {theta0:.2f} deg, off centre - again")

    def _point(self, i, e, direction, coarse, null_sets, w, idle, prev, track=False,
               t_fall=None):
        """One point: the null (angle and Imin), the bright angle (Imax),
        the monitors, and with `track` the azimuth against time. `e` is the
        (x1, x2) pair, `prev` the previous point (None for the first)."""
        p = self.p
        e1, e2 = float(e[0]), float(e[1])
        b = e1 + e2
        what = f"X1 {e1:g} / X2 {e2:g}"
        dark_c = self.dark[_key(coarse)][0]
        mons = {}
        traces = {}
        # 1. where is the polarization: predicted from the previous point
        # (the null moves by sense x the rotation change), else 4 angles
        psi = imax4 = imin4 = None
        predicted = False
        sense = sense_from(self.points, float(p.get("sense", -1.0)))
        if prev is not None and p.get("predict_null", True) and prev.get("theta_n") is not None:
            theta0 = (prev["theta_n"] + sense * (b - prev["bias"])) % 180
            imax4 = prev.get("imax") or prev.get("imax4")
            imin4 = max(prev.get("imin", 0.0) or 0.0, 0.0)
            predicted = True
        else:
            base = 0.0 if prev is None else prev.get("psi", 0.0)
            psi, imax4, imin4 = self._coarse_azimuth(base, coarse, w, dark_c, mons, traces)
            theta0 = (psi + 90) % 180
        # 2. around the null at the most sensitive setting that holds it
        est = max(imin4, 0.0) + imax4 * math.sin(math.radians(p["null_half_deg"] + 1)) ** 2
        fit, scan, setting, ok = self._null_scan(what, theta0, est, coarse, null_sets, w, traces)
        if not ok and predicted:
            # the prediction missed: find the azimuth the long way and scan again
            self.log(f"  {what}: predicted null {theta0:.2f} deg not found - 4 angles")
            psi, imax4, imin4 = self._coarse_azimuth(theta0 - 90, coarse, w, dark_c, mons, traces)
            theta0 = (psi + 90) % 180
            est = max(imin4, 0.0) + imax4 * math.sin(math.radians(p["null_half_deg"] + 1)) ** 2
            fit, scan, setting, ok = self._null_scan(what, theta0, est, coarse, null_sets, w,
                                                     traces)
            predicted = False
        # 3. Imax at the bright angle
        t, s = self._acquire((fit["theta_n"] + 90) % 180, coarse)
        imax, imax_s = self._window(t, s["PD"], w)
        imax -= dark_c
        if self._clipped(s["PD"], coarse, t, w):
            self.log(f"  {what}: the bright reading clips at the coarse V/div")
        traces["bright"] = s["PD"].mean(axis=0)
        for r in ("MonX1", "MonX2"):
            if r in s:
                mons.setdefault(r, []).append(s[r])
        # 4. the azimuth against time: two slope angles at the hold's null
        # (the hold itself), and for a hold past 25 deg two more at the rest
        # null (the lead and the tail after the fall, which the hold's pair
        # cannot read beyond 45 deg from it)
        tr = None
        if track:
            # the rest null: the point measured at zero rotation, else from
            # this one and the sense
            rest_null = next((q["theta_n"] for q in self.points
                              if abs(q["bias"]) < 1e-6 and q.get("theta_n") is not None), None)
            if rest_null is None:
                rest_null = fit["theta_n"] - sense * b
            pairs = [("hold", fit["theta_n"])]
            if abs(b) > 25.0:
                pairs.append(("rest", rest_null))
            got = {}
            for which, centre in pairs:
                slope = {}
                for sign, name in ((+1, "plus"), (-1, "minus")):
                    t, s = self._acquire((centre + sign * 45.0) % 180, coarse)
                    slope[name] = s["PD"].mean(axis=0) - dark_c
                    for r in ("MonX1", "MonX2"):
                        if r in s:
                            mons.setdefault(r, []).append(s[r])
                    traces[f"slope_{which}_{name}"] = slope[name]
                mon_t = {r: np.vstack(v).mean(axis=0) for r, v in mons.items()}
                ref = w if which == "hold" else (max(t[0], idle[0]), idle[1])
                got[which] = track_azimuth(t, slope["plus"], slope["minus"], fit["imin"],
                                           mon_t, ref, sense=sense)
            tr = got["hold"]
            tr["t"] = t
            if "rest" in got:
                tr["lm_rest"] = got["rest"]["lm"]
                tr["dpsi_rest"] = got["rest"]["dpsi"]
                tr["valid_rest"] = got["rest"]["valid"]
        # monitors: hold-window mean, rest-referenced on the idle lead
        mv, phi = {}, 0.0
        for name, c in CHAN.items():
            r = c["mon"]
            if r in mons:
                stk = np.vstack(mons[r])
                hold = self._window(t, stk, w)[0]
                lead = self._window(t, stk, (max(t[0], idle[0]), idle[1]))[0] \
                    if (t < idle[1]).sum() > 5 else 0.0
                mv[r] = hold - lead
                phi += 90 * mv[r] / c["v90"]
        er = er_point(fit["imin"], fit["sig_imin"], imax, imax_s, fit["k"])
        pt = {"i": i, "bias": b, "x1": e1, "x2": e2, "dir": direction,
              "awg": ends_volts(e1, e2), "mon_V": mv, "phi_mon": phi if mv else None,
              "psi_coarse": psi, "imax4": imax4, "imin4": imin4, "null_predicted": predicted,
              "null_converged": ok,
              "theta_n": fit["theta_n"], "sig_theta_n": fit["sig_theta_n"],
              "psi": (fit["theta_n"] - 90) % 180, "fit": fit, "scan": scan,
              "imax": imax, "sig_imax": imax_s, **er}
        pt["sense"] = sense
        if tr is not None:
            pt["track"] = track_summary(tr["t"], tr["lm"], w, t_fall, tr.get("lm_rest"),
                                        tr.get("valid_rest"))
            pt["track"]["a0_hold_rel"] = float(np.mean(
                tr["a0_rel"][(tr["t"] >= w[0]) & (tr["t"] <= w[1])]))
        extra = {} if tr is None else {f"track_{k}": v for k, v in tr.items()
                                       if isinstance(v, np.ndarray)}
        np.savez_compressed(os.path.join(self.folder, f"point_{i:02d}.npz"),
                            t=t[::10], **{k: v[::10] for k, v in traces.items()}, **extra)
        e_ = (f"ER {er['er']:.0f}" if er["er"] else f"ER > {er['er_lower']:.0f}")
        tk_ = ""
        if tr is not None and pt["track"].get("hold_slope_mdeg_ms") is not None:
            tk_ = (f", creep {pt['track']['hold_slope_mdeg_ms']:+.1f} mdeg/ms"
                   + (f", tau {pt['track']['tau_ms']:.0f} ms" if "tau_ms" in pt["track"] else ""))
        self.log(f"  {what} deg ({direction}): null {fit['theta_n']:7.3f} "
                 f"+- {fit['sig_theta_n']*1e3:.0f} mdeg{' (predicted)' if predicted else ''}, "
                 f"Imin {fit['imin']*1e3:.3f} +- {fit['sig_imin']*1e3:.3f} mV at "
                 f"{setting[0]*1e3:g} mV/div, Imax {imax:.3f} V, {e_}, monitors "
                 f"{phi:+.2f} deg{tk_}")
        return pt

    def _save(self):
        tmp = os.path.join(self.folder, "bias.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=1, default=_jsonable)
        from .config import replace_retrying
        replace_retrying(tmp, os.path.join(self.folder, "bias.json"))


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def _key(setting):
    return f"{setting[0]:.6g}V/div@{setting[1]:+.6g}"


def _nice_up(x):
    """The next 1-2-5 step at or above x."""
    e = math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * 10 ** e >= x * (1 - 1e-9):
            return m * 10 ** e
    return 10 ** (e + 1)


def _ladder(imax, half_deg, coarse_vdiv):
    """Null V/div settings, finest first: from what the scan's edge needs
    (Imax sin^2(half + 1 deg) over ~5.5 div) up to just under the coarse V/div;
    1 mV/div is the MSO-X floor."""
    need = imax * math.sin(math.radians(half_deg + 1)) ** 2 / 5.5
    v = max(_nice_up(need), 1e-3)
    out = []
    while v < coarse_vdiv * 0.99 and len(out) < 8:
        out.append(v)
        v = _nice_up(v * 1.5)
    return out


def load(folder):
    """A finished (or stopped) bias run's manifest, with the transfer curve
    recomputed from its points."""
    path = folder if folder.endswith(".json") else os.path.join(folder, "bias.json")
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("format") != FORMAT:
        raise ValueError(f"{path} is not a bias-point run")
    man["transfer"] = transfer(man.get("points", []))
    man["folder"] = os.path.dirname(path)
    return man


# ------------------------------------------------------------ find an angle
def find_extremum(link, rot, roles, kind="min", window_s=None, bias_deg=None,
                  awg=None, plan=None, half_deg=None, points=None, log=print,
                  cancelled=None, ask=None, eomilc=None, ilc_bench=None,
                  session=None, progress=None):
    """The analyzer angle of minimum (crossed) or maximum transmission for the
    light as it is in a time window of the record, and the analyzer left
    there.

    window_s: (t0, t1) after the trigger in s - the rest before the ramp, a
    hold of the experiment's own sequence - or None for the whole record. With
    bias_deg (and an AWG) the EOMs are instead held at that rotation by a
    plateau, as in a bias run, and the window is the plateau's.

    4 analyzer angles at the PD's V/div give the azimuth; then
    min: +-half_deg (3) around crossed at the most sensitive V/div that keeps
         the scan on screen, I = Imin + K sin^2(theta - theta_n);
    max: +-half_deg (10) around the azimuth at the PD's V/div, the same shape
         upside down (the maximum is flat, so this one is less sharp).
    No dark is needed: the fit's floor is free. Returns dict(kind, angle,
    sig, level (raw V at the extremum), psi_coarse, theta, I, sem, vdiv,
    fit, window). progress(done, total, text) gets each acquisition with the
    time left at the pace so far."""
    import tempfile
    p = dict(PLAN, **(plan or {}))
    br = BiasRun(tempfile.gettempdir(), "find", link, rot, awg, roles, plan=p,
                 log=log, cancelled=cancelled, ask=ask, eomilc=eomilc,
                 ilc_bench=ilc_bench, session=session)
    br.pd_only = True
    half = float(half_deg if half_deg is not None else (p["null_half_deg"] if kind == "min" else 10.0))
    npts = int(points or p["null_points"])
    prog = {"t0": time.time(), "done": 0, "total": 4 + npts}

    def acquire(angle, setting, what):
        if progress is not None:
            d, n = prog["done"], max(prog["total"], prog["done"] + 1)
            progress(d, n, f"Find {kind}: {what} ({d + 1}/{n}{eta(prog['t0'], d, n)})")
        out = br._acquire(angle, setting)
        prog["done"] += 1
        return out
    pd = roles["PD"]
    sc = link.scope
    coarse = link.channel_state([pd])[pd]
    saved_tb, on = {}, False
    try:
        if bias_deg is not None:
            if awg is None:
                raise RuntimeError("holding a bias needs the AWG")
            check_plateaus([bias_deg], p, eomilc)
            t_rec, _ = plateau(0.0, p)
            period = float(t_rec[-1] + p["dt_us"] * 1e-6)
            window_s = windows(p)[0]
            saved_tb = {k: sc.get(k) for k in (":TIMebase:SCALe", ":TIMebase:POSition",
                                               ":TIMebase:REFerence")}
            div = _nice_up(period * 1.05 / 10)
            sc.put(":TIMebase:REFerence", "LEFT")
            sc.put(":TIMebase:SCALe", f"{div:.6g}")
            sc.put(":TIMebase:POSition", f"{-0.2e-3 + div:.6g}")
            on = True
            br._play(bias_deg)
            if not br.ask("Find angle", f"The AWG holds {bias_deg:g} deg. Switch both "
                          f"outputs ON now? (At the end: {p.get('end', 'off')}.)"):
                raise RuntimeError("outputs left off - nothing measured")
            br.sess.on()
            time.sleep(p["upload_settle_s"])
        w = window_s if window_s is not None else (-1e9, 1e9)
        th4 = (0.0, 45.0, 90.0, 135.0)
        I4 = []
        for a in th4:
            t, s = acquire(a, coarse, f"azimuth, analyzer {a:g} deg")
            I4.append(br._window(t, s["PD"], w)[0])
        psi, imax4, imin4 = malus4(th4, I4)
        amp = max(imax4 - imin4, 1e-4)
        log(f"Find {kind}: azimuth {psi:.2f} deg from 4 angles, Imax {imax4:.3f} V, "
            f"Imin {imin4*1e3:.1f} mV (raw, coarse) in "
            + ("the whole record" if window_s is None else
               f"{w[0]*1e3:.2f}..{w[1]*1e3:.2f} ms"))
        z = 0.0         # the floor kept on screen: offset = z + 3 V/div (see range_setting)
        if kind == "min":
            center = (psi + 90) % 180
            ladder = _ladder(amp, half, coarse[0])
            est = amp * math.sin(math.radians(half + 1)) ** 2
            sets = [range_setting(v, z) for v in ladder] + [coarse]
            k = next((i for i, s_ in enumerate(sets) if est < 5.5 * s_[0]), len(sets) - 1)
        else:
            center = psi % 180
            sets, k = [coarse], 0
        fit, recentred = None, 0
        while True:
            setting = sets[k]
            th = center + np.linspace(-half, half, npts)
            I, S, clipped = [], [], False
            for a in th:
                t, s = acquire(a % 180, setting, f"analyzer {a % 180:.2f} deg")
                side = br._clip_side(s["PD"], setting, t, w) if setting != coarse else None
                if side == "low" and kind == "min":
                    # below the screen: the floor (PD dark) is under 0 V -
                    # lower it and take the same V/div again
                    mwin = (t >= w[0]) & (t <= w[1])
                    z = float(np.min(s["PD"][:, mwin])) - 1.5 * setting[0]
                    sets = [range_setting(v, z) for v in ladder] + [coarse]
                    log(f"  below the screen at {setting[0]*1e3:g} mV/div - floor now "
                        f"{z*1e3:+.1f} mV")
                    clipped = True
                    prog["total"] += npts
                    break
                if side is not None:
                    clipped = True
                    break
                m_, se = br._window(t, s["PD"], w)
                I.append(m_)
                S.append(max(se, 1e-6))
            if clipped and side == "low":
                continue
            if clipped:
                prog["total"] += npts
                k = min(k + 1, len(sets) - 1)
                log(f"  off screen at {setting[0]*1e3:g} mV/div - now {sets[k][0]*1e3:g} mV/div")
                continue
            y = np.array(I) if kind == "min" else -np.array(I)
            fit = fit_null(th, y, S)
            if fit["k"] <= 0 and recentred < 2:
                # curving the wrong way: the sweep sits on the other extremum
                prog["total"] += npts
                center = (center + 90.0) % 180.0
                recentred += 1
                log(f"  the sweep curves the wrong way - the {kind} is 90 deg away; again")
                continue
            if (fit["inside"] and abs(fit["theta_n"] - center) < 0.6 * half) or recentred >= 2:
                break
            center = fit["theta_n"]
            recentred += 1
            prog["total"] += npts
            log(f"  extremum at {center:.2f} deg, off centre - again")
        angle = fit["theta_n"] % 180
        if progress is not None:
            progress(prog["done"], prog["done"], f"Find {kind} done")
        got = rot.approach(angle)
        level = fit["imin"] if kind == "min" else -fit["imin"]
        log(f"  {kind} transmission at analyzer {angle:.3f} +- {fit['sig_theta_n']*1e3:.0f} "
            f"mdeg ({level*1e3:.2f} mV raw at {setting[0]*1e3:g} mV/div); analyzer now "
            f"at {got:.3f}")
        return {"kind": kind, "angle": float(angle), "sig": fit["sig_theta_n"],
                "level": float(level), "level_raw": float(level),
                "psi_coarse": psi, "theta": th.tolist(),
                "I": [float(x) for x in I], "sem": S, "vdiv": setting[0], "fit": fit,
                "window": None if window_s is None else list(w), "bias": bias_deg,
                "setting": list(setting), "keys": [_key(coarse), _key(setting)]}
    finally:
        if on:
            br.sess.end(p.get("end", "off"))
            br.sess.forget()
        try:
            link.set_channel(pd, *coarse)
            for k_, v in saved_tb.items():
                if v is not None:
                    sc.put(k_, v)
        except Exception as exc:
            log(f"  could not restore the scope ({exc})")


# ------------------------------------------------- the light as it is (static)
class static_light:
    """The scope set for the light as it is, no ramp: trigger on the LINE
    (the experiment runs on, synchronous with the mains), 2 ms/div from the
    trigger - a 20 ms record - and the window one whole line period from
    just after the trigger, so the PD mean takes the 60 Hz out. Everything
    changed is put back on exit.

        with static_light(link, 60.0, log) as window:
            ... find_extremum(..., window_s=window) ...
    """

    KEYS = (":TRIGger:EDGE:SOURce", ":TRIGger:SWEep", ":TIMebase:SCALe",
            ":TIMebase:POSition", ":TIMebase:REFerence")

    def __init__(self, link, line_hz=60.0, log=print):
        self.link, self.line_hz, self.log = link, float(line_hz), log
        self.saved = {}

    def __enter__(self):
        sc = self.link.scope
        self.saved = {k: sc.get(k) for k in self.KEYS}
        sc.put(":TRIGger:EDGE:SOURce", "LINE")
        sc.put(":TRIGger:SWEep", "NORMal")
        sc.put(":TIMebase:REFerence", "LEFT")
        sc.put(":TIMebase:SCALe", "2.0E-03")
        # LEFT starts the record one division before the position
        sc.put(":TIMebase:POSition", "2.0E-03")
        period = 1.0 / self.line_hz
        self.log(f"  scope: LINE trigger, 2 ms/div from the trigger; PD mean over one "
                 f"{self.line_hz:g} Hz period ({period*1e3:.3f} ms)")
        return (0.05e-3, 0.05e-3 + period)

    def __exit__(self, *exc):
        sc = self.link.scope
        for k, v in self.saved.items():
            if v is not None:
                try:
                    sc.put(k, v)
                except Exception as e:
                    self.log(f"  could not restore {k} ({e})")
        self.log("  scope trigger and timebase put back")
        return False


def range_setting(vdiv, z=0.0):
    """(V/div, offset) with the floor z three divisions below the centre:
    the screen holds z - 1 div .. z + 7 div. The floor has to be on screen at
    every setting, or the dark / background (beam blocked: ~z) could not be
    measured there - and it is measured at the same setting because the
    scope's own offset error moves with V/div and offset (-34 mV at 1 V/div,
    2.65 V offset on 5 Oct 2026)."""
    return (float(vdiv), round(float(z) + 3.0 * float(vdiv), 6))


def vdiv_ladder(coarse_vdiv, finest=1e-3):
    """Every 1-2-5 V/div from `finest` up to just under the coarse one."""
    out, v = [], float(finest)
    while v < coarse_vdiv * 0.99 and len(out) < 14:
        out.append(v)
        v = _nice_up(v * 1.5)
    return out


def malus_fit(theta_deg, I, sem=None):
    """I = a0 + c2 cos 2theta + s2 sin 2theta, weighted by 1/sem^2 when
    given. Returns dict(psi (max), imax, imin, sig_imin, er, er_lower, a0,
    c2, s2): Imin = a0 - B with its error through the full covariance."""
    th = np.deg2rad(np.asarray(theta_deg, float))
    I = np.asarray(I, float)
    A = np.column_stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th)])
    w = np.ones_like(I)
    if sem is not None:
        se = np.maximum(np.asarray(sem, float), 1e-7)
        w = 1.0 / se
    coef, *_ = np.linalg.lstsq(A * w[:, None], I * w, rcond=None)
    a0, c2, s2 = coef
    B = math.hypot(c2, s2)
    res = (I - A @ coef) * w
    dof = max(len(I) - 3, 1)
    chi2 = float(res @ res / dof)
    cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None])) * max(chi2, 1.0)
    g = np.array([1.0, -c2 / B, -s2 / B]) if B > 0 else np.array([1.0, 0, 0])
    sig = float(np.sqrt(max(g @ cov @ g, 0.0)))
    imax, imin = a0 + B, a0 - B
    lower = imin < 2 * sig
    er = imax / (2 * sig) if lower else imax / imin
    return {"psi": math.degrees(0.5 * math.atan2(s2, c2)) % 180, "imax": float(imax),
            "imin": float(imin), "sig_imin": sig, "er": float(er), "er_lower": bool(lower),
            "a0": float(a0), "c2": float(c2), "s2": float(s2), "chi2": chi2}


def malus_scan(link, rot, roles, angles, window_s=None, plan=None, log=print,
               cancelled=None, progress=None, autorange=True):
    """The analyzer stepped over `angles` (deg), the PD mean in the window at
    each. With `autorange` every angle is read at the most sensitive V/div
    that holds it - predicted from the Malus fit of the points so far, one
    step coarser on a clip, re-read finer when a much finer setting would
    hold it - so the points near crossed are resolved instead of sitting at
    one ADC code of the coarse V/div. Offsets follow range_setting (the floor
    on screen), so a dark / background can be measured at each setting used
    afterwards (`keys`, apply_offsets). Returns the raw result; apply_offsets
    subtracts and fits. Leaves the analyzer at the last angle."""
    import tempfile
    p = dict(PLAN, **(plan or {}))
    br = BiasRun(tempfile.gettempdir(), "scan", link, rot, None, roles, plan=p, log=log,
                 cancelled=cancelled, session=object())
    br.pd_only = True
    pd = roles["PD"]
    coarse = link.channel_state([pd])[pd]
    w = window_s if window_s is not None else (-1e9, 1e9)
    vs = vdiv_ladder(coarse[0]) if autorange else []
    z = 0.0

    def settings():
        return [range_setting(v, z) for v in vs] + [coarse]      # finest first

    def finest_for(lo, hi):
        """The finest setting whose screen holds lo..hi with 0.5 div spare."""
        for st_ in settings():
            v, off = st_
            if st_ == coarse or (lo > off - 3.5 * v and hi < off + 3.5 * v):
                return st_
        return coarse

    th, I, S, keys, vd, rng = [], [], [], [], [], []
    t_run, n = time.time(), len(angles)
    spread = 0.0
    # every setting tried is the PD's; it goes back to the one it had
    # (7 Oct 2026: left at 5 mV/div, the next ramp scan clipped)
    try:
        for i, a in enumerate(angles):
            if progress is not None:
                progress(i, n, f"Malus scan: analyzer {a:.1f} deg ({i + 1}/{n}{eta(t_run, i, n)})")
            # the starting setting: predicted from the points so far
            if autorange and len(I) >= 3:
                f_ = malus_fit(th, I, S)
                r = math.radians(a)
                pred = f_["a0"] + f_["c2"] * math.cos(2 * r) + f_["s2"] * math.sin(2 * r)
                setting = finest_for(min(0.5 * pred, z) - spread, 1.6 * pred + spread)
            else:
                setting = coarse
            move = float(a) % 360.0
            for _try in range(6):
                t, s_ = br._acquire(move, setting)
                move = None                       # retries at the same angle
                mwin = (t >= w[0]) & (t <= w[1])
                x = s_["PD"][:, mwin]
                lo, hi = float(x.min()), float(x.max())
                side = br._clip_side(s_["PD"], setting, t, w) if setting != coarse else None
                if side == "low":
                    z = lo - 1.5 * setting[0]
                    setting = range_setting(setting[0], z)
                    continue
                if side == "high":
                    ss = settings()
                    setting = ss[min(ss.index(setting) + 1, len(ss) - 1)]
                    continue
                if autorange:
                    best = finest_for(lo, hi)
                    ss = settings()
                    if setting in ss and ss.index(best) <= ss.index(setting) - 2:
                        setting = best            # two or more steps finer: worth a re-read
                        continue
                break
            m_, se = br._window(t, s_["PD"], w)
            spread = max(spread * 0.5, hi - lo)
            th.append(float(a))
            I.append(m_)
            S.append(max(se, 1e-7))
            keys.append(_key(setting))
            vd.append(setting[0])
            rng.append(list(setting))
            log(f"  analyzer {a:7.2f}: PD {m_*1e3:10.3f} +- {se*1e3:.3f} mV raw at "
                f"{setting[0]*1e3:g} mV/div")
    finally:
        link.set_channel(pd, *coarse)
    if progress is not None:
        progress(n, n, "Malus scan done")
    out = {"kind": "scan", "theta": th, "I_raw": I, "sem": S, "keys": keys, "vdivs": vd,
           "settings": rng, "window": None if window_s is None else list(w),
           "vdiv": coarse[0], "offsets": {}, "offset_note": "nothing subtracted"}
    return apply_offsets(out, {}, log=log)


# ------------------------------------- dark / background for the analyzer scans
OFFSETS_FILE = "analyzer_offsets.json"


def apply_offsets(out, picked, log=None):
    """Subtract the dark / background measured at each reading's own
    setting (picked: {key: {level, sem, kind, when}}) and fit. Readings at a
    setting with nothing picked stay raw and are counted. Works for a Malus
    scan (kind 'scan') and a Find result (its level)."""
    if out["kind"] == "scan":
        I_raw = np.asarray(out["I_raw"], float)
        sub = np.array([picked[k]["level"] if k in picked else 0.0 for k in out["keys"]])
        ssub = np.array([picked[k].get("sem", 0.0) if k in picked else 0.0
                         for k in out["keys"]])
        I = I_raw - sub
        S = np.sqrt(np.asarray(out["sem"], float) ** 2 + ssub ** 2)
        fit = malus_fit(out["theta"], I, S)
        miss = sorted({k for k in out["keys"] if k not in picked})
        out.update(I=I.tolist(), sem_corr=S.tolist(), subtracted=sub.tolist(),
                   psi=fit["psi"], angle_max=fit["psi"], angle_min=(fit["psi"] + 90) % 180,
                   imax=fit["imax"], imin=fit["imin"], sig_imin=fit["sig_imin"],
                   er=fit["er"], er_lower=fit["er_lower"], fit=fit,
                   er_coarse=fit["er"], missing=miss)
        if log:
            log(f"Malus scan: maximum at analyzer {out['angle_max']:.2f} deg, minimum at "
                f"{out['angle_min']:.2f} deg; Imax {fit['imax']:.4f} V, Imin "
                f"{fit['imin']*1e3:.3f} +- {fit['sig_imin']*1e3:.3f} mV, ER "
                f"{'>' if fit['er_lower'] else ''}{fit['er']:.0f}"
                + (f" ({len(miss)} setting(s) without an offset - raw)" if miss and picked
                   else ""))
    else:
        k = out["keys"][-1]
        lv = picked.get(k)
        out["level"] = out["level_raw"] - (lv["level"] if lv else 0.0)
        out["missing"] = [] if lv else [k]
    out["offsets"] = picked
    kinds = sorted({v["kind"] for v in picked.values()})
    out["offset_note"] = (f"{' / '.join(kinds)} subtracted at {len(picked)} setting(s)"
                          if picked else "nothing subtracted")
    return out


def measure_offsets(link, rot, roles, keys_settings, window_s, plan, log=print,
                    cancelled=None, progress=None, label=""):
    """The PD's level with no light (dark: PD covered; background: beam
    blocked) at each setting in {key: (V/div, offset)}, in the window, as
    the readings were taken. Returns {key: {level, sem, n}}."""
    import tempfile
    p = dict(PLAN, **(plan or {}))
    br = BiasRun(tempfile.gettempdir(), "offs", link, rot, None, roles, plan=p, log=log,
                 cancelled=cancelled, session=object())
    br.pd_only = True
    w = window_s if window_s is not None else (-1e9, 1e9)
    out, t_run, n = {}, time.time(), len(keys_settings)
    pd = roles["PD"]
    saved = link.channel_state([pd])[pd]
    try:
        for i, (key, setting) in enumerate(sorted(keys_settings.items())):
            if progress is not None:
                progress(i, n, f"{label} at {setting[0]*1e3:g} mV/div ({i + 1}/{n}"
                               f"{eta(t_run, i, n)})")
            t, s_ = br._acquire(None, tuple(setting))
            m_, se = br._window(t, s_["PD"], w)
            out[key] = {"level": m_, "sem": se, "n": int(s_["PD"].shape[0])}
            log(f"  {label} at {setting[0]*1e3:g} mV/div, offset {setting[1]*1e3:+.1f} mV: "
                f"{m_*1e3:+.3f} +- {se*1e3:.3f} mV")
    finally:
        # the PD back where it was (it was left at the last setting read, and
        # the next ramp scan clipped - 7 Oct 2026)
        link.set_channel(pd, *saved)
    if progress is not None:
        progress(n, n, f"{label} done")
    return out


def save_offsets(outdir, kind, mode, levels, settings):
    """Append a measurement to <outdir>/analyzer_offsets.json (newest last)."""
    import datetime
    from .config import replace_retrying
    path = os.path.join(outdir, OFFSETS_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            db = json.load(fh)
    except (OSError, ValueError):
        db = {"records": []}
    db["records"].append({"kind": kind, "mode": mode,
                          "when": datetime.datetime.now().isoformat(timespec="seconds"),
                          "levels": levels,
                          "settings": {k: list(v) for k, v in settings.items()}})
    os.makedirs(outdir, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=1)
    replace_retrying(tmp, path)
    return path


def pick_offsets(outdir, settings, mode, measured=None, kinds=("background", "dark")):
    """{key: {level, sem, kind, when}} for each setting {key: (V/div, offset)}:
    `measured` (this run's, {kind: {key: ...}}) first, else the newest stored
    measurement in the same mode at the same V/div with an offset within
    max(2 div, 5 %) (the scope's offset error moves ~1.3 % of the offset).
    Background before dark (it includes the stray light)."""
    measured = measured or {}
    try:
        with open(os.path.join(outdir, OFFSETS_FILE), encoding="utf-8") as fh:
            recs = json.load(fh).get("records", [])
    except (OSError, ValueError):
        recs = []
    out = {}
    for key, (v, off) in settings.items():
        for kind in [k for k in ("background", "dark") if k in kinds]:
            got = (measured.get(kind) or {}).get(key)
            if got:
                out[key] = dict(got, kind=kind, when="this run")
                break
            best = None
            for r in reversed(recs):
                if r.get("kind") != kind or r.get("mode") != mode:
                    continue
                for k2, (v2, off2) in r.get("settings", {}).items():
                    if abs(v2 / v - 1) < 1e-6 and abs(off2 - off) <= max(2 * v, 0.05 * abs(off)):
                        best = dict(r["levels"][k2], kind=kind, when=r["when"])
                        break
                if best:
                    break
            if best:
                out[key] = best
                break
    return out

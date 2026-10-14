"""Measured polarization against the ILC's original target, and the target
correction that would close the gap (the EOM-ILC panel's Corrections tab
reads what this writes).

The target is NOT the rotation the Trek monitors predict: it is the target the
ILC was given (monitor volts, one per crystal), which the converged drives
were built to follow. With the QWP making the EOM pair a rotator, each crystal
turns the polarization by 90 deg x V / V90:
    phi_target = 90 (T1/V90_1 + T2/V90_2)      [deg]
with V90 the monitor voltage for 90 deg measured optically (5.1283 / 5.1374 V).

TIMING. The experiment does not play the ILC record as it is: on 5 Oct 2026
the recorded command was the P92PX1H drive compressed in time (rise 4.61 ->
4.16 ms, hold 1.24 -> 0.68 ms, fall 4.61 -> 4.17 ms; 11 mV rms fit to 9.2 V).
So the played waveform is located with a piecewise-linear time map - rise
start, rise, hold and fall durations, plus command gain and offset - fitted
to the recorded command channel (CmdX1/CmdX2) against that crystal's ILC drive
u; without a command channel the monitors are fitted against the target
instead. The second leg takes the same durations and its own start. The
target is mapped through the same warp: that is the polarization the
experiment's waveform was meant to produce.

SEVERAL TRANSPORTS IN ONE RECORD (13 Oct 2026). The polarization drifts in a
hold at high voltage and the drift carries into the next transport, so the
ILC may train a record holding two or more transports at the spin echo's
spacing (transport 1, idle, transport 2 16.667 ms later), each getting its
own learned correction. `transports` splits the record's target into its
transports (stretches above 0.1 % of the maximum separated by idle, each
with its own 99.9 % hold); `landmarks` is their envelope, which for a
one-transport record is that transport, so nothing changes there. The time
map's knots are ALL transports' landmarks: `fit_time_map` fits each
transport on its own (fit_timing: start and its three durations, in a
window that stops halfway through the idle on either side) and the warp runs
through the concatenated knots, idle stretched linearly between them. A
second leg (the whole record played again leg_gap_ms later, when that is
past the record's last fall) keeps every duration and spacing and fits its
own start.
    per transport   its landmarks, and from the whole record (band-limited at
                    f_cut, the EOM-ILC panel's definitions): the signed
                    extreme of light - monitors and light - target over its
                    hold +- 0.3 ms and over the first ms after its fall, and
                    their rms over its active stretch plus that ms
                    (summary["transports"] for leg 1; summary["legs"][k]
                    ["transports"] for each leg; a one-transport record gives a
                    one-element list)
    whole record    the time map's gain, offset and rms, the per-leg gain /
                    offset / delay decomposition, the correction (one file per
                    crystal over the whole record, faded to zero only where
                    the target rests, i.e. at every transport's ends and in the
                    idle between them) and its per-crystal numbers, the line
                    ripple (fitted on the undriven stretches: before the first
                    transport, between the transports and between the legs).
summary["timing"][k]["landmarks_ms"] and summary["source_landmarks_ms"]
stay one array each (the envelope); "transports_ms" and
"source_transports_ms" hold each transport's.

On each leg:
    light - target      the total error
    monitors - target   the drive's own error as the Trek monitors see it:
                        the ILC removes this part by itself, once it runs on
                        the played target
    light - monitors    what the monitors cannot see: THIS is the correction
                        the target needs (minus it, low-passed)
Using light - target instead (what the first version of this tool wrote on
5 Oct as *_optcorr.csv) counts the monitor error twice once the ILC is run on
the corrected target: the loop removes the monitor error, and the target
change removes it again, so it comes back with the opposite sign.

LINE RIPPLE. The experiment's trigger is line-synchronous, so 60 Hz and its
harmonics sit at a fixed phase in every record and survive the averaging
(1 Oct dedicated captures: 0.87 / 0.92 mV on the monitors, 27 mdeg on the
light). A correction formed from such a record carries an anti-ripple tied
to this sequence's phase. The ramp record itself cannot measure it: on
test-4 the undriven stretches (12 ms before leg 1, 6.7 ms between the legs,
with the Trek settling and the crystal memory in them) gave 60 Hz estimates
that moved 2-5x, and their phase by 100+ deg, with 0.3 ms changes of the
windows. So: give `line_ref`, a scan of the SAME sequence with the ramps
disabled; light - monitors is fitted there over the whole record (a single
offset and slope, harmonics of line_f) and subtracted before the correction
is formed. Without it the correction keeps the ripple and its file says so
(line_removed false; the ILC panel then warns).

OUTPUTS (SCAN/analysis/target_compare/, nothing else is touched):
    target_<name>_played.csv    the ILC target with the experiment's timing, on
                                the ILC's 2 us grid and CSV format: what the
                                ILC should converge to for this sequence
    corr_<name>_optical.csv     the target correction for that crystal
                                (eomilc.corrections format: delta and sigma,
                                V at the EOM, with its provenance), for the
                                ILC panel's Corrections tab on a campaign
                                whose target is the played one
    summary.json, target_compare.npz, fig1-5
The leg-to-leg difference cannot be removed by one waveform played on both
legs; it is reported separately.
"""
import datetime
import json
import os

import numpy as np

from . import analysis as an

V90 = {"EO1": 5.1283, "EO2": 5.1374}   # monitor volts for 90 deg (V_pi, 31 Aug 2026)


def load_ilc(path):
    z = np.load(path, allow_pickle=True)
    return {"t": np.asarray(z["t"], float), "target": np.asarray(z["target"], float),
            "u": np.asarray(z["u"], float), "name": str(z["name"]),
            "channel": str(z["channel"]), "path": os.path.abspath(path)}


def transports(tt, T, min_gap=0.1e-3, min_peak=0.05):
    """The transports in a target record: one landmark array per transport,
    [rise start, hold start, hold end, fall end]. A transport is a stretch
    where T > 0.1 % of the record's maximum, separated from the next by at
    least min_gap of idle; its hold is where it is above 99.9 % of its OWN
    maximum. A stretch peaking under min_peak of the record's maximum (an
    ILC ripple after a fall, say) is not a transport: it joins the one before
    it (the first one, when it comes before them all). In record order."""
    T = np.asarray(T, float)
    idx = np.flatnonzero(T > 1e-3 * T.max())
    if not idx.size:
        raise ValueError("the target never leaves zero: no transport in it")
    br = np.flatnonzero(np.diff(idx) > 1)
    runs = [[a, b] for a, b in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]])]
    merged = [runs[0]]
    for a, b in runs[1:]:
        if tt[a] - tt[merged[-1][1]] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    big = [r for r in merged if T[r[0]:r[1] + 1].max() >= min_peak * T.max()]
    for r in merged:
        if r in big:
            continue
        host = max((b for b in big if b[0] < r[0]), key=lambda b: b[0], default=big[0])
        host[0], host[1] = min(host[0], r[0]), max(host[1], r[1])
    out = []
    for a, b in big:
        seg = T[a:b + 1]
        hold = np.flatnonzero(seg > 0.999 * seg.max()) + a
        out.append(np.array([tt[a], tt[hold[0]], tt[hold[-1]], tt[b]]))
    return out


def envelope(tr):
    """One landmark array over several transports: the first's rise start and
    hold start, the last's hold end and fall end."""
    return np.array([tr[0][0], tr[0][1], tr[-1][2], tr[-1][3]])


def landmarks(tt, T):
    """Rise start, hold start, hold end, fall end of a target (0.1 % / 99.9 %).
    For a record with one transport this is that transport (transports()[0]);
    with several, their envelope (first rise and hold start, last hold end
    and fall end) - transports() has each one."""
    return envelope(transports(tt, T))


def warp(S, E, tx):
    """Experiment time -> ILC-record time through landmark pairs E -> S. S and
    E: one transport's 4 landmarks, or the knots of several (a list of them,
    or concatenated), in time order."""
    S, E = np.ravel(np.asarray(S, float)), np.ravel(np.asarray(E, float))
    return np.interp(tx, np.r_[E[0] - 1, E, E[-1] + 1], np.r_[S[0] - 1, S, S[-1] + 1])


def _gain_offset(t, y, src_t, src_y, S, E, m):
    """Command gain and offset of src_y mapped through S -> E onto y over the
    mask m, by linear least squares; and the rms left."""
    x = np.interp(warp(S, E, t[m]), src_t, src_y)
    A = np.column_stack([x, np.ones_like(x)])
    (g, o), *_ = np.linalg.lstsq(A, y[m], rcond=None)
    return float(g), float(o), float(np.std(A @ [g, o] - y[m]))


def fit_time_map(t, y, src_t, src_y, S_tr, start_guess, window=12e-3, log=None):
    """The time map of a record with one or more transports (S_tr, from
    transports()). Each transport is fitted on its own (fit_timing: start and
    its three durations) inside a window that stops halfway through the idle
    on either side; the first starts from start_guess, each next one from the
    previous one's fitted start plus the record's start-to-start spacing,
    refined by a coarse search over +-1 ms. Returns (E_tr, gain, offset, rms):
    one landmark array per transport, and the command gain, offset and rms of
    the whole map (for one transport exactly fit_timing's)."""
    if len(S_tr) == 1:
        E, g, o, r = fit_timing(t, y, src_t, src_y, S_tr[0], start_guess, window=window)
        return [E], g, o, r
    E_tr, masks = [], []
    for j, S in enumerate(S_tr):
        if j == 0:
            guess, lo = start_guess, -np.inf
        else:
            Ep, Sp = E_tr[-1], S_tr[j - 1]
            guess = Ep[0] + (S[0] - Sp[0])
            lo = Ep[3] + 0.5 * (S[0] - Sp[3])
        hi = (guess + (S[3] - S[0]) + 0.5 * (S_tr[j + 1][0] - S[3])
              if j + 1 < len(S_tr) else np.inf)
        if j > 0:
            # the spacing in the experiment need not be the record's
            best = None
            for sh in np.arange(-1e-3, 1e-3 + 1e-9, 10e-6):
                m = (t > max(lo, guess + sh - 1.5e-3)) & (t < min(hi, guess + sh + window))
                if m.sum() < 20:
                    continue
                r = _gain_offset(t, y, src_t, src_y, S, S - S[0] + guess + sh, m)[2]
                if best is None or r < best[0]:
                    best = (r, sh)
            if best is not None:
                guess += best[1]
        E, g, o, r = fit_timing(t, y, src_t, src_y, S, guess, window=window, span=(lo, hi))
        E_tr.append(E)
        masks.append((t > max(lo, guess - 1.5e-3)) & (t < min(hi, guess + window)))
        if log:
            log(f"  transport {j + 1}: rise {np.diff(E)[0]*1e3:.3f} / hold "
                f"{np.diff(E)[1]*1e3:.3f} / fall {np.diff(E)[2]*1e3:.3f} ms from "
                f"{E[0]*1e3:.3f} ms, gain {g:.4f}, {r*1e3:.1f} mV rms")
    m = np.any(masks, axis=0)
    g, o, r = _gain_offset(t, y, src_t, src_y, np.concatenate(S_tr), np.concatenate(E_tr), m)
    return E_tr, g, o, r


def fit_timing(t, y, src_t, src_y, S, start_guess, durations=None, window=12e-3, span=None):
    """Fit the time map (and gain, offset) taking src_y on its record onto y,
    the recorded trace. Free: start, and unless given the rise, hold and fall
    durations. S: one transport's 4 landmarks; with `durations` given it may
    hold the knots of several (len(S) - 1 durations: only the start is free).
    span (lo, hi): the fit window is also kept inside it (the halfway points
    of the idle to the neighbouring transports). Returns (E landmarks, gain,
    offset, rms)."""
    from scipy.optimize import least_squares
    S = np.ravel(np.asarray(S, float))
    m = (t > start_guess - 1.5e-3) & (t < start_guess + window)
    if span is not None:
        m &= (t > span[0]) & (t < span[1])
    S_d = np.diff(S)

    def unpack(p):
        d = np.abs(p[1:len(S)]) if durations is None else durations
        return np.cumsum(np.r_[p[0], d]), p[-2], p[-1]

    def resid(p):
        E, g, o = unpack(p)
        return g * np.interp(warp(S, E, t[m]), src_t, src_y) + o - y[m]
    p0 = ([start_guess, *S_d] if durations is None else [start_guess]) + [1.0, 0.0]
    scale = ([1e-4] * (len(S) if durations is None else 1)) + [0.01, 0.01]
    r = least_squares(resid, p0, x_scale=scale)
    E, g, o = unpack(r.x)
    return E, float(g), float(o), float(np.std(r.fun))


def decompose(delta, phi, dphi, mask):
    """delta ~ gain*phi + offset + delay*(-dphi/dt) over mask. A positive
    delay means the measured curve lags the target."""
    A = np.column_stack([phi, np.ones_like(phi), -dphi])[mask]
    coef, *_ = np.linalg.lstsq(A, delta[mask], rcond=None)
    left = delta[mask] - A @ coef
    return {"gain_pct": float(coef[0] * 100), "offset_mdeg": float(coef[1] * 1e3),
            "delay_us": float(coef[2] * 1e6), "rms_mdeg": float(np.std(delta[mask]) * 1e3),
            "rms_left_mdeg": float(np.sqrt(np.mean(left ** 2)) * 1e3)}


def write_target_csv(path, t, hv, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(f"# {ln}\n")
        fh.write("time_us,voltage_V\n")
        for a, b in zip(t, hv):
            fh.write(f"{a * 1e6:.6f},{b:.6f}\n")


def undriven(t, legs_E, pre_pad=0.3e-3, post_pad=0.6e-3):
    """The stretches with no ramp: before leg 1, between the transports of a
    leg and between the legs (after the last leg the lock may be off, so not
    used). legs_E: per leg its landmarks - 4 per transport, as one array of
    knots or a list of 4-arrays. Bool masks."""
    tr = [np.asarray(x, float) for E in legs_E for x in np.reshape(np.ravel(
        np.asarray(E, float)), (-1, 4))]
    segs = [(t > t[0] + 0.2e-3) & (t < tr[0][0] - pre_pad)]
    for a, b in zip(tr[:-1], tr[1:]):
        segs.append((t > a[-1] + post_pad) & (t < b[0] - pre_pad))
    return [s for s in segs if s.sum() > 20]


TOP_PAD = 0.3e-3      # the hold window of a transport reaches this far past its hold
AFTER_FALL = 1.0e-3   # the after-fall window: the first ms after a transport's fall


def _extreme(t, x, lo, hi):
    """The signed value of x where |x| is largest over lo <= t <= hi (mdeg)."""
    m = (t >= lo) & (t <= hi)
    if not m.any():
        return float("nan")
    return float(x[np.flatnonzero(m)[int(np.argmax(np.abs(x[m])))]] * 1e3)


def transport_numbers(t, lm_f, lt_f, phi_t, E_tr, next_start=np.inf):
    """Per transport of one leg: the signed extremes of light - monitors and
    light - target (lm_f, lt_f, deg, band-limited, on the whole record t)
    over its hold +- TOP_PAD and over the first ms after its fall (stopped at
    the next transport's rise), and their rms over its active stretch
    (target phi_t > 0.5 deg) plus that after-fall window. E_tr: the leg's
    landmark arrays; next_start: where the next leg starts. List of dicts."""
    out = []
    for j, E in enumerate(E_tr):
        nxt = E_tr[j + 1][0] if j + 1 < len(E_tr) else next_start
        aft = (E[3], min(E[3] + AFTER_FALL, nxt, t[-1]))
        act = (((t >= E[0]) & (t <= E[3]) & (phi_t > 0.5))
               | ((t > aft[0]) & (t <= aft[1])))
        rms = (lambda x: float(np.sqrt(np.mean(x[act] ** 2)) * 1e3) if act.any()
               else float("nan"))
        out.append({"index": j + 1, "landmarks_ms": (np.asarray(E) * 1e3).tolist(),
                    "top_lm_mdeg": _extreme(t, lm_f, E[1] - TOP_PAD, E[2] + TOP_PAD),
                    "top_lt_mdeg": _extreme(t, lt_f, E[1] - TOP_PAD, E[2] + TOP_PAD),
                    "after_fall_lm_mdeg": _extreme(t, lm_f, *aft),
                    "after_fall_lt_mdeg": _extreme(t, lt_f, *aft),
                    "lt_rms_mdeg": rms(lt_f), "lm_rms_mdeg": rms(lm_f)})
    return out


def light_minus_monitors(d, pol, sign, v90, pd_delay_us=0.0):
    """(t, rot, pm, lm, mon): the measured rotation (sign-flipped to the
    target's sense, PD delay taken out), the rotation the monitors predict,
    their difference (deg), and the rest-referenced monitors (V)."""
    t = d.t
    steps = pol["steps"]

    def mean(r):
        return np.mean([s["v"][r] for s in steps if r in s["v"]], axis=0)
    for r in ("MonX1", "MonX2"):
        if r not in d.roles:
            raise ValueError(f"{d.name} has no {r} channel - light minus monitors "
                             f"needs both monitors")
    rest = an.rest_index(t)
    mon = {k: mean(f"MonX{k}") - np.median(mean(f"MonX{k}")[rest]) for k in (1, 2)}
    tau = float(pd_delay_us) * 1e-6
    rot = sign * (np.interp(t + tau, t, pol["rotation"]) if tau else pol["rotation"])
    pm = 90 * (mon[1] / v90[1] + mon[2] / v90[2])
    return t, rot, pm, rot - pm, mon


def line_reference(path, sign, v90, line_f, line_h, t_end, load_capture,
                   lock_tol=0.006, pd_delay_us=0.0):
    """The line ripple of light - monitors (deg) from a scan with the ramps
    disabled: one offset and slope, harmonics of line_f, fitted from the
    record start to t_end. Returns (fits, the scan's name); light - monitors
    and the light in mdeg, the monitors in mV."""
    from eomilc import mains
    d = an.load_scan(path, load_capture, lock_tol=lock_tol)
    pol = an.polarization(d)
    t, rot, pm, lm, mon = light_minus_monitors(d, pol, sign, v90, pd_delay_us)
    if np.max(np.abs(pm)) > 1.0:
        raise ValueError(f"{d.name} is not undriven: the monitors predict "
                         f"{np.max(np.abs(pm)):.1f} deg of rotation")
    m = (t > t[0] + 0.2e-3) & (t < t_end)
    if (t[m][-1] - t[m][0]) * line_f < 1.5:
        raise ValueError(f"{d.name}: {1e3*(t[m][-1]-t[m][0]):.1f} ms of record "
                         f"is under 1.5 periods of {line_f:g} Hz")
    fits = {"light_minus_monitors_mdeg": mains.fit_line(t, lm * 1e3, line_f, line_h, [m]),
            "light_mdeg": mains.fit_line(t, rot * 1e3, line_f, line_h, [m]),
            "MonX1_mV": mains.fit_line(t, mon[1] * 1e3, line_f, line_h, [m]),
            "MonX2_mV": mains.fit_line(t, mon[2] * 1e3, line_f, line_h, [m])}
    return fits, d.name


def compare(scan, x1, x2, f_cut=2000.0, lock_tol=0.006, leg_gap_ms=16.667,
            pd_delay_us=0.0, line_f=60.0, line_h=3, split=0.5, line_ref=None,
            scope_grab_path=None, eomilc_path=None, log=print, figs=True,
            whole_record=False):
    """Everything the CLI does; returns the summary dict (and writes the
    outputs). x1/x2: ILC state files (drive_*.state.npz) for EO1 and EO2.
    line_ref: a scan of the same sequence with the ramps off (see LINE
    RIPPLE above), or None."""
    from . import config, hw
    sg = hw.load_scope_grab(scope_grab_path or config.DEFAULTS["scope_grab_path"])
    hw.load_eomilc(eomilc_path or config.DEFAULTS["eomilc_path"])
    from eomilc import corrections as corrmod, mains

    d = an.load_scan(scan, sg.load_capture, lock_tol=lock_tol)
    pol = an.polarization(d)
    log(f"{d.name}: {pol['n_angles']} analyzer angles")
    x = {1: load_ilc(x1), 2: load_ilc(x2)}
    tt = x[1]["t"]
    for k in (1, 2):
        if not np.allclose(x[k]["t"], tt):
            x[k]["target"] = np.interp(tt, x[k]["t"], x[k]["target"])
            x[k]["u"] = np.interp(tt, x[k]["t"], x[k]["u"])
    v90 = {k: V90.get(x[k]["channel"], (5.1283, 5.1374)[k - 1]) for k in (1, 2)}
    # SOLO scans (13 Oct 2026): one crystal drives while the other is parked
    # at its idle, so the light's departure from the monitors is that
    # crystal's alone and the whole correction is attributed to it. A parked
    # state carries a flat target; everything that reads "the" target (the
    # transports, the timing reference, the active stretch, the fade) reads
    # the first ACTIVE crystal's, and `split` is overridden to send the
    # correction to it. Both active: as before (X1 leads).
    driven = [k for k in (1, 2) if np.ptp(x[k]["target"]) > 1e-9]
    if not driven:
        raise ValueError("both states carry a flat target: nothing was driven")
    lead = driven[0]
    solo = None
    if len(driven) == 1:
        solo = f"X{lead}"
        split = 1.0 if lead == 1 else 0.0
        log(f"{solo} alone: X{3 - lead}'s target is flat (parked), so the whole "
            f"correction goes to {solo} (split {split:g})")
    # the record's transports; S is their envelope (one transport: itself),
    # Sk all their landmarks: the knots of the time map
    S_tr = transports(tt, x[lead]["target"])
    S, Sk = envelope(S_tr), np.concatenate(S_tr)
    if len(S_tr) > 1:
        log(f"the ILC record holds {len(S_tr)} transports, starting at "
            + ", ".join(f"{s[0]*1e3:.3f}" for s in S_tr) + " ms")
    t = d.t
    dt_e = float(np.median(np.diff(t)))
    steps = pol["steps"]

    def mean(r):
        return np.mean([s["v"][r] for s in steps if r in s["v"]], axis=0)
    sign0 = 1.0
    _, _, _, _, mon = light_minus_monitors(d, pol, sign0, v90)

    # the reference for timing: an ACTIVE crystal's recorded command against
    # its drive, else the leading active crystal's monitor against its target
    ref_k = next((k for k in driven if f"CmdX{k}" in d.roles), None)
    if ref_k is not None:
        ref_y, src_y, ref_name = mean(f"CmdX{ref_k}"), x[ref_k]["u"], f"CmdX{ref_k} vs {x[ref_k]['name']} drive"
    else:
        ref_k = lead
        ref_y, src_y, ref_name = mon[lead], x[lead]["target"], f"MonX{lead} vs {x[lead]['name']} target"
    act = np.flatnonzero(mon[lead] > 0.01 * mon[lead].max())
    # leg 1: every transport on its own; legs_E holds each leg's knots (4 per
    # transport), legs_tr its landmark arrays, E1 leg 1's envelope
    E1_tr, gain, off, rms = fit_time_map(t, ref_y, tt, src_y, S_tr, t[act[0]], log=log)
    E1k, E1 = np.concatenate(E1_tr), envelope(E1_tr)
    timing = [{"leg": 1, "landmarks_ms": (E1 * 1e3).tolist(),
               "transports_ms": [(E * 1e3).tolist() for E in E1_tr],
               "gain": gain, "offset_V": off, "rms_mV": rms * 1e3}]
    legs_E, legs_tr = [E1k], [E1_tr]
    # leg 2: the whole record again, same durations and spacings, its own start
    g2 = E1k[0] + leg_gap_ms * 1e-3
    if g2 + (E1k[-1] - E1k[0]) < t[-1] and g2 > E1k[-1]:
        win = 12e-3 if len(S_tr) == 1 else max(12e-3, E1k[-1] - E1k[0] + 1.5e-3)
        E2k, g_, o_, r_ = fit_timing(t, ref_y, tt, src_y, Sk, g2, durations=np.diff(E1k),
                                     window=win)
        E2_tr = list(np.reshape(E2k, (-1, 4)))
        legs_E.append(E2k)
        legs_tr.append(E2_tr)
        timing.append({"leg": 2, "landmarks_ms": (envelope(E2_tr) * 1e3).tolist(),
                       "transports_ms": [(E * 1e3).tolist() for E in E2_tr], "gain": g_,
                       "offset_V": o_, "rms_mV": r_ * 1e3,
                       "start_after_leg1_ms": (E2k[0] - E1k[0]) * 1e3})
    elif g2 + (E1k[-1] - E1k[0]) < t[-1]:
        log(f"no leg 2: {leg_gap_ms:g} ms after leg 1 is inside leg 1's record "
            f"(its last fall ends {(E1k[-1] - E1k[0])*1e3:.3f} ms after its start)")
    if len(E1_tr) == 1:
        log(f"time map ({ref_name}): rise {np.diff(E1)[0]*1e3:.3f} / hold "
            f"{np.diff(E1)[1]*1e3:.3f} / fall {np.diff(E1)[2]*1e3:.3f} ms, "
            f"{rms*1e3:.1f} mV rms")
    else:
        log(f"time map ({ref_name}): {len(E1_tr)} transports (above), gain {gain:.4f}, "
            f"{rms*1e3:.1f} mV rms over them all")

    sign = -1.0 if np.median(pol["rotation"][(t > E1_tr[0][1]) & (t < E1_tr[0][2])]) < 0 else 1.0
    # the PD's own delay (an anti-alias RC, the detector) is MEASUREMENT, not
    # light: read the light that far later so it lines up with the monitors
    t, rot, pm, lm, mon = light_minus_monitors(d, pol, sign, v90, pd_delay_us)
    sig = pol["sig_psi"]

    # line ripple: from a drive-off scan of the same sequence when there is
    # one; the in-record estimate is drawn and reported, never subtracted
    segs = undriven(t, legs_E)
    line, line_src = {}, None
    if line_ref:
        t_end = legs_E[-1][-1]
        line, ref_name = line_reference(line_ref, sign, v90, line_f, line_h, t_end,
                                        sg.load_capture, lock_tol, pd_delay_us)
        line_src = f"drive-off scan {ref_name}, fitted to {t_end*1e3:.1f} ms"
    else:
        for key, y in (("light_minus_monitors_mdeg", lm), ("light_mdeg", rot),
                       ("MonX1_mV", mon[1]), ("MonX2_mV", mon[2])):
            try:
                line[key] = mains.fit_line(t, y * 1e3, line_f, line_h, segs)
            except ValueError:
                line[key] = None
    lfit = line.get("light_minus_monitors_mdeg")
    if line_src and lfit is not None:
        lm_line = mains.line_wave(t, lfit) * 1e-3
        log(f"line ripple ({line_src}): light - monitors "
            + mains.describe(lfit, unit="mdeg") + " -- subtracted")
    else:
        lm_line = np.zeros_like(t)
        if lfit is not None:
            log("line ripple NOT removed (no drive-off reference scan). In-record "
                "estimate, light - monitors, NOT reliable (short undriven "
                "stretches): " + mains.describe(lfit, unit="mdeg"))
    for k in (1, 2):
        f_ = line.get(f"MonX{k}_mV")
        if f_ is not None:
            log(f"  MonX{k}: " + mains.describe(f_, unit="mV")
                + ("" if line_src else " (in-record, unreliable)"))
    lm_clean = lm - lm_line

    # each leg on its own time window, all compared on leg 1's time axis
    pad = 0.3e-3
    if whole_record:
        # the ILC's use (14 Oct 2026): the light after the fall is part of
        # what the loop must correct (the crystal's memory keeps the light
        # rotated for tens of ms after a transport), so the comparison and
        # the correction run over the whole played record, lead to tail,
        # not the transports' window; the record itself is built long enough
        # to hold the time of interest (a spin echo's full 2 x 16.667 ms)
        tl = np.arange(E1[0] - (S[0] - tt[0]), E1[0] + (tt[-1] - S[0]), dt_e)
    else:
        tl = np.arange(E1[0] - pad, E1[-1] + pad, dt_e)
    legs = []
    for k, E in enumerate(legs_E, 1):
        te = tl + (E[0] - E1[0])
        src = warp(Sk, E, te)
        T = {c: np.interp(src, tt, x[c]["target"], left=0, right=0) for c in (1, 2)}
        phi_t = 90 * (T[1] / v90[1] + T[2] / v90[2])
        phi_l = np.interp(te, t, rot)
        phi_m = np.interp(te, t, pm)
        legs.append({"leg": k, "phi_target": phi_t, "phi_light": phi_l, "phi_mon": phi_m,
                     "sig": np.interp(te, t, sig), "d_light": phi_l - phi_t,
                     "d_mon": phi_m - phi_t, "d_lm": phi_l - phi_m,
                     "d_lm_clean": np.interp(te, t, lm_clean),
                     "line": np.interp(te, t, lm_line)})
    phi_t = legs[0]["phi_target"]
    dphi_t = np.gradient(phi_t, tl)
    active = phi_t > 0.5
    for L in legs:
        for key in ("d_light", "d_mon", "d_lm"):
            L[key + "_fit"] = decompose(L[key], phi_t, dphi_t, active)

    # the correction: minus what the monitors cannot see, mean of the legs,
    # line ripple out, low-passed, on leg-1 time; its standard error
    n_l = len(legs)
    d_mean = np.mean([L["d_lm_clean"] for L in legs], axis=0)
    sig_mean = np.sqrt(np.sum([L["sig"] ** 2 for L in legs], axis=0)) / n_l
    # whole_record: everywhere the light was measured, the idle after a
    # transport included (the memory the monitors cannot see); else only
    # where the target is active
    corr = -corrmod.lowpass(d_mean if whole_record else np.where(active, d_mean, 0.0), dt_e, f_cut)
    n_t = int(0.2e-3 / dt_e)
    taper = np.ones_like(corr)
    taper[:n_t] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_t))
    taper[-n_t:] = taper[:n_t][::-1]
    corr *= taper
    # ILC grid: same 2 us spacing and length, the played rise starting where the
    # original's did, so the record's padding is unchanged
    tg = tt.copy()
    dt_g = float(np.median(np.diff(tg)))
    te_g = tg - S[0] + E1[0]
    src_g = warp(Sk, E1k, te_g)
    corr_g = np.interp(te_g, tl, corr, left=0, right=0)
    # per-sample error on the coarser ILC grid: the polarimeter's samples
    # merge, so it falls as sqrt(dt_e / dt_g)
    sig_g = np.interp(te_g, tl, sig_mean, left=0, right=0) * np.sqrt(min(dt_e / dt_g, 1.0))
    # the rotation is measured from rest, so there is nothing to correct where
    # the target is at rest, and the ILC needs both ends exactly at 0 V
    T1p = np.interp(src_g, tt, x[lead]["target"], left=0, right=0)
    if whole_record:
        # nothing faded where the target rests: the correction holds over the
        # whole record and only its two ends go exactly to zero (the AWG idles
        # on the first sample; the ILC's gate fades 200 us at each end)
        fade = np.ones_like(T1p)
        n_g = max(2, int(0.2e-3 / dt_g))
        fade[:n_g] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_g))
        fade[-n_g:] = fade[:n_g][::-1]
        fade[[0, -1]] = 0.0
    else:
        fade = np.clip(T1p / (0.02 * T1p.max()), 0, 1)
    corr_g = corr_g * fade

    out = os.path.join(d.folder, "analysis", "target_compare")
    os.makedirs(out, exist_ok=True)
    files, per_crystal = [], {}
    frac = {1: float(split), 2: 1.0 - float(split)}
    if line_src:
        lr_text = (f"{line_f:g} Hz x{line_h} from the {line_src}: "
                   + mains.describe(lfit, unit="mdeg") + " (light - monitors), subtracted")
    else:
        lr_text = ("NOT removed: no drive-off reference scan; this record's undriven "
                   "stretches are too short to fit it (1 Oct: ~27 mdeg at 60 Hz on "
                   "the light)")
    for c in (1, 2):
        ch = x[c]["channel"]
        Tp = np.interp(src_g, tt, x[c]["target"], left=0, right=0)
        Tp[[0, -1]] = 0.0
        hv_played = np.round(Tp * 1000, 6)        # exactly what the CSV holds
        base = (f"{x[c]['name']} target, time-mapped to the experiment's waveform as recorded "
                f"in scan {d.name} (" + "; ".join(
                    (f"transport {j + 1} at +{(E[0] - E1[0])*1e3:.3f} ms: " if len(E1_tr) > 1
                     else "")
                    + f"rise {np.diff(E)[0]*1e3:.3f} / hold {np.diff(E)[1]*1e3:.3f} / "
                    f"fall {np.diff(E)[2]*1e3:.3f} ms; source {np.diff(Sj)[0]*1e3:.3f} / "
                    f"{np.diff(Sj)[1]*1e3:.3f} / {np.diff(Sj)[2]*1e3:.3f} ms"
                    for j, (E, Sj) in enumerate(zip(E1_tr, S_tr))) + ")")
        src_line = (f"source {x[c]['path']}; {len(tg)} pts at {dt_g*1e6:.4f} us; "
                    f"HV volts at the EOM; written {datetime.date.today()} by "
                    f"ramp-polarimeter rampol/ilc_target.py")
        p_play = os.path.join(out, f"target_{x[c]['name']}_played.csv")
        write_target_csv(p_play, tg, hv_played, [base, src_line])
        # deg -> V at the EOM for this crystal's share of the rotation
        k_v = frac[c] / 90 * v90[c] * 1000
        delta, sigma = corr_g * k_v, sig_g * k_v
        meta = {
            "channel": ch, "slot": "optical",
            "units": "V at the EOM (output units, as a target CSV's voltage_V)",
            "quantity": (f"polarization rotation: -(light - monitors), "
                         f"{frac[c]:.0%} of it on this crystal"),
            "sensitivity": f"{90 / (v90[c] * 1000):.6g} deg per V at the EOM",
            "band_hz": float(f_cut),
            "source": os.path.abspath(d.folder),
            "method": ("rotating-analyzer polarimeter, per-sample Malus fit with "
                       "per-angle transmission; light minus the rotation the "
                       f"monitors predict (V90 {v90[1]}/{v90[2]} V), mean of "
                       f"{n_l} legs, time-mapped onto {x[c]['name']}'s record"),
            "line_removed": bool(line_src and lfit is not None),
            "line": lr_text,
            "pd_delay_us": float(pd_delay_us),
            "notes": ("apply to a campaign whose BASE target is target_"
                      f"{x[c]['name']}_played.csv; sigma is per 2 us sample, "
                      "statistical only"),
            "target_file": os.path.abspath(p_play),
            "producer": "ramp-polarimeter rampol/ilc_target.py",
        }
        meta.update(corrmod.target_fingerprint(hv_played, dt_g))
        p_corr = os.path.join(out, f"corr_{x[c]['name']}_optical.csv")
        corrmod.write_correction(p_corr, tg, delta, sigma, meta)
        files += [p_play, p_corr]
        per_crystal[ch] = {"peak_V": float(np.max(np.abs(delta))),
                           "rms_V_active": float(np.sqrt(np.mean(delta[fade > 0.5] ** 2))),
                           "sigma_V_median": float(np.median(sigma[fade > 0.5]))}
        log(f"{ch}: correction {per_crystal[ch]['peak_V']:.1f} V peak, "
            f"{per_crystal[ch]['rms_V_active']:.1f} V rms -> {os.path.basename(p_corr)}")
    # the old tool's light - target files would double-count the monitor
    # error under a re-run ILC: take them out of the way
    for c in (1, 2):
        old = os.path.join(out, f"target_{x[c]['name']}_optcorr.csv")
        if os.path.exists(old):
            os.remove(old)
            log(f"removed {os.path.basename(old)} (light - target: double-counts "
                f"the monitor error once the ILC runs on it)")

    # per transport, from the whole record (a leg's window stops 0.3 ms after
    # its last fall): light - monitors and light - target band-limited at
    # f_cut, against the target of every leg (the EOM-ILC panel's definitions)
    phi_all = np.zeros_like(t)
    for E in legs_E:
        src = warp(Sk, E, t)
        part = 90 * sum(np.interp(src, tt, x[c]["target"], left=0, right=0) / v90[c]
                        for c in (1, 2))
        phi_all += np.where((t > E[0] - pad) & (t < E[-1] + AFTER_FALL + pad), part, 0.0)
    lm_f = corrmod.lowpass(lm, dt_e, f_cut)
    lt_f = corrmod.lowpass(rot - phi_all, dt_e, f_cut)
    per_tr = [transport_numbers(t, lm_f, lt_f, phi_all, legs_tr[k],
                                legs_E[k + 1][0] if k + 1 < len(legs_E) else np.inf)
              for k in range(len(legs_E))]
    if len(S_tr) > 1:
        for k, P in enumerate(per_tr, 1):
            for p in P:
                log(f"leg {k} transport {p['index']}: top light-monitors "
                    f"{p['top_lm_mdeg']:+.0f} mdeg, light-target {p['top_lt_mdeg']:+.0f}; "
                    f"after the fall {p['after_fall_lm_mdeg']:+.0f} / "
                    f"{p['after_fall_lt_mdeg']:+.0f} mdeg")

    summary = {
        "scan": d.name, "angles": pol["n_angles"],
        "fit_rms_mV": float(np.median(pol["rms"]) * 1e3),
        "sig_psi_mdeg_median": float(np.median(sig) * 1e3),
        "timing_reference": ref_name, "timing": timing,
        "source_landmarks_ms": (S * 1e3).tolist(),
        "source_transports_ms": [(s_ * 1e3).tolist() for s_ in S_tr],
        "transports": per_tr[0],
        "rotation_sign_vs_target": sign, "V90_mon": [v90[1], v90[2]],
        "solo": solo, "driven": [f"X{k}" for k in driven], "whole_record": bool(whole_record),
        "targets": [x[1]["path"], x[2]["path"]], "f_cut_Hz": f_cut,
        "pd_delay_us": float(pd_delay_us),
        "legs": [{"leg": L["leg"], **{k: L[k] for k in L if k.endswith("_fit")},
                  "transports": P} for L, P in zip(legs, per_tr)],
        "leg2_minus_leg1": (decompose(legs[1]["d_light"] - legs[0]["d_light"], phi_t, dphi_t, active)
                            if len(legs) == 2 else None),
        "line_ripple": {k: (None if v is None else
                            {"f_line": v["f_line"], "amp": v["amp"].tolist(),
                             "sig": v["sig"].tolist(), "phase_deg": v["phase_deg"].tolist()})
                        for k, v in line.items()},
        "line_ripple_source": line_src or "in-record estimate, not subtracted",
        "correction": {"basis": "-(light - monitors)"
                                + (", line ripple removed" if line_src else ""),
                       "peak_mdeg": float(np.max(np.abs(corr)) * 1e3),
                       "rms_mdeg_active": float(np.sqrt(np.mean(corr[active] ** 2)) * 1e3),
                       "split_X1": frac[1], "per_crystal": per_crystal, "files": files},
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    np.savez_compressed(os.path.join(out, "target_compare.npz"), t_leg1=tl, phi_target=phi_t,
                        correction_deg=corr, t_ilc=tg, correction_ilc_deg=corr_g,
                        sigma_ilc_deg=sig_g, source_knots_s=Sk,
                        **{f"leg{k}_knots_s": E for k, E in enumerate(legs_E, 1)},
                        **{f"leg{L['leg']}_{k}": L[k] for L in legs
                           for k in ("phi_light", "phi_mon", "sig", "d_light", "d_mon",
                                     "d_lm", "d_lm_clean", "line")})
    if figs:
        figures(out, d.name, t, ref_y, tt, src_y, Sk, legs_E, timing, ref_name, tl, phi_t,
                legs, corr, f_cut, active, line, segs, lm, rot, mon, v90, line_src,
                legs_tr=legs_tr)
    summary["out"] = out
    return summary


def figures(out, name, t, ref_y, tt, src_y, S, legs_E, timing, ref_name, tl, phi_t, legs,
            corr, f_cut, active, line, segs, lm, rot, mon, v90, line_src=None, legs_tr=None):
    """fig1-5. S, legs_E: the record's and each leg's time-map knots (4 per
    transport); legs_tr: per leg its transports' landmark arrays (the hold
    start and end of each are drawn dotted), default from legs_E."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eomilc import mains
    cols = {1: "#1f77b4", 2: "#d62728"}
    ms = (tl - tl[0]) * 1e3
    if legs_tr is None:
        legs_tr = [list(np.reshape(np.ravel(E), (-1, 4))) for E in legs_E]

    def holds(axes, t0=None):
        """Each transport's hold start and end, dotted: per leg on the time
        from the trigger (t0 None), else leg 1's on the time from t0."""
        for a_ in axes:
            for k, tr in enumerate(legs_tr if t0 is None else legs_tr[:1], 1):
                for E in tr:
                    for x_ in (E[1], E[2]):
                        a_.axvline((x_ - (t0 or 0.0)) * 1e3, ls=":", lw=0.7,
                                   color=cols.get(k, "0.5") if t0 is None else "0.4")

    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
    ax[0].plot(t * 1e3, ref_y, color="k", lw=0.8, label="recorded")
    for tm, E in zip(timing, legs_E):
        y = tm["gain"] * np.interp(warp(S, E, t), tt, src_y) + tm["offset_V"]
        ax[0].plot(t * 1e3, np.where((t > E[0] - 0.3e-3) & (t < E[-1] + 0.3e-3), y, np.nan),
                   color=cols[tm["leg"]], lw=0.8, ls="--", label=f"ILC record, time-mapped, leg {tm['leg']}")
    ax[0].set_ylabel("V")
    ax[0].set_title(f"Time map of the ILC waveform onto the experiment ({ref_name}, {name})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for tm, E in zip(timing, legs_E):
        y = tm["gain"] * np.interp(warp(S, E, t), tt, src_y) + tm["offset_V"]
        m = (t > E[0] - 0.3e-3) & (t < E[-1] + 0.3e-3)
        ax[1].plot(t[m] * 1e3, (ref_y - y)[m] * 1e3, color=cols[tm["leg"]], lw=0.6,
                   label=f"leg {tm['leg']}: rms {tm['rms_mV']:.1f} mV")
    holds(ax)
    ax[1].set_ylabel("recorded - mapped (mV)")
    ax[1].set_xlabel("time from trigger (ms)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig1_time_map.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    ax[0].plot(ms, phi_t, color="k", lw=1.2, label="ILC target (time-mapped)")
    for L in legs:
        ax[0].plot(ms, L["phi_light"], color=cols[L["leg"]], lw=0.8, ls="--", label=f"light, leg {L['leg']}")
    ax[0].set_ylabel("rotation (deg)")
    ax[0].set_title(f"Polarization rotation and ILC target ({name})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for L in legs:
        ax[1].plot(ms, L["d_light"] * 1e3, color=cols[L["leg"]], lw=0.7, label=f"light - target, leg {L['leg']}")
        ax[1].plot(ms, L["d_mon"] * 1e3, color=cols[L["leg"]], lw=0.7, alpha=0.35,
                   label=f"monitors - target, leg {L['leg']}")
    s = legs[0]["sig"] * 1e3
    ax[1].fill_between(ms, -s, s, color="0.6", alpha=0.5, lw=0, label="+-1 SD, light")
    holds(ax, tl[0])
    ax[1].set_ylabel("difference (mdeg)")
    ax[1].set_xlabel("time from leg start (ms)")
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig2_rotation_vs_target.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    for L in legs:
        ax[0].plot(phi_t[active], L["d_light"][active] * 1e3, ".", ms=1, color=cols[L["leg"]],
                   label=f"light - target, leg {L['leg']}")
        ax[0].plot(phi_t[active], L["d_lm"][active] * 1e3, ".", ms=1, color=cols[L["leg"]], alpha=0.2,
                   label=f"light - monitors, leg {L['leg']}")
    ax[0].set_xlabel("target rotation (deg)")
    ax[0].set_ylabel("difference (mdeg)")
    ax[0].set_title("Differences against target rotation (ramps and hold)")
    ax[0].legend(fontsize=7, markerscale=8)
    ax[0].grid(alpha=0.3)
    if len(legs) == 2:
        ax[1].plot(ms, (legs[1]["d_light"] - legs[0]["d_light"]) * 1e3, color="k", lw=0.7, label="light")
        ax[1].plot(ms, (legs[1]["d_mon"] - legs[0]["d_mon"]) * 1e3, color="0.6", lw=0.7, label="monitors")
        ax[1].set_xlabel("time from leg start (ms)")
        ax[1].set_ylabel("leg 2 - leg 1 (mdeg)")
        ax[1].set_title("Leg 2 minus leg 1 (one drive plays both)")
        holds(ax[1:], tl[0])
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig3_error_structure.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    d_mean = np.mean([L["d_lm_clean"] for L in legs], axis=0)
    ax[0].plot(ms, -d_mean * 1e3, color="0.6", lw=0.6,
               label="-(light - monitors), mean of legs"
                     + (", line ripple removed" if line_src else ""))
    ax[0].plot(ms, corr * 1e3, color="k", lw=1.0, label=f"correction, low-passed at {f_cut:g} Hz")
    ax[0].set_ylabel("rotation (mdeg)")
    ax[0].set_title("Correction to the target rotation: -(light - monitors)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(ms, corr / 2 / 90 * v90[1] * 1e3, label="EO1 target change")
    ax[1].plot(ms, corr / 2 / 90 * v90[2] * 1e3, ls="--", label="EO2 target change")
    ax[1].set_ylabel("V at the EOM")
    ax[1].set_xlabel("time from leg start (ms)")
    ax[1].set_title("Per-crystal change to the ILC targets (half the angle each)")
    holds(ax, tl[0])
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig4_correction.png"), dpi=130)
    plt.close(fig)

    # line ripple on the undriven stretches, with the fitted harmonics
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    any_seg = np.zeros_like(t, bool)
    for sgm in segs:
        any_seg |= sgm
    for a_, (y, key, lab, unit) in zip(ax, (
            (mon[1] * 1e3, "MonX1_mV", "Trek monitor X1", "mV"),
            (mon[2] * 1e3, "MonX2_mV", "Trek monitor X2", "mV"),
            (lm * 1e3, "light_minus_monitors_mdeg", "light - monitors", "mdeg"))):
        a_.plot(t * 1e3, np.where(any_seg, y, np.nan), color="0.4", lw=0.5, label=f"{lab}, undriven")
        f_ = line.get(key)
        if f_ is not None:
            # the whole model: the harmonics plus each stretch's own offset and
            # slope (given the harmonics, those are the line fit of what is left)
            w = mains.line_wave(t, f_)
            for sgm in segs:
                tl_ = t[sgm]
                base = np.polyval(np.polyfit(tl_, (y - w)[sgm], 1), tl_)
                a_.plot(tl_ * 1e3, w[sgm] + base, color="#d62728", lw=1.0)
            a_.set_title(f"{lab}: " + mains.describe(f_, unit=unit), fontsize=8)
        a_.set_ylabel(unit)
        a_.grid(alpha=0.3)
    ax[-1].set_xlabel("time from trigger (ms)")
    fig.suptitle(f"Undriven stretches of {name}; red: harmonics fitted "
                 + (f"in the {line_src}" if line_src else
                    "HERE (in-record: window-dependent, not used)"), fontsize=9)
    fig.savefig(os.path.join(out, "fig5_line_ripple.png"), dpi=130)
    plt.close(fig)

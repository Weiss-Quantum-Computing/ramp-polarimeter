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


def landmarks(tt, T):
    """Rise start, hold start, hold end, fall end of a target (0.1 % / 99.9 %)."""
    on = np.flatnonzero(T > 1e-3 * T.max())
    hold = np.flatnonzero(T > 0.999 * T.max())
    return np.array([tt[on[0]], tt[hold[0]], tt[hold[-1]], tt[on[-1]]])


def warp(S, E, tx):
    """Experiment time -> ILC-record time through landmark pairs E -> S."""
    return np.interp(tx, np.r_[E[0] - 1, E, E[-1] + 1], np.r_[S[0] - 1, S, S[-1] + 1])


def fit_timing(t, y, src_t, src_y, S, start_guess, durations=None, window=12e-3):
    """Fit the time map (and gain, offset) taking src_y on its record onto y,
    the recorded trace. Free: start, and unless given the rise, hold and fall
    durations. Returns (E landmarks, gain, offset, rms)."""
    from scipy.optimize import least_squares
    m = (t > start_guess - 1.5e-3) & (t < start_guess + window)
    S_d = np.diff(S)

    def unpack(p):
        d = np.abs(p[1:4]) if durations is None else durations
        return np.cumsum(np.r_[p[0], d]), p[-2], p[-1]

    def resid(p):
        E, g, o = unpack(p)
        return g * np.interp(warp(S, E, t[m]), src_t, src_y) + o - y[m]
    p0 = ([start_guess, *S_d] if durations is None else [start_guess]) + [1.0, 0.0]
    scale = ([1e-4] * (4 if durations is None else 1)) + [0.01, 0.01]
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
    """The stretches with no ramp: before leg 1 and between the legs (after
    the last leg the lock may be off, so not used). Bool masks."""
    segs = [(t > t[0] + 0.2e-3) & (t < legs_E[0][0] - pre_pad)]
    for a, b in zip(legs_E[:-1], legs_E[1:]):
        segs.append((t > a[-1] + post_pad) & (t < b[0] - pre_pad))
    return [s for s in segs if s.sum() > 20]


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
            scope_grab_path=None, eomilc_path=None, log=print, figs=True):
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
    S = landmarks(tt, x[1]["target"])
    t = d.t
    dt_e = float(np.median(np.diff(t)))
    steps = pol["steps"]

    def mean(r):
        return np.mean([s["v"][r] for s in steps if r in s["v"]], axis=0)
    sign0 = 1.0
    _, _, _, _, mon = light_minus_monitors(d, pol, sign0, v90)

    # the reference for timing: a recorded command against its drive, else a monitor
    ref_k = next((k for k in (1, 2) if f"CmdX{k}" in d.roles), None)
    if ref_k is not None:
        ref_y, src_y, ref_name = mean(f"CmdX{ref_k}"), x[ref_k]["u"], f"CmdX{ref_k} vs {x[ref_k]['name']} drive"
    else:
        ref_k = 1
        ref_y, src_y, ref_name = mon[1], x[1]["target"], f"MonX1 vs {x[1]['name']} target"
    act = np.flatnonzero(mon[1] > 0.01 * mon[1].max())
    E1, gain, off, rms = fit_timing(t, ref_y, tt, src_y, S, t[act[0]])
    timing = [{"leg": 1, "landmarks_ms": (E1 * 1e3).tolist(), "gain": gain, "offset_V": off,
               "rms_mV": rms * 1e3}]
    legs_E = [E1]
    g2 = E1[0] + leg_gap_ms * 1e-3
    if g2 + (E1[-1] - E1[0]) < t[-1]:
        E2, g_, o_, r_ = fit_timing(t, ref_y, tt, src_y, S, g2, durations=np.diff(E1))
        legs_E.append(E2)
        timing.append({"leg": 2, "landmarks_ms": (E2 * 1e3).tolist(), "gain": g_,
                       "offset_V": o_, "rms_mV": r_ * 1e3,
                       "start_after_leg1_ms": (E2[0] - E1[0]) * 1e3})
    log(f"time map ({ref_name}): rise {np.diff(E1)[0]*1e3:.3f} / hold "
        f"{np.diff(E1)[1]*1e3:.3f} / fall {np.diff(E1)[2]*1e3:.3f} ms, "
        f"{rms*1e3:.1f} mV rms")

    sign = -1.0 if np.median(pol["rotation"][(t > E1[1]) & (t < E1[2])]) < 0 else 1.0
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
    tl = np.arange(E1[0] - pad, E1[-1] + pad, dt_e)
    legs = []
    for k, E in enumerate(legs_E, 1):
        te = tl + (E[0] - E1[0])
        src = warp(S, E, te)
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
    corr = -corrmod.lowpass(np.where(active, d_mean, 0.0), dt_e, f_cut)
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
    src_g = warp(S, E1, te_g)
    corr_g = np.interp(te_g, tl, corr, left=0, right=0)
    # per-sample error on the coarser ILC grid: the polarimeter's samples
    # merge, so it falls as sqrt(dt_e / dt_g)
    sig_g = np.interp(te_g, tl, sig_mean, left=0, right=0) * np.sqrt(min(dt_e / dt_g, 1.0))
    # the rotation is measured from rest, so there is nothing to correct where
    # the target is at rest, and the ILC needs both ends exactly at 0 V
    T1p = np.interp(src_g, tt, x[1]["target"], left=0, right=0)
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
                f"in scan {d.name} (rise {np.diff(E1)[0]*1e3:.3f} / hold {np.diff(E1)[1]*1e3:.3f} / "
                f"fall {np.diff(E1)[2]*1e3:.3f} ms; source {np.diff(S)[0]*1e3:.3f} / "
                f"{np.diff(S)[1]*1e3:.3f} / {np.diff(S)[2]*1e3:.3f} ms)")
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

    summary = {
        "scan": d.name, "angles": pol["n_angles"],
        "fit_rms_mV": float(np.median(pol["rms"]) * 1e3),
        "sig_psi_mdeg_median": float(np.median(sig) * 1e3),
        "timing_reference": ref_name, "timing": timing,
        "source_landmarks_ms": (S * 1e3).tolist(),
        "rotation_sign_vs_target": sign, "V90_mon": [v90[1], v90[2]],
        "targets": [x[1]["path"], x[2]["path"]], "f_cut_Hz": f_cut,
        "pd_delay_us": float(pd_delay_us),
        "legs": [{"leg": L["leg"], **{k: L[k] for k in L if k.endswith("_fit")}} for L in legs],
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
                        sigma_ilc_deg=sig_g,
                        **{f"leg{L['leg']}_{k}": L[k] for L in legs
                           for k in ("phi_light", "phi_mon", "sig", "d_light", "d_mon",
                                     "d_lm", "d_lm_clean", "line")})
    if figs:
        figures(out, d.name, t, ref_y, tt, src_y, S, legs_E, timing, ref_name, tl, phi_t,
                legs, corr, f_cut, active, line, segs, lm, rot, mon, v90, line_src)
    summary["out"] = out
    return summary


def figures(out, name, t, ref_y, tt, src_y, S, legs_E, timing, ref_name, tl, phi_t, legs,
            corr, f_cut, active, line, segs, lm, rot, mon, v90, line_src=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eomilc import mains
    cols = {1: "#1f77b4", 2: "#d62728"}
    ms = (tl - tl[0]) * 1e3

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
    ax[0].set_title("Correction to the target rotation: what the monitors cannot see")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(ms, corr / 2 / 90 * v90[1] * 1e3, label="EO1 target change")
    ax[1].plot(ms, corr / 2 / 90 * v90[2] * 1e3, ls="--", label="EO2 target change")
    ax[1].set_ylabel("V at the EOM")
    ax[1].set_xlabel("time from leg start (ms)")
    ax[1].set_title("Per-crystal change to the ILC targets (half the angle each)")
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
            # the fit has its own offset/slope per stretch; draw the harmonics
            # around each stretch's mean so the shape is comparable
            w = mains.line_wave(t, f_)
            for sgm in segs:
                base = np.mean(y[sgm] - w[sgm])
                a_.plot(t[sgm] * 1e3, w[sgm] + base, color="#d62728", lw=1.0)
            a_.set_title(f"{lab}: " + mains.describe(f_, unit=unit), fontsize=8)
        a_.set_ylabel(unit)
        a_.grid(alpha=0.3)
    ax[-1].set_xlabel("time from trigger (ms)")
    fig.suptitle(f"Undriven stretches of {name}; red: harmonics fitted "
                 + (f"in the {line_src}" if line_src else
                    "HERE (in-record: window-dependent, not used)"), fontsize=9)
    fig.savefig(os.path.join(out, "fig5_line_ripple.png"), dpi=130)
    plt.close(fig)

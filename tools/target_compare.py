#!/usr/bin/env python3
"""Measured polarization against the ILC's original target, and the correction
that would close the gap.

    python tools/target_compare.py SCAN_FOLDER --x1 EOM-ILC/run/drive_P92PX1H.state.npz
                                               --x2 EOM-ILC/run/drive_P92PX2A.state.npz

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
    light - target      what a correction has to remove
    monitors - target   the drive's own error as the Trek monitors see it
    light - monitors    what the monitors cannot see
each decomposed over the ramps into a gain (a V90 scale), an offset and a
delay, and what is left.

OUTPUTS (SCAN_FOLDER/analysis/target_compare/, nothing else is touched):
    target_<name>_played.csv    the ILC target with the experiment's timing, on
                                the ILC's 2 us grid and CSV format: what the
                                ILC should converge to for this sequence
    target_<name>_optcorr.csv   the same plus the optical correction: minus the
                                light's error (mean of the legs), low-passed,
                                split equally between the crystals, zero at
                                both ends
    summary.json, target_compare.npz, fig1-4
The leg-to-leg difference cannot be removed by one waveform played on both
legs; it is reported separately.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import analysis as an, config, hw  # noqa: E402

V90 = {"EO1": 5.1283, "EO2": 5.1374}   # monitor volts for 90 deg (V_pi, 31 Aug 2026)


def load_ilc(path):
    z = np.load(path, allow_pickle=True)
    return {"t": np.asarray(z["t"], float), "target": np.asarray(z["target"], float),
            "u": np.asarray(z["u"], float), "name": str(z["name"]),
            "channel": str(z["channel"]), "path": path}


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


def lowpass(y, dt, f_cut):
    """Zero-phase low-pass: cosine roll-off from 0.7 f_cut to f_cut, on the
    record mirrored so its ends do not wrap into each other."""
    n = len(y)
    ext = np.r_[y, y[::-1]]
    F = np.fft.rfft(ext)
    f = np.fft.rfftfreq(len(ext), dt)
    w = np.clip((f_cut - f) / (0.3 * f_cut), 0, 1)
    return np.fft.irfft(F * (0.5 - 0.5 * np.cos(np.pi * w)), len(ext))[:n]


def write_target_csv(path, t, hv, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(f"# {ln}\n")
        fh.write("time_us,voltage_V\n")
        for a, b in zip(t, hv):
            fh.write(f"{a * 1e6:.6f},{b:.6f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scan")
    ap.add_argument("--x1", required=True, help="ILC state for EO1 (drive_*.state.npz)")
    ap.add_argument("--x2", required=True, help="ILC state for EO2")
    ap.add_argument("--f-cut", type=float, default=2000.0,
                    help="correction bandwidth, Hz (default 2000: below the 4.94 kHz "
                         "motional resonance the ILC's own error was found to heat)")
    ap.add_argument("--lock-tol", type=float, default=0.006)
    ap.add_argument("--leg-gap-ms", type=float, default=16.667)
    a = ap.parse_args(argv)

    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    d = an.load_scan(a.scan, sg.load_capture, lock_tol=a.lock_tol)
    pol = an.polarization(d)
    x = {1: load_ilc(a.x1), 2: load_ilc(a.x2)}
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
    mean = lambda r: np.mean([s["v"][r] for s in steps if r in s["v"]], axis=0)
    rest = an.rest_index(t)
    mon = {k: mean(f"MonX{k}") - np.median(mean(f"MonX{k}")[rest]) for k in (1, 2)}

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
    g2 = E1[0] + a.leg_gap_ms * 1e-3
    if g2 + (E1[-1] - E1[0]) < t[-1]:
        E2, g_, o_, r_ = fit_timing(t, ref_y, tt, src_y, S, g2, durations=np.diff(E1))
        legs_E.append(E2)
        timing.append({"leg": 2, "landmarks_ms": (E2 * 1e3).tolist(), "gain": g_,
                       "offset_V": o_, "rms_mV": r_ * 1e3,
                       "start_after_leg1_ms": (E2[0] - E1[0]) * 1e3})

    sign = -1.0 if np.median(pol["rotation"][(t > E1[1]) & (t < E1[2])]) < 0 else 1.0
    rot = sign * pol["rotation"]
    sig = pol["sig_psi"]
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
        phi_m = 90 * (np.interp(te, t, mon[1]) / v90[1] + np.interp(te, t, mon[2]) / v90[2])
        legs.append({"leg": k, "phi_target": phi_t, "phi_light": phi_l, "phi_mon": phi_m,
                     "sig": np.interp(te, t, sig), "d_light": phi_l - phi_t,
                     "d_mon": phi_m - phi_t, "d_lm": phi_l - phi_m})
    phi_t = legs[0]["phi_target"]
    dphi_t = np.gradient(phi_t, tl)
    active = phi_t > 0.5
    for L in legs:
        for key in ("d_light", "d_mon", "d_lm"):
            L[key + "_fit"] = decompose(L[key], phi_t, dphi_t, active)

    # the correction, on leg-1 time; then onto the ILC grid with the played timing
    d_mean = np.mean([L["d_light"] for L in legs], axis=0)
    corr = -lowpass(np.where(active, d_mean, 0.0), dt_e, a.f_cut)
    n_t = int(0.2e-3 / dt_e)
    taper = np.ones_like(corr)
    taper[:n_t] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_t))
    taper[-n_t:] = taper[:n_t][::-1]
    corr *= taper
    # ILC grid: same 2 us spacing and length, the played rise starting where the
    # original's did, so the record's padding is unchanged
    tg = tt.copy()
    te_g = tg - S[0] + E1[0]
    src_g = warp(S, E1, te_g)
    corr_g = np.interp(te_g, tl, corr, left=0, right=0)
    # fade the correction out where the target is at rest: the rotation is
    # measured from rest, so there is nothing to correct there, and the ILC
    # needs both ends of the record at exactly 0 V
    T1p = np.interp(src_g, tt, x[1]["target"], left=0, right=0)
    corr_g = corr_g * np.clip(T1p / (0.02 * T1p.max()), 0, 1)
    out = os.path.join(d.folder, "analysis", "target_compare")
    os.makedirs(out, exist_ok=True)
    files = []
    for c in (1, 2):
        Tp = np.interp(src_g, tt, x[c]["target"], left=0, right=0)
        dT = corr_g / 2 / 90 * v90[c]
        Tp[[0, -1]] = 0.0
        base = (f"{x[c]['name']} target, time-mapped to the experiment's waveform as recorded "
                f"in scan {d.name} (rise {np.diff(E1)[0]*1e3:.3f} / hold {np.diff(E1)[1]*1e3:.3f} / "
                f"fall {np.diff(E1)[2]*1e3:.3f} ms; source {np.diff(S)[0]*1e3:.3f} / "
                f"{np.diff(S)[1]*1e3:.3f} / {np.diff(S)[2]*1e3:.3f} ms)")
        src_line = (f"source {x[c]['path']}; {len(tg)} pts at {np.median(np.diff(tg))*1e6:.4f} us; "
                    f"HV volts at the EOM; written {datetime.date.today()} by "
                    f"ramp-polarimeter tools/target_compare.py")
        p_play = os.path.join(out, f"target_{x[c]['name']}_played.csv")
        write_target_csv(p_play, tg, Tp * 1000, [base, src_line])
        p_corr = os.path.join(out, f"target_{x[c]['name']}_optcorr.csv")
        write_target_csv(p_corr, tg, (Tp + dT) * 1000,
                         [base + f"; plus the optical correction from that scan: minus the "
                          f"light's error (mean of {len(legs)} legs), low-passed at "
                          f"{a.f_cut:g} Hz, half the angle on each crystal", src_line])
        files += [p_play, p_corr]

    summary = {
        "scan": d.name, "angles": pol["n_angles"],
        "fit_rms_mV": float(np.median(pol["rms"]) * 1e3),
        "sig_psi_mdeg_median": float(np.median(sig) * 1e3),
        "timing_reference": ref_name, "timing": timing,
        "source_landmarks_ms": (S * 1e3).tolist(),
        "rotation_sign_vs_target": sign, "V90_mon": [v90[1], v90[2]],
        "targets": [x[1]["path"], x[2]["path"]], "f_cut_Hz": a.f_cut,
        "legs": [{"leg": L["leg"], **{k: L[k] for k in L if k.endswith("_fit")}} for L in legs],
        "leg2_minus_leg1": (decompose(legs[1]["d_light"] - legs[0]["d_light"], phi_t, dphi_t, active)
                            if len(legs) == 2 else None),
        "correction": {"peak_mdeg": float(np.max(np.abs(corr)) * 1e3),
                       "rms_mdeg_active": float(np.sqrt(np.mean(corr[active] ** 2)) * 1e3),
                       "peak_dT_mV_mon_per_crystal": float(np.max(np.abs(corr)) / 2 / 90 * 5.13 * 1e3),
                       "files": files},
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    np.savez_compressed(os.path.join(out, "target_compare.npz"), t_leg1=tl, phi_target=phi_t,
                        correction_deg=corr, t_ilc=tg, correction_ilc_deg=corr_g,
                        **{f"leg{L['leg']}_{k}": L[k] for L in legs
                           for k in ("phi_light", "phi_mon", "sig", "d_light", "d_mon", "d_lm")})
    figures(out, d.name, t, ref_y, tt, src_y, S, legs_E, timing, ref_name, tl, phi_t, legs,
            corr, a.f_cut, active)
    return summary


def figures(out, name, t, ref_y, tt, src_y, S, legs_E, timing, ref_name, tl, phi_t, legs,
            corr, f_cut, active):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    d_mean = np.mean([L["d_light"] for L in legs], axis=0)
    ax[0].plot(ms, -d_mean * 1e3, color="0.6", lw=0.6, label="-(light - target), mean of legs")
    ax[0].plot(ms, corr * 1e3, color="k", lw=1.0, label=f"correction, low-passed at {f_cut:g} Hz")
    ax[0].set_ylabel("rotation (mdeg)")
    ax[0].set_title("Proposed correction to the target rotation")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(ms, corr / 2 / 90 * 5.1283 * 1e3, label="EO1 target change")
    ax[1].plot(ms, corr / 2 / 90 * 5.1374 * 1e3, ls="--", label="EO2 target change")
    ax[1].set_ylabel("monitor mV (= V at the EOM)")
    ax[1].set_xlabel("time from leg start (ms)")
    ax[1].set_title("Per-crystal change to the ILC targets (half the angle each)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(out, "fig4_correction.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    s = main()
    print(json.dumps(s, indent=1))

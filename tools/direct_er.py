#!/usr/bin/env python3
"""Directly measured extinction ratios of a ramp scan - no Malus fit in the values.

    python tools/direct_er.py SCAN_FOLDER

At every time the ramp sweeps the light through crossed for one of the scan's
analyzer angles: Imin = that angle's measured trace at its minimum (4 us
boxcar, dark subtracted, no drift correction), Imax = the trace of the angle
90 deg away at the same instant. The fit's azimuth is used only to find WHEN
a crossing happens. Static stretches (rest, holds, between legs): the
measured angle nearest crossed against the one 90 deg from it, with how far
from crossed that angle sat (Imax sin^2 of it is the Imin that offset alone
would give). Lower bounds where Imin is under 2 sigma.

Writes SCAN_FOLDER/analysis/direct_er/direct_er.json and direct_er.png.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rampol import analysis as an, config, hw

if len(sys.argv) < 2:
    sys.exit(__doc__)
SCAN = sys.argv[1]
sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
d = an.load_scan(SCAN, sg.load_capture, lock_tol=0.006)
pol = an.polarization(d)
t = d.t
dt = float(np.median(np.diff(t)))
th, I, sem_s, steps = an.scan_matrix(d, "scan", correct_drift=False)
sem = np.array([np.asarray(s["sem"]["PD"], float) for s in steps])
nb = max(1, int(round(4e-6 / dt)))            # 4 us boxcar
box = np.ones(nb) / nb
Is = np.array([np.convolve(x, box, mode="same") for x in I])
sems = sem / np.sqrt(nb)
psi = pol["psi_u"]
rot = -pol["rotation"]                       # target sense (test-4: light turns negative)
wrap = lambda x: (x + 90.0) % 180.0 - 90.0
orth = [[j for j in range(len(th)) if abs(wrap(th[j] - th[k] - 90)) < 1.0] for k in range(len(th))]
segs = an.segments(t, pol["rotation"])
moving = np.zeros(len(t), bool)
for s in segs:
    if s["kind"].startswith(("up", "down")):
        moving |= (t >= s["t0"]) & (t <= s["t1"])
t_end = max(s["t1"] for s in segs if s["kind"].startswith("down")) + 1e-3   # lock off after leg 2

pts = []
for k in range(len(th)):
    if not orth[k]:
        continue
    delta = wrap(psi - th[k] - 90)
    cross = np.flatnonzero((np.sign(delta[:-1]) != np.sign(delta[1:])) & moving[:-1]
                           & (np.abs(delta[:-1]) < 5))
    for c in cross:
        a, b = c, c
        while a > 0 and abs(delta[a - 1]) < 4 and moving[a - 1]:
            a -= 1
        while b < len(t) - 1 and abs(delta[b + 1]) < 4 and moving[b + 1]:
            b += 1
        if b - a < 3 * nb:
            continue
        m = a + nb + int(np.argmin(Is[k, a + nb:b - nb + 1]))
        imin, s_min = Is[k, m], sems[k, m]
        imax = float(np.mean([Is[j, m] for j in orth[k]]))
        seg = next(s["kind"] for s in segs if s["t0"] <= t[m] <= s["t1"])
        lower = imin < 2 * s_min
        pts.append(dict(kind="crossing", seg=seg, t_ms=t[m] * 1e3, theta=float(th[k]),
                        rotation=float(rot[m]), rate=float(abs(np.gradient(rot, t)[m]) * 1e-3),
                        imin_mV=imin * 1e3, sig_mV=s_min * 1e3, imax_V=imax,
                        er=imax / (2 * s_min) if lower else imax / imin, lower=bool(lower)))
# static stretches: the measured angle nearest crossed
for s in segs:
    if s["kind"].startswith(("up", "down")) or s["t0"] > t_end:
        continue
    w = (t > s["t0"] + 0.2e-3) & (t < min(s["t1"], t_end) - 0.2e-3)
    if w.sum() < 100:
        continue
    means = I[:, w].mean(axis=1)
    k = int(np.argmin(means))
    if not orth[k]:
        continue
    imax = float(np.mean([means[j] for j in orth[k]]))
    s_min = float(np.mean(sem[k, w]) / np.sqrt(w.sum() / max(1, int(1e-6 / dt))))
    off = float(np.median(wrap(psi[w] - th[k] - 90)))
    pts.append(dict(kind="static", seg=s["kind"], t_ms=float(t[w].mean() * 1e3),
                    theta=float(th[k]), rotation=float(np.median(rot[w])), rate=0.0,
                    imin_mV=float(means[k] * 1e3), sig_mV=s_min * 1e3, imax_V=imax,
                    er=imax / means[k], lower=False, off_deg=off,
                    imin_from_offset_mV=float(imax * np.sin(np.deg2rad(off)) ** 2 * 1e3)))

out = os.path.join(SCAN, "analysis", "direct_er")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "direct_er.json"), "w") as fh:
    json.dump(pts, fh, indent=1)
cr = [p for p in pts if p["kind"] == "crossing"]
st = [p for p in pts if p["kind"] == "static"]
print(f"{d.name}: {len(cr)} crossings, {len(st)} static stretches; PD at "
      f"{an._pd_vdiv(d, 'scan')} V/div")
lo = min(pts, key=lambda p: p["er"])
print(f"LOWEST direct ER {lo['er']:.1f}: {lo['kind']} {lo['seg']}, t {lo['t_ms']:.3f} ms, rotation "
      f"{lo['rotation']:.1f} deg, analyzer {lo['theta']:g}, Imin {lo['imin_mV']:.1f} +- {lo['sig_mV']:.2f} mV, Imax {lo['imax_V']:.3f} V")
ers = np.array([p["er"] for p in cr])
print("crossing ER percentiles 0/10/50/90/100:", np.percentile(ers, [0, 10, 50, 90, 100]).round(1))
print("crossings by rotation band (median ER, min ER, n):")
for a, b in ((0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 185)):
    sel = [p["er"] for p in cr if a <= p["rotation"] < b]
    if sel:
        print(f"  {a:3d}-{b:3d} deg: median {np.median(sel):7.1f}  min {min(sel):7.1f}  n {len(sel)}")
print("lower bounds (Imin under 2 sigma):", sum(p["lower"] for p in pts))
for p in st:
    print(f"static {p['seg']:8s} t {p['t_ms']:6.2f} ms rot {p['rotation']:7.2f}: analyzer {p['theta']:g} "
          f"({p['off_deg']:+.2f} deg from crossed), Imin {p['imin_mV']:.2f} +- {p['sig_mV']:.3f} mV "
          f"(offset alone would give {p['imin_from_offset_mV']:.2f}), ER {p['er']:.0f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8.5, 7.5), constrained_layout=True)
cols = {"up": "#1f77b4", "down": "#d62728"}
er_fit = pol["imax"] / np.maximum(pol["imin"], 1e-6)
act = (t < t_end)
ax[0].plot(rot[act], er_fit[act], ",", color="0.75", label="per-sample Malus fit (for comparison)")
for leg in ("1", "2"):
    for dr, mk in (("up", "o"), ("down", "s")):
        sel = [p for p in cr if p["seg"] == f"{dr} {leg}" and not p["lower"]]
        if sel:
            ax[0].plot([p["rotation"] for p in sel], [p["er"] for p in sel], mk, ms=4,
                       mfc=cols[dr] if leg == "1" else "none", mec=cols[dr],
                       label=f"crossing, {dr} ramp, leg {leg}")
lb = [p for p in cr if p["lower"]]
if lb:
    ax[0].plot([p["rotation"] for p in lb], [p["er"] for p in lb], "^", color="k", ms=5,
               label="lower bound (Imin < 2 sigma)")
ax[0].plot([p["rotation"] for p in st], [p["er"] for p in st], "D", color="#2ca02c", ms=6,
           label="static stretch, nearest measured angle")
for p in st:
    ax[0].annotate(p["seg"], (p["rotation"], p["er"]), fontsize=7, xytext=(4, 4),
                   textcoords="offset points", color="#2ca02c")
ax[0].set_yscale("log")
ax[0].set_ylim(5, 2e4)
ax[0].set_xlabel("rotation from rest (deg)")
ax[0].set_ylabel("extinction ratio, Imax / Imin")
ax[0].set_title(f"Directly measured extinction ratio, {d.name} ({len(th)} analyzer angles, "
                f"PD at {an._pd_vdiv(d, 'scan'):g} V/div)", fontsize=9)
ax[0].legend(fontsize=7, ncol=2, loc="lower center")
ax[0].grid(alpha=0.3, which="both")
for leg in ("1", "2"):
    for dr, mk in (("up", "o"), ("down", "s")):
        sel = [p for p in cr if p["seg"] == f"{dr} {leg}"]
        if sel:
            ax[1].errorbar([p["rotation"] for p in sel], [p["imin_mV"] for p in sel],
                           [p["sig_mV"] for p in sel], fmt=mk, ms=4,
                           mfc=cols[dr] if leg == "1" else "none", mec=cols[dr],
                           color=cols[dr], label=f"{dr} ramp, leg {leg}")
ax[1].errorbar([p["rotation"] for p in st], [p["imin_mV"] for p in st],
               [p["sig_mV"] for p in st], fmt="D", color="#2ca02c", ms=6, label="static")
ax[1].axhline(40.25 / 1, color="k", lw=0.6, ls=":")
ax[1].text(2, 42, "one ADC code at 1 V/div (40 mV)", fontsize=7)
ax[1].set_yscale("log")
ax[1].set_xlabel("rotation from rest (deg)")
ax[1].set_ylabel("Imin, measured (mV, dark subtracted)")
ax[1].legend(fontsize=7, ncol=3)
ax[1].grid(alpha=0.3, which="both")
fig.savefig(os.path.join(out, "direct_er.png"), dpi=130)
print(os.path.join(out, "direct_er.png"))

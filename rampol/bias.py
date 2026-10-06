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


def windows(p):
    """(hold window, idle window) in seconds after the trigger."""
    t_hold = (p["lead_ms"] + p["rise_ms"]) * 1e-3
    w = (t_hold + p["settle_ms"] * 1e-3, t_hold + (p["hold_ms"] - 0.2) * 1e-3)
    idle = (-1.0, (p["lead_ms"] - 0.05) * 1e-3)
    if w[1] - w[0] < 0.5e-3:
        raise ValueError(f"the hold ({p['hold_ms']} ms) leaves under 0.5 ms after "
                         f"the {p['settle_ms']} ms settle")
    return w, idle


def check_plateaus(biases, p, eomilc=None):
    """[(bias, {ch: u peak}, report text)] and raises ValueError on the first
    that fails the AWG cap or (with eomilc) the Trek chain's limit check."""
    out = []
    dt = p["dt_us"] * 1e-6
    if not 0.0 <= float(p["split"]) <= 1.0:
        raise ValueError(f"split {p['split']} must be between 0 and 1")
    idle = p.get("idle") or {}
    for b in sorted(set(biases)):
        if b < 0:
            raise ValueError(f"bias {b:g} deg: negative biases are not driven "
                             f"(the ramps' drives are unipolar)")
        v = awg_volts(b, p["split"])
        msgs = []
        for name, amp in v.items():
            i0 = float(idle.get(name, 0.0))
            if abs(i0) > 0.1:
                raise ValueError(f"{name} idle {i0*1e3:+.0f} mV: past the 100 mV idle cap")
            if abs(amp + i0) > p["awg_max"]:
                raise ValueError(
                    f"bias {b:g} deg needs {amp + i0:.2f} V on {name} at the AWG, past "
                    f"the {p['awg_max']:g} V cap - lower the bias or change the split")
            if eomilc is not None and abs(amp) > 0:
                from eomilc.config import CHANNELS
                from eomilc.ilc import check_limits
                ch = CHANNELS[name]
                _, u = plateau(amp, p, i0)
                rep = check_limits(u, u * CHAN[name]["gain"], dt, ch, ch.limits)
                if not rep.ok:
                    raise ValueError(f"bias {b:g} deg, {name}: {rep}")
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
    i0 = max(float(np.min(I)), 0.0)

    def resid(p):
        tn, imin, k = p
        return (imin + k * np.sin(np.deg2rad(th - tn)) ** 2 - I) * w
    r = least_squares(resid, [tn0, i0, k0], x_scale=[0.1, max(abs(i0), 1e-6), k0])
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


# ------------------------------------------------------------- the run
class BiasRun:
    """One bias-point measurement on the bench. `link` is hw.ScopeLink, `rot`
    hw.Rotator, `awg` a BK4063B (or the simulator's), `roles` {role: channel}.
    `ask(title, text)` -> bool comes from the GUI thread (blocking is fine: it
    is called on the worker). `on_point(point)` gets every finished point."""

    def __init__(self, folder, name, link, rot, awg, roles, plan=None, log=print,
                 cancelled=None, ask=None, on_point=None, progress=None,
                 eomilc=None, ilc_bench=None, provenance=None, session=None):
        self.folder = os.path.join(folder, name)
        self.provenance = provenance
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
        self.sess = session or awgmod.Session(awg, ilc_bench, log=log)
        self.points = []
        self.dark = {}
        self.manifest = None

    # -- helpers ----------------------------------------------------------
    def _check(self):
        if self.cancelled():
            from .hw import Cancelled
            raise Cancelled()

    def _chans(self):
        want = ["PD", "MonX1", "MonX2", "CmdX1", "CmdX2"]
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
        sc, off = setting
        m = (t >= w[0]) & (t <= w[1])
        hi, lo = off + 3.9 * sc, off - 3.9 * sc
        x = stack[:, m]
        return bool((x > hi).any() or (x < lo).any())

    # -- the AWG -----------------------------------------------------------
    def _wave(self, bias_deg):
        """The plateau for a bias on both channels, as an awg.Wave."""
        from . import awg as awgmod
        p = self.p
        idle = p.get("idle") or {}
        volts = awg_volts(bias_deg, p["split"])
        u, t = {}, None
        for name in CHAN:
            t, u[name] = plateau(volts[name], p, float(idle.get(name, 0.0)))
        return awgmod.Wave(t, u, p["dt_us"] * 1e-6, f"bias {bias_deg:g} deg",
                           rotation=bias_deg, source="bias")

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
    def run(self):
        p = self.p
        if "PD" not in self.roles:
            raise RuntimeError("no channel has the PD role")
        biases = order_biases(parse_biases(p["biases"]), p["order"])
        checks = check_plateaus(biases, p, self.eomilc)
        t_rec, _ = plateau(0.0, p)
        period = float(t_rec[-1] + p["dt_us"] * 1e-6)
        w, idle = windows(p)
        os.makedirs(self.folder, exist_ok=True)
        self.manifest = {"format": FORMAT, "name": self.name, "plan": p,
                         "created": datetime.datetime.now().isoformat(timespec="seconds"),
                         "roles": self.roles, "window_s": w, "biases": biases,
                         "limit_checks": [c[2] for c in checks], "points": [],
                         "provenance": self.provenance}
        self._save()
        self.log(f"Bias run {self.name}: {len(biases)} points, record "
                 f"{period*1e3:.2f} ms, window {w[0]*1e3:.2f}-{w[1]*1e3:.2f} ms")
        sc = self.link.scope
        saved_tb = {k: sc.get(k) for k in (":TIMebase:SCALe", ":TIMebase:POSition",
                                           ":TIMebase:REFerence")}
        pd = self.roles["PD"]
        saved_pd = self.link.channel_state([pd])[pd]
        coarse = saved_pd
        on = False
        try:
            # the scope covers the record: 10 divisions from the lead's start
            div = _nice_up(period * 1.05 / 10)
            sc.put(":TIMebase:REFerence", "LEFT")
            sc.put(":TIMebase:SCALe", f"{div:.6g}")
            sc.put(":TIMebase:POSition", f"{-0.2e-3 + div:.6g}")   # LEFT: start 1 div before
            self.log(f"  scope {div*1e3:g} ms/div from -0.2 ms; PD coarse "
                     f"{coarse[0]:g} V/div offset {coarse[1]:g} V")
            on = True                  # from here every exit ends the AWG
            self._play(biases[0])
            end = "parked at idle" if p.get("end") == "park" else "OFF"
            if not self.ask("Bias points", "The AWG holds the plateaus. Switch "
                            f"both outputs ON now? (At the end they go {end}.)"):
                raise RuntimeError("outputs left off - nothing measured")
            self.sess.on()
            time.sleep(p["upload_settle_s"])

            # Imax at the first bias from 4 angles (one angle can sit at the
            # null), then the dark at every V/div used
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

            psi_prev = None
            for i, b in enumerate(biases):
                self._check()
                d = "up" if (p["order"] != "updown" or i < len(biases) // 2 + 1) else "down"
                self.progress(i, len(biases), f"bias {b:g} deg ({i + 1}/{len(biases)})")
                if i > 0:
                    self._play(b)
                    time.sleep(p["upload_settle_s"])
                pt = self._point(i, b, d, coarse, null_sets, w, idle, psi_prev)
                psi_prev = pt.get("psi")
                self.points.append(pt)
                self.manifest["points"].append(pt)
                self._save()
                if self.on_point:
                    self.on_point(pt)
            self.progress(len(biases), len(biases), "bias points done")
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
                self.manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
                self._save()
        return self.points

    def _point(self, i, b, direction, coarse, null_sets, w, idle, psi_prev):
        p = self.p
        dark_c = self.dark[_key(coarse)][0]
        mons = {}
        # 1. where is the polarization: 4 angles, Malus
        base = 0.0 if psi_prev is None else psi_prev
        th4 = [(base + a) % 180 for a in (0, 45, 90, 135)]
        I4, traces = [], {}
        for a in th4:
            t, s = self._acquire(a, coarse)
            I4.append(self._window(t, s["PD"], w)[0] - dark_c)
            for r in ("MonX1", "MonX2"):
                if r in s:
                    mons.setdefault(r, []).append(s[r])
            traces[f"coarse_{a:.2f}"] = s["PD"].mean(axis=0)
        psi, imax4, imin4 = malus4(th4, I4)
        theta0 = (psi + 90) % 180
        # 2. around the null at the most sensitive setting that holds it
        est = max(imin4, 0.0) + imax4 * math.sin(math.radians(p["null_half_deg"] + 1)) ** 2
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
                self.log(f"  bias {b:g}: off screen at the null V/div, now "
                         f"{setting[0]*1e3:g} mV/div")
                continue
            fit = fit_null(th, I, S)
            scan = {"theta": th.tolist(), "I": I, "sem": S, "vdiv": setting[0]}
            if (fit["inside"] and abs(fit["theta_n"] - theta0) < 0.6 * p["null_half_deg"]) \
                    or recentred >= 2:
                break
            theta0 = fit["theta_n"]
            recentred += 1
            self.log(f"  bias {b:g}: null at {theta0:.2f} deg, off centre - again")
        # 3. Imax at the bright angle
        t, s = self._acquire((fit["theta_n"] + 90) % 180, coarse)
        imax, imax_s = self._window(t, s["PD"], w)
        imax -= dark_c
        if self._clipped(s["PD"], coarse, t, w):
            self.log(f"  bias {b:g}: the bright reading clips at the coarse V/div")
        traces["bright"] = s["PD"].mean(axis=0)
        for r in ("MonX1", "MonX2"):
            if r in s:
                mons.setdefault(r, []).append(s[r])
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
        pt = {"i": i, "bias": b, "dir": direction, "awg": awg_volts(b, p["split"]),
              "mon_V": mv, "phi_mon": phi if mv else None,
              "psi_coarse": psi, "imax4": imax4, "imin4": imin4,
              "theta_n": fit["theta_n"], "sig_theta_n": fit["sig_theta_n"],
              "psi": (fit["theta_n"] - 90) % 180, "fit": fit, "scan": scan,
              "imax": imax, "sig_imax": imax_s, **er}
        np.savez_compressed(os.path.join(self.folder, f"point_{i:02d}.npz"),
                            t=t[::10], **{k: v[::10] for k, v in traces.items()})
        e = (f"ER {er['er']:.0f}" if er["er"] else f"ER > {er['er_lower']:.0f}")
        self.log(f"  bias {b:6.1f} deg ({direction}): null {fit['theta_n']:7.3f} "
                 f"+- {fit['sig_theta_n']*1e3:.0f} mdeg, Imin {fit['imin']*1e3:.3f} "
                 f"+- {fit['sig_imin']*1e3:.3f} mV, Imax {imax:.3f} V, {e}, "
                 f"monitors {phi:+.2f} deg")
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
                  session=None):
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
    fit, window)."""
    import tempfile
    p = dict(PLAN, **(plan or {}))
    br = BiasRun(tempfile.gettempdir(), "find", link, rot, awg, roles, plan=p,
                 log=log, cancelled=cancelled, ask=ask, eomilc=eomilc,
                 ilc_bench=ilc_bench, session=session)
    half = float(half_deg if half_deg is not None else (p["null_half_deg"] if kind == "min" else 10.0))
    npts = int(points or p["null_points"])
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
            t, s = br._acquire(a, coarse)
            I4.append(br._window(t, s["PD"], w)[0])
        psi, imax4, imin4 = malus4(th4, I4)
        amp = max(imax4 - imin4, 1e-4)
        log(f"Find {kind}: azimuth {psi:.2f} deg from 4 angles, Imax {imax4:.3f} V, "
            f"Imin {imin4*1e3:.1f} mV (raw, coarse) in "
            + ("the whole record" if window_s is None else
               f"{w[0]*1e3:.2f}..{w[1]*1e3:.2f} ms"))
        if kind == "min":
            center = (psi + 90) % 180
            ladder = _ladder(amp, half, coarse[0])
            est = amp * math.sin(math.radians(half + 1)) ** 2
            sets = [(v, imin4 + 3 * v) for v in ladder] + [coarse]
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
                t, s = br._acquire(a % 180, setting)
                if setting != coarse and br._clipped(s["PD"], setting, t, w):
                    clipped = True
                    break
                m_, se = br._window(t, s["PD"], w)
                I.append(m_)
                S.append(max(se, 1e-6))
            if clipped:
                k = min(k + 1, len(sets) - 1)
                log(f"  off screen at {setting[0]*1e3:g} mV/div - now {sets[k][0]*1e3:g} mV/div")
                continue
            y = np.array(I) if kind == "min" else -np.array(I)
            fit = fit_null(th, y, S)
            if (fit["inside"] and abs(fit["theta_n"] - center) < 0.6 * half) or recentred >= 2:
                break
            center = fit["theta_n"]
            recentred += 1
            log(f"  extremum at {center:.2f} deg, off centre - again")
        angle = fit["theta_n"] % 180
        got = rot.approach(angle)
        level = fit["imin"] if kind == "min" else -fit["imin"]
        log(f"  {kind} transmission at analyzer {angle:.3f} +- {fit['sig_theta_n']*1e3:.0f} "
            f"mdeg ({level*1e3:.2f} mV raw at {setting[0]*1e3:g} mV/div); analyzer now "
            f"at {got:.3f}")
        return {"kind": kind, "angle": float(angle), "sig": fit["sig_theta_n"],
                "level": float(level), "psi_coarse": psi, "theta": th.tolist(),
                "I": [float(x) for x in I], "sem": S, "vdiv": setting[0], "fit": fit,
                "window": None if window_s is None else list(w), "bias": bias_deg}
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

"""From a scan folder to polarization vs time.

Three measurements come out of one rotating-analyzer scan:

1. A harmonic fit at every time sample. Across the analyzer angles theta_k,
   I(theta) = a0 + c2 cos 2theta + s2 sin 2theta (+ diagnostic 1theta and
   4theta terms). That gives the azimuth psi = atan2(s2, c2)/2 (the analyzer
   angle of maximum transmission), Imax = a0 + B and Imin = a0 - B with
   B = hypot(c2, s2), the visibility B/a0 and ER_fit = Imax/Imin - at every
   point of the ramp from the same data.

2. Extinction ratio from the dips. With the analyzer fixed at theta_k the ramp
   sweeps the polarization past the crossed position theta_k + 90 at some time
   t_c; near it I(t) = Imin + Imax sin^2(delta(t)) with delta = psi(t) -
   theta_k - 90 known from (1). Fitting that dip gives Imin directly from
   samples near the null instead of as the small difference of two large fit
   terms. Every scan angle gives one point on the up ramp and one on the down
   ramp - with 5 deg steps, an ER every 5 deg of rotation.

3. Null refinement for static parts (rest, hold, after the ramp), where the
   polarization does not sweep through any null: a few analyzer angles close
   to crossed, at a sensitive V/div, fitted for Imin.

Everything here is numpy. Captures are read with Scope Grab's own
load_capture (passed in), so the NPZ layout has one reader per program.
"""
import json
import os

import numpy as np


class ScanData:
    """A scan folder loaded: per-step block-mean traces by role."""

    def __init__(self, manifest, folder):
        self.manifest = manifest
        self.folder = folder
        self.name = manifest.get("name", "")
        self.t = None
        self.steps = []               # dicts: the manifest step + arrays
        self.roles = {}               # role -> ch
        self.notes = []


def _role_columns(columns, channels):
    """{role: column index} from capture columns like CH3_Trek_monitor_X1_V."""
    out = {}
    for j, c in enumerate(columns):
        if not c.startswith("CH"):
            continue
        ch = c[2:].split("_", 1)[0]
        info = channels.get(ch)
        if info and info["role"] not in ("off",):
            out[info["role"]] = j
    return out


def _step_key(folder, s, trim):
    """What a step's reduced data depends on: its files as they are on disk."""
    parts = [trim]
    for f in s.get("files", []):
        try:
            st = os.stat(os.path.join(folder, f))
            parts.append((f, st.st_size, st.st_mtime_ns))
        except OSError:
            parts.append((f, None, None))
    return tuple(parts)


def _reduce_step(s, folder, chans, load_capture, trim, notes):
    """Read one step's files and reduce them as they come: per role the sum
    and sum of squares (for mean and standard error) and the off-screen mask;
    the PD's shots are kept, as float32, for the missed-lock check. Holding
    every shot of every channel instead peaked above 1 GB for a 36-angle
    scan of 100 kpt single shots."""
    acc, t, pd_shots, cols_map = {}, None, [], None
    n = None
    for f in s["files"]:
        fp = os.path.join(folder, f)
        if not os.path.exists(fp):
            notes.append(f"missing {f}")
            continue
        columns, data = load_capture(fp)
        cols_map = _role_columns(columns, chans)
        data = data[trim:, :]
        if n is None:
            n, t = data.shape[0], data[:, 0].copy()
        if data.shape[0] < n:          # a short file: keep what all of them have
            n, t = data.shape[0], t[:data.shape[0]]
            for a in acc.values():
                a["sum"], a["sq"] = a["sum"][:n], a["sq"][:n]
                a["lo"], a["hi"] = a["lo"][:n], a["hi"][:n]
            pd_shots = [p[:n] for p in pd_shots]
        for role, j in cols_map.items():
            y = data[:n, j]
            a = acc.setdefault(role, {"sum": np.zeros(n), "sq": np.zeros(n),
                                      "lo": np.full(n, np.inf), "hi": np.full(n, -np.inf),
                                      "k": 0})
            a["sum"] += y
            a["sq"] += y * y
            np.minimum(a["lo"], y, out=a["lo"])
            np.maximum(a["hi"], y, out=a["hi"])
            a["k"] += 1
            if role == "PD":
                pd_shots.append(y.astype(np.float32))
    if t is None:
        return None
    return {"t": t, "acc": acc,
            "pd": np.array(pd_shots) if pd_shots else None}


def _mean_sem(a):
    k = a["k"]
    mean = a["sum"] / k
    if k < 2:
        return mean, np.full(len(mean), np.nan)
    var = np.maximum(a["sq"] - k * mean * mean, 0) / (k - 1)
    return mean, np.sqrt(var / k)


def load_scan(path, load_capture, trim=10, lock_tol=0.0, cache=None):
    """Read a scan's manifest and captures. `path` is the manifest or its
    folder. Each step gets 'v' {role: mean over its files (blocks or shots)},
    'sem' {role: standard error from their scatter}, 'nb' files used,
    'rejected' files dropped, and 'offscreen' {role: bool mask of samples
    outside the screen in any file}. The first `trim` samples are dropped (the
    MSO-X puts a fixed ~1.2 V artefact on sample 0).

    Steps are loaded as far as they got: a 'partial' step (stopped, or still
    being measured) contributes the shots it has, and is marked
    step['partial'] = True, so a scan can be shown while it runs or after it
    was stopped. `cache` (a dict kept by the caller) holds each
    step's reduced data keyed by its files' sizes and times, so reloading a
    growing scan reads only the new steps.

    lock_tol > 0 drops shots the intensity lock missed: a file whose PD level
    before the trigger (t < 0, else the first 4 % of the record) is off its
    step's median by more than lock_tol x the brightest such level in the
    scan. Relative to the brightest level, because at angles near crossed the
    level is ~0 and a relative test there would reject on noise. Steps with
    fewer than 4 files are left alone (no median to trust). Only the PD is
    re-averaged without them; the other channels keep every shot."""
    if os.path.isdir(path):
        cands = [f for f in os.listdir(path) if f.endswith("_scan.json")]
        if not cands:
            raise FileNotFoundError(f"no *_scan.json in {path}")
        path = os.path.join(path, cands[0])
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    folder = os.path.dirname(path)
    d = ScanData(man, folder)
    chans = man["channels"]
    d.roles = {v["role"]: int(k) for k, v in chans.items() if v["role"] != "off"}
    cache = {} if cache is None else cache
    loaded = []
    for s in man["steps"]:
        if s.get("status") not in ("done", "partial") or not s.get("files"):
            continue
        key = _step_key(folder, s, trim)
        red = cache.get(key)
        if red is None:
            red = _reduce_step(s, folder, chans, load_capture, trim, d.notes)
            if red is None:
                continue
            cache[key] = red
        loaded.append((s, red))
    if not loaded:
        raise ValueError("no completed steps with data in this scan yet")
    t_ref = loaded[0][1]["t"]
    d.t = t_ref
    pre = rest_index(t_ref)

    def on_ref(red, y):
        t = red["t"]
        if len(t) == len(t_ref) and np.allclose(t[[0, -1]], t_ref[[0, -1]]):
            return y
        return np.interp(t_ref, t, y)

    keep = {}
    if lock_tol > 0:
        levels = [np.median(red["pd"][:, pre[:red["pd"].shape[1]]].mean(axis=1))
                  for s, red in loaded if red["pd"] is not None and s["kind"] != "dark"]
        top = max([abs(x) for x in levels] or [0.0])
        dropped = 0
        for i, (s, red) in enumerate(loaded):
            sh = red["pd"]
            if sh is None or s["kind"] == "dark" or len(sh) < 4 or top <= 0:
                continue
            base = sh[:, pre[:sh.shape[1]]].mean(axis=1)
            ok = np.abs(base - np.median(base)) <= lock_tol * top
            if not ok.all() and ok.sum() >= 2:
                keep[i] = ok
                dropped += int((~ok).sum())
        if dropped:
            d.notes.append(f"dropped {dropped} shot(s) whose pre-trigger level was "
                           f"off their step's median by > {lock_tol:.1%} of the "
                           f"brightest level (intensity lock missed?)")
    for i, (s, red) in enumerate(loaded):
        step = dict(s)
        step["partial"] = s.get("status") == "partial"
        ok = keep.get(i)
        step["rejected"] = 0 if ok is None else int((~ok).sum())
        step["v"], step["sem"], step["offscreen"] = {}, {}, {}
        for role, a in red["acc"].items():
            if role == "PD" and ok is not None:
                y = red["pd"][ok].astype(np.float64)
                mean = y.mean(axis=0)
                sem = y.std(axis=0, ddof=1) / np.sqrt(len(y))
                lo, hi = y.min(axis=0), y.max(axis=0)
                step["nb"] = len(y)
            else:
                mean, sem = _mean_sem(a)
                lo, hi = a["lo"], a["hi"]
                if role == "PD" or "nb" not in step:
                    step["nb"] = a["k"]
            step["v"][role] = on_ref(red, mean)
            step["sem"][role] = on_ref(red, sem)
            sc = s.get("scales", {}).get(str(d.roles.get(role, "")))
            if sc and np.isfinite(sc[0]):
                vdiv, off = sc
                # 4 div either side of the offset is the screen; past it the
                # trace is outside the calibrated range and may be clipped
                lim = 4.0 * vdiv
                mask = (np.abs(on_ref(red, hi) - off) > lim) | (np.abs(on_ref(red, lo) - off) > lim)
                step["offscreen"][role] = mask
                step["offscreen_frac"] = step.get("offscreen_frac", {})
                step["offscreen_frac"][role] = float(np.mean(mask))
        d.steps.append(step)
    return d


# -- corrections ------------------------------------------------------------

def _pd(step, d, norm_ref=True):
    """PD trace of a step, divided by the reference PD's shape if one was
    recorded (normalised to its own mean, so the units stay volts)."""
    y = step["v"]["PD"]
    if norm_ref and "Ref" in step["v"]:
        r = step["v"]["Ref"]
        y = y / (r / np.mean(r))
    return y


def dark_level(d, vdiv=None):
    """Mean PD of the dark capture taken at (closest to) `vdiv`. 0 with a note
    if none was taken."""
    darks = [s for s in d.steps if s["kind"] == "dark" and "PD" in s["v"]]
    if not darks:
        return 0.0, None
    if vdiv is None:
        s = darks[0]
    else:
        def sc(s):
            v = s.get("scales", {}).get(str(d.roles.get("PD")), [np.nan])[0]
            return abs(np.log(v / vdiv)) if v and np.isfinite(v) else 99
        s = min(darks, key=sc)
    return float(np.mean(s["v"]["PD"])), s


def step_clock(s):
    """When a step's light was measured: the middle of its acquisition (a
    64-shot step lasts ~17 s, a real fraction of a drift's time scale)."""
    return 0.5 * (s["clock"] + s.get("clock_end", s["clock"]))


def drift(d, dark=0.0, window=None):
    """Intensity drift from the reference-angle returns: the mean PD of each
    ref capture over `window` (slice of samples, default the whole record),
    relative to their mean, against the step clock. Returns (clocks, levels,
    gain function of clock). Fewer than two refs -> gain 1."""
    refs = [s for s in d.steps if s["kind"] == "ref" and "PD" in s["v"]]
    sl = window if window is not None else slice(None)
    clocks = np.array([step_clock(s) for s in refs], float)
    levels = np.array([np.mean(_pd(s, d)[sl]) - dark for s in refs], float)
    if len(refs) < 2 or not np.all(levels > 0):
        return clocks, levels, (lambda c: np.ones_like(np.asarray(c, float)))
    rel = levels / levels.mean()
    order = np.argsort(clocks)
    cx, ry = clocks[order], rel[order]
    return clocks, levels, (lambda c: np.interp(c, cx, ry))


def drift_residual(clocks, levels):
    """How well the ref returns predict each other: leave each interior ref
    out, interpolate it from its neighbours, and return the rms relative
    miss (None with fewer than 3 refs). A gain error of this size between
    angles moves the fitted Imin by about that fraction of Imax, so ER_fit
    stops meaning anything above roughly 1 / this - the reason the dip and
    null-refine measurements exist."""
    if len(clocks) < 3:
        return None
    o = np.argsort(clocks)
    c, lv = np.asarray(clocks)[o], np.asarray(levels)[o]
    miss = [lv[i] / np.interp(c[i], np.r_[c[:i], c[i + 1:]], np.r_[lv[:i], lv[i + 1:]]) - 1
            for i in range(1, len(c) - 1)]
    return float(np.sqrt(np.mean(np.square(miss))))


def scan_matrix(d, kind="scan", correct_drift=True):
    """(theta_deg[K], I[K, N], sem_scalar[K], steps) for steps of `kind`, dark
    subtracted, drift corrected, reference normalised."""
    dark, _ = dark_level(d, _pd_vdiv(d, kind))
    _, _, gain = drift(d, dark)
    steps = [s for s in d.steps if s["kind"] == kind and "PD" in s["v"]
             and "landed" in s]
    if not steps:
        return np.zeros(0), np.zeros((0, len(d.t))), np.zeros(0), []
    th = np.array([s["landed"] for s in steps], float)
    g = gain(np.array([step_clock(s) for s in steps])) if correct_drift else np.ones(len(steps))
    I = np.array([(_pd(s, d) - dark) / gk for s, gk in zip(steps, g)])
    sem = np.array([np.nanmedian(s["sem"]["PD"]) for s in steps])
    return th, I, sem, steps


def _pd_vdiv(d, kind):
    ch = str(d.roles.get("PD"))
    for s in d.steps:
        if s["kind"] == kind:
            return s.get("scales", {}).get(ch, [None])[0]
    return None


# -- 1. the per-sample harmonic fit ----------------------------------------

def harmonic_fit(theta_deg, I, sem=None, diagnostics=None):
    """Fit I[k, n] = a0 + c2 cos2theta_k + s2 sin2theta_k (+ c1, s1, c4, s4)
    at every sample n. Weighted by 1/sem_k^2 when given (one weight per
    angle, from its block scatter). Diagnostic terms are included when the
    angles cover >= 300 deg with >= 9 of them, unless forced.

    Returns a dict of (N,) arrays: a0, B, psi (deg, in [-90, 90)), imax,
    imin, vis, er, er_lower (True where imin < 2 sigma and er is a lower
    bound), sig_psi (deg), sig_imin, rms (residual), and c1/s1/c4/s4 when
    fitted, plus 'n_angles'."""
    th = np.deg2rad(np.asarray(theta_deg, float))
    K = len(th)
    cover = np.ptp(np.unwrap(np.sort(th))) if K else 0
    if diagnostics is None:
        diagnostics = K >= 9 and np.rad2deg(cover) >= 300
    cols = [np.ones(K), np.cos(2 * th), np.sin(2 * th)]
    names = ["a0", "c2", "s2"]
    if diagnostics:
        cols += [np.cos(th), np.sin(th), np.cos(4 * th), np.sin(4 * th)]
        names += ["c1", "s1", "c4", "s4"]
    A = np.column_stack(cols)
    p = A.shape[1]
    if K < p:
        raise ValueError(f"{K} angles cannot fit {p} terms")
    sem_ok = sem is not None and np.all(np.isfinite(sem)) and np.all(np.asarray(sem) > 0)
    if sem_ok:
        w = 1.0 / np.asarray(sem, float) ** 2
        sigma0_sq = 1.0 / w.mean()        # the shot-scatter variance the weights stand for
        w = w / w.mean()
    else:
        w = np.ones(K)
    sw = np.sqrt(w)[:, None]
    Aw = A * sw
    coef, *_ = np.linalg.lstsq(Aw, I * sw, rcond=None)          # (p, N)
    resid = I - A @ coef
    dof = K - p
    if dof >= 2:
        s2 = np.sum(w[:, None] * resid ** 2, axis=0) / dof      # (N,) from the residual
        err_source = "residual"
    elif sem_ok:
        # As many angles as terms (3 angles, 5 Oct 2026 test-2): the fit passes
        # through every point, the residual is zero and says nothing. Fall back
        # on the shot-to-shot scatter - statistical error only, no model check.
        s2 = np.full(I.shape[1], sigma0_sq)
        err_source = "shot scatter (no residual)"
    else:
        s2 = np.full(I.shape[1], np.nan)
        err_source = "none"
    # how much of the 2-theta circle the angles cover: Malus is periodic in
    # 2 theta, and angles bunched in a quarter of it leave a0, B and psi
    # strongly correlated however many there are
    ang = np.sort(np.mod(2 * th, 2 * np.pi))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi]) if K else np.array([2 * np.pi])
    theta_span = float(np.rad2deg(2 * np.pi - gaps.max()) / 2)
    cov = np.linalg.inv(Aw.T @ Aw)                              # (p, p), unit variance
    out = {k: coef[i] for i, k in enumerate(names)}
    a0, c2, s2c = out["a0"], out["c2"], out["s2"]
    B = np.hypot(c2, s2c)
    psi = 0.5 * np.rad2deg(np.arctan2(s2c, c2))
    # propagate through the covariance with the gradient of each quantity:
    # dB = (c2 dc2 + s2 ds2)/B, dImin = da0 - dB
    with np.errstate(divide="ignore", invalid="ignore"):
        uc, us = np.where(B > 0, c2 / B, 0.0), np.where(B > 0, s2c / B, 0.0)
    var_b = s2 * (uc ** 2 * cov[1, 1] + us ** 2 * cov[2, 2] + 2 * uc * us * cov[1, 2])
    var_imin = s2 * cov[0, 0] + var_b - 2 * s2 * (uc * cov[0, 1] + us * cov[0, 2])
    # the azimuth error is the perpendicular component
    var_perp = s2 * (us ** 2 * cov[1, 1] + uc ** 2 * cov[2, 2] - 2 * uc * us * cov[1, 2])
    sig_b = np.sqrt(np.maximum(var_perp, 0))
    sig_imin = np.sqrt(np.maximum(var_imin, 0))
    imax, imin = a0 + B, a0 - B
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_psi = np.rad2deg(0.5 * sig_b / B)
        lower = imin < 2 * sig_imin
        er = np.where(lower, imax / (2 * sig_imin), imax / imin)
        vis = B / a0
    with np.errstate(divide="ignore", invalid="ignore"):
        mod_snr = float(np.nanmedian(B / np.sqrt(np.maximum(var_b, 1e-30))))
    out.update(B=B, psi=psi, imax=imax, imin=imin, vis=vis, er=er,
               er_lower=lower, sig_psi=sig_psi, sig_imin=sig_imin,
               rms=np.sqrt(s2), n_angles=K, mod_snr=mod_snr, dof=dof,
               err_source=err_source, theta_span=theta_span)
    return out


def unwrap_psi(psi_deg):
    """Continuous azimuth: psi is defined mod 180, unwrap it in time."""
    return 0.5 * np.rad2deg(np.unwrap(np.deg2rad(2 * np.asarray(psi_deg))))


def rest_index(t, frac=0.04):
    """Samples that count as 'before the ramp': t < 0 if the record has a
    pre-trigger part, else the first `frac` of it."""
    pre = t < 0
    if pre.sum() >= 10:
        return pre
    m = np.zeros(len(t), bool)
    m[:max(10, int(frac * len(t)))] = True
    return m


def polarization(d, correct_drift=True, diagnostics=None):
    """The full per-sample result for a scan: harmonic_fit plus psi_u
    (unwrapped azimuth), rotation (psi_u minus its rest value), t, and the
    drift record."""
    th, I, sem, steps = scan_matrix(d, "scan", correct_drift)
    if len(th) < 3:
        raise ValueError(f"{len(th)} analyzer angles measured - need at least 3")
    fit = harmonic_fit(th, I, sem, diagnostics)
    psi_u = unwrap_psi(fit["psi"])
    rest = rest_index(d.t)
    psi_rest = float(np.median(psi_u[rest]))
    # put the rest value on the branch nearest 0 so 'rotation' starts at 0
    shift = 180.0 * np.round(psi_rest / 180.0)
    psi_u = psi_u - shift
    psi_rest -= shift
    dark, _ = dark_level(d, _pd_vdiv(d, "scan"))
    clocks, levels, _ = drift(d, dark)
    fit.update(t=d.t, theta=th, I=I, steps=steps, psi_u=psi_u,
               psi_rest=psi_rest, rotation=psi_u - psi_rest,
               dark=dark, ref_clocks=clocks, ref_levels=levels,
               drift_resid=drift_residual(clocks, levels))
    return fit


# -- 2. extinction ratio from the dips ---------------------------------------

def _runs(mask):
    """(start, stop) index pairs of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(m))
    return list(zip(edges[::2], edges[1::2]))


def dip_er(pol, window_deg=8.0, min_samples=6, polarizer_er=None):
    """Extinction ratio at every time the polarization sweeps through a scan
    angle's crossed position. Returns a list of dicts, in time order:
    t (s), theta (analyzer deg), rotation (deg from rest at the crossing),
    rate (deg/us), imax, imin, sig_imin, er, er_lower, er_light (with the
    analyzer's own ER divided out, None at or past its limit), n."""
    t, psi = pol["t"], pol["psi_u"]
    imax_t = pol["imax"]
    rot = pol["rotation"]
    dt = np.gradient(t)
    rate = np.gradient(psi) / dt * 1e-6                 # deg/us
    out = []
    for th, I in zip(pol["theta"], pol["I"]):
        delta = (psi - th - 90.0 + 90.0) % 180.0 - 90.0
        for a, b in _runs(np.abs(delta) < window_deg):
            seg = slice(a, b)
            dl = delta[seg]
            if b - a < min_samples or dl.min() > 0 or dl.max() < 0:
                continue                                  # no crossing inside
            r = np.deg2rad(dl)
            im = imax_t[seg]
            M = np.column_stack([np.ones(b - a), im * np.sin(r) ** 2,
                                 im * np.sin(2 * r)])
            coef, *_ = np.linalg.lstsq(M, I[seg], rcond=None)
            c0, Ak, Ek = coef
            # A dip is only a dip if the light is modulated and the fitted
            # sin^2 has about the scale the per-sample fit predicts (Ak ~ 1).
            # Without these, noise crossings count: a no-light dry run on
            # 5 Oct 2026 reported 94 "dips" from 0.6 mV of noise.
            if not 0.5 < Ak < 2.0:
                continue
            j = a + int(np.argmin(np.abs(dl + np.rad2deg(Ek / Ak))))
            imax_c = float(imax_t[j])
            imin = c0 - imax_c * Ek ** 2 / Ak
            res = I[seg] - M @ coef
            dof = max(b - a - 3, 1)
            cov = np.linalg.pinv(M.T @ M) * (res @ res / dof)
            sig = float(np.sqrt(max(cov[0, 0], 0)))
            if imax_c < 20 * max(sig, float(np.median(pol["sig_imin"][seg]))):
                continue
            lower = imin < 2 * sig
            er = imax_c / (2 * sig) if lower else imax_c / imin
            er_light = None
            if polarizer_er and not lower and 1 / er > 1 / polarizer_er:
                er_light = 1 / (1 / er - 1 / polarizer_er)
            out.append({"t": float(t[j]), "theta": float(th),
                        "rotation": float(rot[j]), "rate": float(rate[j]),
                        "imax": imax_c, "imin": float(imin), "sig_imin": sig,
                        "er": float(er), "er_lower": bool(lower),
                        "er_light": er_light, "n": int(b - a)})
    out.sort(key=lambda r: r["t"])
    return out


# -- 3. null refinement ------------------------------------------------------

def parse_windows(text):
    """'1.0-2.0, 6.5-7.5' (ms) -> [(1e-3, 2e-3), ...]."""
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part or part.lower() == "auto":
            continue
        a, b = part.split("-", 1) if not part.startswith("-") else _neg_split(part)
        out.append((float(a) * 1e-3, float(b) * 1e-3))
    return out


def _neg_split(part):
    # a window that starts before the trigger: "-1.5-0.5" or "-1.5--0.5"
    head, rest = part[1:].split("-", 1)
    return "-" + head, rest


def segments(t, rotation, static_deg_per_ms=2.0, min_ms=0.3):
    """Split the record into static and moving parts by |d rotation/dt|.
    Returns [{'kind': rest|hold|after|up|down, 't0', 't1', 'rotation'}]."""
    # Smooth over ~0.1 ms BEFORE differentiating: the per-sample azimuth noise
    # divided by a sub-microsecond sample spacing is tens of deg/ms, which
    # would chop a hold into pieces. Edges are padded, not zero-filled.
    n = max(1, int(1e-4 / max(np.median(np.diff(t)), 1e-12)))
    box = np.ones(n) / n

    def smooth(y):
        if n <= 1:
            return y
        yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
        return np.convolve(yp, box, mode="same")[n:-n]

    rate = smooth(np.abs(np.gradient(smooth(rotation), t))) * 1e-3   # deg/ms
    static = rate < static_deg_per_ms
    out = []
    edges = np.flatnonzero(np.diff(static.astype(int))) + 1
    bounds = [0, *edges, len(t)]
    leg = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (t[b - 1] - t[a]) < min_ms * 1e-3:
            continue
        rmean = float(np.mean(rotation[a:b]))
        if static[a]:
            base = "hold" if abs(rmean) > 45 else ("rest" if leg == 0 else "after")
        else:
            # away from rest is "up" whichever way the crystals turn it: in
            # test-2's analyzer frame the ramp ran 0 -> -180 deg
            base = "up" if abs(rotation[b - 1]) > abs(rotation[a]) else "down"
            if base == "up" and not (out and out[-1]["base"] == "up"):
                leg += 1
        if out and out[-1]["base"] == base:
            out[-1]["t1"] = float(t[b - 1])
            continue
        out.append({"base": base, "leg": leg, "t0": float(t[a]),
                    "t1": float(t[b - 1]), "rotation": rmean})
    # "up 1", "hold 2", "after 1" ... - numbered only when a record holds more
    # than one transport (both spin-echo legs); the last static part is
    # "after", the ones between legs "after 1", ...
    for s in out:
        s["kind"] = s["base"] if leg <= 1 or s["base"] == "rest" else f"{s['base']} {s['leg']}"
    return out


def auto_windows(pol, margin_ms=0.2, max_ms=1.0):
    """Null-refine windows for the static segments, inset by `margin_ms` and
    at most `max_ms` long (the last part of each segment, where it has
    settled)."""
    wins = []
    for s in segments(pol["t"], pol["rotation"]):
        if s["base"] in ("rest", "hold", "after"):
            t1 = s["t1"] - margin_ms * 1e-3
            t0 = max(s["t0"] + margin_ms * 1e-3, t1 - max_ms * 1e-3)
            if t1 > t0:
                wins.append((t0, t1, s["kind"]))
    return wins


def null_targets(pol, windows, offsets):
    """Analyzer angles for refining each window: crossed position of the
    window's mean azimuth plus each offset (deg)."""
    plans = []
    for w, (t0, t1, *label) in enumerate(windows):
        m = (pol["t"] >= t0) & (pol["t"] <= t1)
        if not m.any():
            continue
        psi = float(np.mean(pol["psi_u"][m]))
        null = psi + 90.0
        plans.append({"window": w, "t0": t0, "t1": t1,
                      "label": label[0] if label else f"window {w + 1}",
                      "null": null, "angles": [null + o for o in offsets]})
    return plans


def refine_result(d, pol, polarizer_er=None):
    """Fit each refined window: mean PD in the window vs analyzer angle,
    I = a0 + c2 cos2theta + s2 sin2theta, Imin = a0 - B. Imax for the window
    comes from the scan's per-sample fit. Returns a list of dicts."""
    out = []
    nulls = [s for s in d.steps if s["kind"] == "null" and "PD" in s["v"]]
    if not nulls:
        return out
    vdiv = _pd_vdiv(d, "null")
    dark, dstep = dark_level(d, vdiv)
    _, _, gain = drift(d, dark_level(d, _pd_vdiv(d, "scan"))[0])
    by_w = {}
    for s in nulls:
        by_w.setdefault(s["window"], []).append(s)
    for w, ss in sorted(by_w.items()):
        t0, t1 = ss[0]["t0"], ss[0]["t1"]
        m = (d.t >= t0) & (d.t <= t1)
        th = np.array([s["landed"] for s in ss])
        g = gain(np.array([step_clock(s) for s in ss]))
        I = np.array([(np.mean(_pd(s, d)[m]) - dark) / gk for s, gk in zip(ss, g)])
        sem = np.array([np.mean(s["sem"]["PD"][m]) / np.sqrt(max(m.sum(), 1))
                        for s in ss])
        rec = {"window": w, "label": ss[0].get("label", ""), "t0": t0, "t1": t1,
               "theta": th, "I": I, "dark": dark,
               "dark_vdiv_matched": dstep is not None,
               "offscreen": any(np.any(s["offscreen"].get("PD", np.zeros(1))[m])
                                for s in ss)}
        if len(th) >= 3:
            th_r = np.deg2rad(th)
            A = np.column_stack([np.ones_like(th_r), np.cos(2 * th_r), np.sin(2 * th_r)])
            coef, *_ = np.linalg.lstsq(A, I, rcond=None)
            a0, c2, s2 = coef
            B = np.hypot(c2, s2)
            imin = a0 - B
            res = I - A @ coef
            dof = max(len(th) - 3, 1)
            cov = np.linalg.pinv(A.T @ A) * max(res @ res / dof, np.mean(sem ** 2))
            # Imin = a0 - B through the full covariance: over a few degrees
            # around the null a0, c2 and s2 are strongly correlated, and
            # adding their variances as if independent inflates sigma ~10x
            g = np.array([1.0, -c2 / B, -s2 / B]) if B > 0 else np.array([1.0, 0, 0])
            sig = float(np.sqrt(max(g @ cov @ g, 0.0)))
            imax = float(np.mean(pol["imax"][m]))
            lower = imin < 2 * sig
            er = imax / (2 * sig) if lower else imax / imin
            er_light = None
            if polarizer_er and not lower and 1 / er > 1 / polarizer_er:
                er_light = 1 / (1 / er - 1 / polarizer_er)
            theta_null = (0.5 * np.rad2deg(np.arctan2(s2, c2)) + 90.0)
            rec.update(imax=imax, imin=float(imin), sig_imin=sig, er=float(er),
                       er_lower=bool(lower), er_light=er_light,
                       theta_null=float(theta_null),
                       model=(float(a0), float(c2), float(s2)))
        out.append(rec)
    return out


# -- monitors ----------------------------------------------------------------

def monitor_prediction(d, pol, deg_per_mon_v):
    """Rotation predicted from the Trek monitors, sum_i k_i V_i(t) averaged
    over the scan steps, with its sign and offset matched to the measured
    rotation by least squares (the light fixes neither). Returns
    (pred_rotation, residual, sign) or None without monitors."""
    roles = [r for r in ("MonX1", "MonX2") if r in d.roles]
    if not roles:
        return None
    steps = pol["steps"]
    pred = np.zeros(len(d.t))
    for r in roles:
        k = float(deg_per_mon_v.get(r, 0.0))
        pred += k * np.mean([s["v"][r] for s in steps if r in s["v"]], axis=0)
    best = None
    for sign in (1.0, -1.0):
        p = sign * pred
        c = float(np.mean(pol["rotation"] - p))
        res = pol["rotation"] - (p + c)
        cost = float(np.mean(res ** 2))
        if best is None or cost < best[0]:
            best = (cost, p + c, res, sign)
    return best[1], best[2], best[3]


def segment_table(pol, dips=(), refine=()):
    """Rows for the Table tab: one per segment, with the median fitted values
    and any direct ER measured inside it."""
    rows = []
    for s in segments(pol["t"], pol["rotation"]):
        m = (pol["t"] >= s["t0"]) & (pol["t"] <= s["t1"])
        er_fit = pol["er"][m]
        dd = [x for x in dips if s["t0"] <= x["t"] <= s["t1"]]
        rr = [x for x in refine if s["t0"] <= 0.5 * (x["t0"] + x["t1"]) <= s["t1"]
              and "er" in x]
        rows.append({
            "segment": s["kind"], "t0_ms": s["t0"] * 1e3, "t1_ms": s["t1"] * 1e3,
            "rotation_deg": float(np.median(pol["rotation"][m])),
            "sig_psi_deg": float(np.median(pol["sig_psi"][m])),
            "visibility": float(np.median(pol["vis"][m])),
            "er_fit_median": float(np.median(er_fit)),
            "er_dips": [x["er"] for x in dd],
            "er_refine": [x["er"] for x in rr],
        })
    return rows

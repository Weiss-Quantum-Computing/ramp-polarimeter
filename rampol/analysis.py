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
        # what the analysis subtracts from the PD (see pd_offset): turn it off
        # here to look at the light as the scope read it
        self.subtract_dark = True


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
                  for s, red in loaded if red["pd"] is not None
                  and s["kind"] not in OFFSET_KINDS]
        top = max([abs(x) for x in levels] or [0.0])
        dropped = 0
        for i, (s, red) in enumerate(loaded):
            sh = red["pd"]
            if sh is None or s["kind"] in OFFSET_KINDS or len(sh) < 4 or top <= 0:
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
        # which shots (files, in order) the PD average kept: the shot view
        # draws the dropped ones differently
        step["kept"] = None if ok is None else [bool(x) for x in ok]
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


# Steps that measure what the PD reads without the experiment's light:
#   dark        the photodiode covered - no light at all: the PD's own offset
#               plus the scope's offset error at that V/div and offset (-34 mV
#               on 5 Oct, all of it the scope's at a 2.65 V offset)
#   background  the laser beam blocked, the room as during the scan: the dark
#               plus whatever stray light reaches the PD
# What is subtracted is the background when there is one (it contains the
# dark), else the dark. Scans before 6 Oct 2026 have only 'dark' steps, and
# those were taken with the beam blocked before the analyzer - so they were
# backgrounds in this sense; the numbers subtracted are the same either way.
OFFSET_KINDS = ("dark", "background")
OFFSET_WHAT = {"dark": "PD covered (no light)",
               "background": "beam blocked (stray light included)"}


def _scale_of(d, s):
    return s.get("scales", {}).get(str(d.roles.get("PD")), [np.nan, np.nan])


def offset_levels(d, vdiv=None):
    """{'dark': info, 'background': info} for what this scan has - its own
    steps first (the one taken at the V/div closest to `vdiv`), else what its
    manifest borrowed from another scan. info: level (V), sem (V), n (shots),
    vdiv, offset, source ('this scan' or the lending scan's name), measured
    (time), what. Missing kinds are absent."""
    out = {}
    for kind in OFFSET_KINDS:
        own = [s for s in d.steps if s["kind"] == kind and "PD" in s["v"]]
        if own:
            if vdiv is None:
                s = own[0]
            else:
                def sc(s):
                    v = _scale_of(d, s)[0]
                    return abs(np.log(v / vdiv)) if v and np.isfinite(v) else 99
                s = min(own, key=sc)
            y = s["v"]["PD"]
            v, off = _scale_of(d, s)
            what = OFFSET_WHAT[kind]
            if kind == "dark" and d.manifest.get("format") == "rampol-scan/1" and \
                    not d.manifest.get("offsets_v2"):
                what = "beam blocked before the analyzer (scan before 6 Oct 2026)"
            out[kind] = {"level": float(np.mean(y)),
                         "sem": float(np.mean(s["sem"]["PD"]) / np.sqrt(len(y) / 20)),
                         "n": int(s.get("nb", 0)), "vdiv": v, "offset": off,
                         "source": "this scan", "measured": s.get("t_start", ""),
                         "what": what, "step": s}
            continue
        b = (d.manifest.get("borrowed") or {}).get(kind)
        if b:
            out[kind] = dict(b, what=OFFSET_WHAT[kind], step=None)
    return out


def _step_level(d, s):
    y = s["v"]["PD"]
    return float(np.mean(y)), float(np.mean(s["sem"]["PD"]) / np.sqrt(len(y) / 20))


def stray_light(d):
    """Stray light on the PD = background - dark, from a pair taken at the
    same V/div and offset - the finest such pair of this scan's own steps,
    else what its manifest borrowed ('stray'). At 1 V/div each of the two is
    +-1.2-1.5 mV (7 Oct 2026) and their difference says nothing; at a few
    mV/div it is known to ~0.05 mV, and the scope's offset error, the same
    in both, cancels. Returns info (level, sem, vdiv, offset, source,
    measured, dark, background) or None."""
    pairs = {}
    for s in d.steps:
        if s["kind"] in OFFSET_KINDS and "PD" in s.get("v", {}):
            v, off = _scale_of(d, s)[:2]
            if np.isfinite(v):
                pairs.setdefault((round(float(v), 9), round(float(off), 5)), {})[s["kind"]] = s
    full = [(v, off, p) for (v, off), p in pairs.items() if len(p) == 2]
    if full:
        v, off, p = min(full, key=lambda x: x[0])
        lb, sb = _step_level(d, p["background"])
        ld, sd = _step_level(d, p["dark"])
        return {"level": lb - ld, "sem": float(np.hypot(sb, sd)), "vdiv": v, "offset": off,
                "source": "this scan", "measured": p["background"].get("t_start", ""),
                "dark": ld, "background": lb}
    b = (d.manifest.get("borrowed") or {}).get("stray")
    return dict(b) if b else None


def dark_level(d, vdiv=None):
    """What is subtracted from the PD: (level V, info).

    With a stray-light pair at a V/div at least 4x finer than `vdiv`
    (stray_light): the scope's offset at `vdiv` - from the dark there and
    from the background there minus the stray light, inverse-variance
    weighted when both exist - plus the stray light. Otherwise the
    background if there is one, else the dark, own steps before borrowed
    ones. (0.0, None) when there is nothing or d.subtract_dark is off."""
    if not getattr(d, "subtract_dark", True):
        return 0.0, None
    lv = offset_levels(d, vdiv)
    st = stray_light(d)
    if st and vdiv and np.isfinite(st.get("vdiv", np.nan)) and st["vdiv"] * 4 <= vdiv:
        ests = []
        for k in ("dark", "background"):
            e = lv.get(k)
            v = e and e.get("vdiv")
            if not e or not v or not np.isfinite(v) or abs(np.log(v / vdiv)) > 1e-6:
                continue
            ests.append((e["level"] - (st["level"] if k == "background" else 0.0),
                         e.get("sem") or 0.0, k, e))
        if ests:
            sems = np.array([x[1] for x in ests])
            w = 1 / sems ** 2 if np.all(sems > 0) else np.ones(len(ests))
            O = float(np.sum(w * [x[0] for x in ests]) / np.sum(w))
            sO = float(1 / np.sqrt(np.sum(w))) if np.all(sems > 0) else float(np.max(sems))
            srcs = sorted({x[3]["source"] for x in ests} | {st["source"]})
            info = {"kind": "dark + stray light", "level": O + st["level"],
                    "sem": float(np.hypot(sO, st["sem"])), "offset_est": O, "offset_sem": sO,
                    "from": [x[2] for x in ests], "stray": st, "vdiv": vdiv,
                    "offset": ests[0][3].get("offset"),
                    "source": "this scan" if srcs == ["this scan"] else " + ".join(srcs),
                    "what": "the scope / PD offset at this V/div plus the stray light "
                            "read at a fine V/div"}
            return float(info["level"]), info

    def dist(info):
        # a measurement at another V/div is the wrong one (the scope's offset
        # error moves with it): the V/div match comes before the kind
        v = info.get("vdiv")
        if vdiv is None or not v or not np.isfinite(v):
            return 0.0
        return round(abs(np.log(v / vdiv)), 6)
    # then a background before a dark (it includes the stray light the dark
    # misses), whether measured here or reused from another scan - reusing
    # one was asked for (offset_levels already prefers this scan's own of
    # each kind)
    cands = [(dist(lv[k]), r, k) for r, k in enumerate(("background", "dark")) if k in lv]
    if cands:
        _, _, kind = min(cands)
        info = dict(lv[kind], kind=kind)
        return float(info["level"]), info
    return 0.0, None


OFFSET_AGE_WARN_H = 2.0


def offset_age_h(d, info):
    """Hours between a borrowed dark / background's measurement and this
    scan's first capture (None for the scan's own, or without times)."""
    if not info or info.get("source") in (None, "this scan"):
        return None
    import datetime
    try:
        when = datetime.datetime.fromisoformat(str(info.get("measured", "")))
        start = datetime.datetime.fromisoformat(str(d.manifest.get("created", "")))
    except ValueError:
        return None
    return (start - when).total_seconds() / 3600.0


def corrections_summary(d, pol=None):
    """Every correction the analysis applies to this scan, with its size:
    dict(dark, background, light, subtracted, drift, gains, dropped,
    offscreen, text). 'text' is the one line the window shows."""
    vdiv = _pd_vdiv(d, "scan")
    lv = offset_levels(d, vdiv)
    sub, info = dark_level(d, vdiv)
    out = {"dark": lv.get("dark"), "background": lv.get("background"),
           "subtracted": sub, "subtracted_kind": info and info["kind"],
           "subtracted_source": info and info["source"]}
    parts = []
    if not getattr(d, "subtract_dark", True):
        parts.append("NOTHING subtracted (off)")
    elif info is None:
        parts.append("no dark or background: 0 V subtracted")
    else:
        src = "" if info["source"] == "this scan" else f" from {info['source']}"
        age = offset_age_h(d, info)
        if age is not None and age > OFFSET_AGE_WARN_H:
            # 7 Oct 2026: a background from 11:24 under scans at 19:45; the
            # dark had moved 2 mV by 18:30 - the whole rest ER floor
            out["offset_age_h"] = age
            src += f" ({age:.1f} h before this scan: measure a fresh one)"
        old = " (beam blocked: a scan before 6 Oct 2026)" if "before 6 Oct" in info["what"] else ""
        if info["kind"] == "dark + stray light":
            st = info["stray"]
            out["light"], out["light_sem"] = st["level"], st["sem"]
            parts.append(
                f"subtract {sub*1e3:+.2f} +- {info['sem']*1e3:.2f} mV{src} = offset at "
                f"{info['vdiv']:g} V/div {info['offset_est']*1e3:+.2f} +- "
                f"{info['offset_sem']*1e3:.2f} (from the {' and '.join(info['from'])}) + stray "
                f"light {st['level']*1e3:+.3f} +- {st['sem']*1e3:.3f} mV (read at "
                f"{st['vdiv']*1e3:g} mV/div)")
        else:
            parts.append(f"subtract {info['kind']}{old} {sub*1e3:+.2f} mV{src}")
    if "dark" in lv and "background" in lv and not (info and info["kind"] == "dark + stray light"):
        out["light"] = lv["background"]["level"] - lv["dark"]["level"]
        # with its error: at 1 V/div the two are each ~1.3 mV uncertain, and
        # a background below the dark (7 Oct 2026: -0.83 mV) is that noise
        s_l = float(np.hypot(lv["background"].get("sem") or 0, lv["dark"].get("sem") or 0))
        out["light_sem"] = s_l
        zero = s_l > 0 and abs(out["light"]) < 2 * s_l
        parts.append(f"(dark {lv['dark']['level']*1e3:+.2f} + stray light "
                     f"{out['light']*1e3:+.2f} +- {s_l*1e3:.2f} mV"
                     + (": zero within its error" if zero else "") + ")")
    clocks, levels, _ = drift(d, sub)
    if len(levels) >= 2 and np.all(levels > 0):
        rel = levels / levels.mean() - 1
        out["drift"] = {"n": len(levels), "pp": float(np.ptp(rel)),
                        "resid": drift_residual(clocks, levels)}
        parts.append(f"drift {np.ptp(rel)*100:.2f} % p-p over {len(levels)} refs")
    g = pol.get("angle_gain") if pol else None
    if g is not None:
        out["gains"] = {"min": float(g.min()), "max": float(g.max())}
        src = pol.get("gain_from")
        parts.append(f"angle gains {(g.min()-1)*100:+.1f}..{(g.max()-1)*100:+.1f} %"
                     + (f" (from {src})" if src else ""))
    elif pol and pol.get("gain_note"):
        parts.append("no angle gains: " + pol["gain_note"])
    drop = sum(s.get("rejected", 0) for s in d.steps)
    tot = sum(s.get("nb", 0) + s.get("rejected", 0) for s in d.steps
              if s["kind"] not in OFFSET_KINDS)
    out["dropped"] = (drop, tot)
    if drop:
        parts.append(f"{drop} of {tot} shots dropped (lock)")
    off = [s for s in d.steps if any(np.any(v) for v in s.get("offscreen", {}).values())]
    out["offscreen"] = len(off)
    if off:
        parts.append(f"{len(off)} steps with samples off screen")
    out["text"] = "; ".join(parts)
    return out


def step_shots(d, step, load_capture, trim=10):
    """Every shot of a step straight from its files, nothing averaged:
    (t, {role: array (shots, samples)}, files). t on the file's own grid."""
    chans = d.manifest["channels"]
    t, out, files = None, {}, []
    for f in step.get("files", []):
        fp = os.path.join(d.folder, f)
        if not os.path.exists(fp):
            continue
        columns, data = load_capture(fp)
        data = data[trim:, :]
        cm = _role_columns(columns, chans)
        if t is None:
            t = data[:, 0].copy()
        n = min(len(t), data.shape[0])
        t = t[:n]
        for role, j in cm.items():
            out.setdefault(role, []).append(data[:n, j])
        files.append(f)
    return t, {r: np.array([y[:len(t)] for y in v]) for r, v in out.items()}, files


def find_stray(outdir, load_capture, finer_than, exclude=None):
    """The newest stray-light pair (dark and background at one V/div and
    offset, the V/div at least 4x finer than `finer_than`) among the scans
    in outdir: an info dict for a manifest's borrowed['stray'], or None."""
    best = None
    try:
        names = os.listdir(outdir)
    except OSError:
        return None
    for name in names:
        folder = os.path.join(outdir, name)
        mp = os.path.join(folder, f"{name}_scan.json")
        if not os.path.isfile(mp) or (exclude and os.path.normcase(folder) ==
                                      os.path.normcase(exclude)):
            continue
        try:
            with open(mp, encoding="utf-8") as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            continue
        if not man.get("offsets_v2"):
            continue                       # before 6 Oct 2026 'dark' was beam-blocked
        pd_ch = str(next((k for k, v in man.get("channels", {}).items()
                          if v.get("role") == "PD"), ""))
        pairs = {}
        for s in man.get("steps", []):
            if s.get("kind") in OFFSET_KINDS and s.get("status") == "done" and s.get("files"):
                v, off = (s.get("scales", {}).get(pd_ch) or [np.nan, np.nan])[:2]
                if np.isfinite(v) and v * 4 <= finer_than:
                    pairs.setdefault((round(v, 9), round(off, 5)), {})[s["kind"]] = s
        for (v, off), p in pairs.items():
            if len(p) < 2:
                continue
            lev = {}
            for k, s in p.items():
                ys = []
                for f in s["files"]:
                    try:
                        cols, data = load_capture(os.path.join(folder, f))
                    except Exception:
                        continue
                    j = next((i for i, c in enumerate(cols)
                              if c.startswith(f"CH{pd_ch}_") or c == f"CH{pd_ch}_V"), None)
                    if j is not None:
                        ys.append(float(np.mean(data[10:, j])))
                if not ys:
                    break
                lev[k] = (float(np.mean(ys)), float(np.std(ys, ddof=1) / np.sqrt(len(ys)))
                          if len(ys) > 1 else 0.0)
            if len(lev) < 2:
                continue
            when = p["background"].get("t_start", man.get("created", ""))
            if best is None or when > best["measured"]:
                best = {"level": lev["background"][0] - lev["dark"][0],
                        "sem": float(np.hypot(lev["background"][1], lev["dark"][1])),
                        "vdiv": float(v), "offset": float(off),
                        "source": man.get("name", name), "measured": when,
                        "dark": lev["dark"][0], "background": lev["background"][0]}
    return best


def find_offsets(outdir, vdiv, offset, load_capture, kinds=OFFSET_KINDS,
                 exclude=None, tol_v=1e-3):
    """Dark / background measurements in other scans under `outdir` taken
    at this PD V/div and offset (offset within tol_v - the scope's offset
    error moves with the offset, -34 mV at +2.65 V on 5 Oct), newest first:
    [{kind, level, sem, n, vdiv, offset, source, folder, measured}]."""
    found = []
    try:
        names = os.listdir(outdir)
    except OSError:
        return found
    for name in names:
        folder = os.path.join(outdir, name)
        mp = os.path.join(folder, f"{name}_scan.json")
        if not os.path.isfile(mp) or (exclude and os.path.normcase(folder) ==
                                      os.path.normcase(exclude)):
            continue
        try:
            with open(mp, encoding="utf-8") as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            continue
        pd_ch = next((k for k, v in man.get("channels", {}).items()
                      if v.get("role") == "PD"), None)
        for s in man.get("steps", []):
            if s.get("kind") not in kinds or s.get("status") != "done" or not s.get("files"):
                continue
            v, off = (s.get("scales", {}).get(str(pd_ch)) or [np.nan, np.nan])[:2]
            if not (np.isfinite(v) and abs(v - vdiv) <= 1e-6 * max(vdiv, 1)
                    and abs(off - offset) <= tol_v):
                continue
            ys = []
            for f in s["files"]:
                try:
                    cols, data = load_capture(os.path.join(folder, f))
                except Exception:
                    continue
                j = next((i for i, c in enumerate(cols)
                          if c.startswith(f"CH{pd_ch}_") or c == f"CH{pd_ch}_V"), None)
                if j is not None:
                    ys.append(float(np.mean(data[10:, j])))
            if not ys:
                continue
            kind = s["kind"]
            what_old = kind == "dark" and not man.get("offsets_v2")
            found.append({"kind": "background" if what_old else kind,
                          "level": float(np.mean(ys)),
                          "sem": float(np.std(ys, ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0,
                          "n": len(ys), "vdiv": float(v), "offset": float(off),
                          "source": man.get("name", name), "folder": folder,
                          "measured": s.get("t_start", man.get("created", ""))})
    found.sort(key=lambda x: x["measured"], reverse=True)
    return found


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


def angle_gains(theta_deg, I, iters=30):
    """Per-angle transmission factors g_k, fitted jointly with the Malus law:
    I_k(t) = g_k (a0(t) + c2(t) cos 2theta_k + s2(t) sin 2theta_k), mean g = 1.

    MEASURED 5 Oct 2026 (16-ms-spin-echo-test-4, 19 angles over 0-180): the
    analyzer's throughput to the PD depends on the mount angle - +2 % from 0
    to 180 deg, mostly a 1-theta term (1.75 %), i.e. the beam walking on the
    detector as the polarizer turns. Left in, it raised the fit residual from
    1.1 mV (shot noise) to 12.8 mV and tilted psi and Imin during the ramps;
    with it fitted the residual is 1.65 mV. The factors are identifiable
    because the ramp sweeps the polarization through 180 deg; the overall
    scale is not (mean fixed at 1). Also absorbs any intensity drift between
    angles. Returns g (K,)."""
    r = np.deg2rad(np.asarray(theta_deg, float))
    A = np.column_stack([np.ones_like(r), np.cos(2 * r), np.sin(2 * r)])
    g = np.ones(len(r))
    # the same 3-column design every pass: its pseudo-inverse once (lstsq on
    # 100k right-hand sides x 30 passes was 2.8 s of a 3.9 s reanalysis)
    P = np.linalg.pinv(A)
    for _ in range(iters):
        coef = P @ (I / g[:, None])
        M = A @ coef
        g_new = np.sum(I * M, axis=1) / np.maximum(np.sum(M * M, axis=1), 1e-30)
        g_new /= g_new.mean()
        if np.max(np.abs(g_new - g)) < 1e-7:
            g = g_new
            break
        g = g_new
    return g


# The per-angle factors are identifiable only while the polarization sweeps:
# with it standing still they trade off against a0 / c2 / s2 (7 Oct 2026, an
# X1 0 deg ramp: g = -0.83 .. 1.57, Imax 3.8 V where 5.9 V was measured; the
# 22.5 .. 90 deg ramps of the same sequence agreed on the 5 Oct pattern).
GAIN_SWEEP_DEG = 20.0
GAIN_RANGE = (0.8, 1.25)


def azimuth_sweep(th, I, every=50):
    """How far (deg) the polarization azimuth moves over the record, from a
    fit on every `every`-th sample (cheap)."""
    f = harmonic_fit(th, I[:, ::max(1, int(every))], None)
    ok = f["B"] > 0.05 * np.maximum(f["a0"], 1e-12)
    if ok.sum() < 3:
        return 0.0
    return float(np.ptp(unwrap_psi(f["psi"][ok])))


def gains_usable(th, I):
    """(g or None, why): the per-angle factors when the record identifies
    them - the azimuth sweeps >= GAIN_SWEEP_DEG and every factor lands in
    GAIN_RANGE - else None and the reason."""
    sweep = azimuth_sweep(th, I)
    if sweep < GAIN_SWEEP_DEG:
        return None, (f"azimuth sweeps {sweep:.1f} deg (< {GAIN_SWEEP_DEG:g}): the per-angle "
                      f"factors are not identifiable")
    g = angle_gains(th, I)
    if g.min() < GAIN_RANGE[0] or g.max() > GAIN_RANGE[1]:
        return None, (f"fitted per-angle factors {g.min():.2f}..{g.max():.2f} are outside "
                      f"{GAIN_RANGE[0]:g}..{GAIN_RANGE[1]:g}: not used")
    return g, ""


def polarization(d, correct_drift=True, diagnostics=None, angle_gain=None):
    """The full per-sample result for a scan: harmonic_fit plus psi_u
    (unwrapped azimuth), rotation (psi_u minus its rest value), t, and the
    drift record.

    angle_gain: fit a transmission factor per analyzer angle (angle_gains) and
    divide it out first. Default (None): when there are >= 8 angles covering
    >= 150 deg (with fewer the factors trade off against the polarization)
    and gains_usable says the record identifies them; pol['gain_note'] says
    why not. An array: those factors (one per angle, e.g. a sequence
    sibling's), used as given. False: none."""
    th, I, sem, steps = scan_matrix(d, "scan", correct_drift)
    if len(th) < 3:
        raise ValueError(f"{len(th)} analyzer angles measured - need at least 3")
    gains, note = None, ""
    if isinstance(angle_gain, np.ndarray):
        if len(angle_gain) != len(th):
            raise ValueError(f"{len(angle_gain)} angle gains for {len(th)} angles")
        gains = np.asarray(angle_gain, float)
    elif angle_gain is None:
        span = harmonic_fit(th, I[:, :2], None)["theta_span"] if len(th) >= 3 else 0
        if len(th) >= 8 and span >= 150:
            gains, note = gains_usable(th, I)
    elif angle_gain:
        gains = angle_gains(th, I)
    if gains is not None:
        I = I / gains[:, None]
        sem = sem / gains
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
               drift_resid=drift_residual(clocks, levels), angle_gain=gains,
               gain_note=note)
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


# -- 2b. extinction ratio measured directly, no fit in the values -------------

def direct_er(d, pol, box_us=4.0, correct_drift=False, gains=None):
    """Extinction ratios with both intensities MEASURED, at every time the ramp
    sweeps the light through crossed for one of the scan's analyzer angles:
    Imin = that angle's trace at its minimum (`box_us` boxcar), Imax = the
    trace of the angle 90 deg away at the same instant. The fit's azimuth
    only says WHEN a crossing happens. Static stretches (rest, holds, between
    legs): the angle nearest crossed against the one 90 deg from it, averaged
    over the stretch, with how far from crossed that angle sat (Imax sin^2 of
    it is the Imin that offset alone gives).

    Defaults reproduce tools/direct_er.py as first run (5 Oct 2026): raw
    traces, no drift correction, no per-angle transmission. gains: the
    per-angle factors to divide out (pol['angle_gain']), or None. Returns
    dicts: kind (crossing | static), seg, t_ms, theta, rotation (pol's sense),
    rate (deg/ms), imin_mV, sig_mV, imax_V, er, lower (Imin < 2 sigma: er is
    Imax / 2 sigma), and for static ones off_deg, imin_from_offset_mV and
    offset_limited (the offset alone gives over half the measured Imin: with
    coarse angle steps the nearest angle sits degrees from crossed, and the
    point says how far the angle was, not what the light's ER is)."""
    t = d.t
    dt = float(np.median(np.diff(t)))
    th, I, _sem, steps = scan_matrix(d, "scan", correct_drift=correct_drift)
    if len(th) < 2:
        return []
    sem = np.array([np.asarray(s["sem"]["PD"], float) for s in steps])
    if gains is not None and len(gains) == len(th):
        I = I / np.asarray(gains)[:, None]
        sem = sem / np.asarray(gains)[:, None]
    nb = max(1, int(round(box_us * 1e-6 / dt)))
    box = np.ones(nb) / nb
    Is = np.array([np.convolve(x, box, mode="same") for x in I])
    sems = sem / np.sqrt(nb)
    psi = pol["psi_u"]
    rot = pol["rotation"]
    rate_all = np.abs(np.gradient(rot, t)) * 1e-3          # deg/ms

    def wrap(x):
        return (x + 90.0) % 180.0 - 90.0
    orth = [[j for j in range(len(th)) if abs(wrap(th[j] - th[k] - 90)) < 1.0]
            for k in range(len(th))]
    segs = segments(t, rot)
    moving = np.zeros(len(t), bool)
    for s in segs:
        if s["base"] in ("up", "down"):
            moving |= (t >= s["t0"]) & (t <= s["t1"])
    downs = [s["t1"] for s in segs if s["base"] == "down"]
    # the intensity lock switches off after the last leg: nothing past it
    t_end = (max(downs) + 1e-3) if downs else t[-1]

    def seg_of(tm):
        return next((s["kind"] for s in segs if s["t0"] <= tm <= s["t1"]), "")
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
            imin, s_min = float(Is[k, m]), float(sems[k, m])
            imax = float(np.mean([Is[j, m] for j in orth[k]]))
            s_max = float(np.sqrt(np.mean([sems[j, m] ** 2 for j in orth[k]]) / len(orth[k])))
            if not np.isfinite(s_min) and imin <= 0:
                # no error estimate (a step stopped after one shot) and a
                # level at or below zero: neither a value nor a bound
                # (7 Oct 2026: it went into the lab log as ER -2366)
                continue
            lower = bool(imin < 2 * s_min)
            pts.append(dict(kind="crossing", seg=seg_of(t[m]), t_ms=float(t[m] * 1e3),
                            theta=float(th[k]), rotation=float(rot[m]),
                            rate=float(rate_all[m]), imin_mV=imin * 1e3, sig_mV=s_min * 1e3,
                            imax_V=imax, sig_imax_V=s_max,
                            er=imax / (2 * s_min) if lower else imax / imin,
                            lower=lower))
    # one crossing can show up as several sign flips of the azimuth where it
    # creeps through crossed slowly (noise): keep one per angle per 0.1 ms
    kept = []
    for p in sorted(pts, key=lambda p: (p["theta"], p["t_ms"])):
        if kept and abs(kept[-1]["theta"] - p["theta"]) < 1e-6 and \
                p["t_ms"] - kept[-1]["t_ms"] < 0.1:
            continue
        kept.append(p)
    pts = kept
    for s in segs:
        if s["base"] in ("up", "down") or s["t0"] > t_end:
            continue
        w = (t > s["t0"] + 0.2e-3) & (t < min(s["t1"], t_end) - 0.2e-3)
        if w.sum() < 100:
            continue
        means = I[:, w].mean(axis=1)
        k = int(np.argmin(means))
        if not orth[k]:
            continue
        imax = float(np.mean([means[j] for j in orth[k]]))
        n_ind = w.sum() / max(1, int(1e-6 / dt))
        s_min = float(np.mean(sem[k, w]) / np.sqrt(n_ind))
        s_max = float(np.mean([np.mean(sem[j, w]) for j in orth[k]]) / np.sqrt(n_ind * len(orth[k])))
        off = float(np.median(wrap(psi[w] - th[k] - 90)))
        imin = float(means[k])
        if not np.isfinite(s_min) and imin <= 0:
            continue                  # as for a crossing: no value, no bound
        from_off = float(imax * np.sin(np.deg2rad(off)) ** 2)
        # The angle sat `off` deg from crossed, and Imax sin^2(off) of what it
        # read is that offset, not the light: the light's Imin is at most the
        # reading, so once the offset's share is above the noise the point is a
        # LOWER bound on the ER (7 Oct 2026: the 22.5 deg grid left the holds
        # of the 15 / 30 / 60 / 75 deg ramps 6-9 deg from crossed, and
        # "ER 42" was read as the light's). offset_limited marks the bounds
        # the offset makes useless (over half of the reading).
        noise_bound = bool(imin < 2 * s_min)
        lower = noise_bound or from_off > s_min
        er = imax / (2 * s_min) if noise_bound else imax / imin
        pts.append(dict(kind="static", seg=s["kind"], t_ms=float(t[w].mean() * 1e3),
                        theta=float(th[k]), rotation=float(np.median(rot[w])), rate=0.0,
                        imin_mV=imin * 1e3, sig_mV=s_min * 1e3, imax_V=imax,
                        sig_imax_V=s_max, er=er, lower=bool(lower),
                        bound_from="noise" if noise_bound else ("offset" if lower else ""),
                        off_deg=off, imin_from_offset_mV=from_off * 1e3,
                        crossed_deg=float(np.mod(th[k] + off, 180.0)),
                        offset_limited=bool(from_off > 0.5 * max(imin, 0.0))))
    return pts


def malus_check(pol, factor=3.0, floor_mV=2.0):
    """Where the light does not follow Malus: static stretches whose fit
    residual is `factor` times the rest's (and above `floor_mV`). Measured 7
    Oct 2026 (XEO1 hold series): at rest 0.9 mV, in every hold 4-6 mV, an
    additive per-analyzer-angle pattern of up to 0.3 % of Imax that is the
    same for every held voltage at a given angle, so not the polarization
    state - the fitted azimuth there is biased by up to ~0.1 deg and Imin by
    its size. Returns [(segment kind, rms_mV, rest_rms_mV, psi_scale_deg)]
    and a list of note strings."""
    t, rms = pol["t"], pol["rms"]
    if not np.any(np.isfinite(rms)):
        return [], []
    segs = segments(t, pol["rotation"])
    rest = [s for s in segs if s["base"] == "rest"]
    if not rest:
        return [], []
    m0 = (t >= rest[0]["t0"]) & (t <= rest[0]["t1"])
    r0 = float(np.nanmedian(rms[m0]) * 1e3)
    found, notes = [], []
    for s in segs:
        if s["base"] not in ("hold", "after"):
            continue
        m = (t >= s["t0"] + 0.2e-3) & (t <= s["t1"] - 0.2e-3)
        if m.sum() < 10:
            continue
        r = float(np.nanmedian(rms[m]) * 1e3)
        if r > factor * r0 and r > floor_mV:
            B = float(np.nanmedian(pol["B"][m]))
            scale = float(np.rad2deg(r * 1e-3 / (2 * B))) if B > 0 else float("nan")
            found.append((s["kind"], r, r0, scale))
            notes.append(f"{s['kind']}: the light departs from Malus (fit residual {r:.1f} mV "
                         f"vs {r0:.1f} at rest): its azimuth there is uncertain by ~{scale * 1e3:.0f} "
                         f"mdeg and Imin by ~{r:.0f} mV beyond the statistical errors")
    return found, notes


def er_sigma(er, imin, sig_imin, imax=None, sig_imax=0.0):
    """(down, up) error of ER = Imax / Imin from the statistical errors -
    asymmetric because ER goes as 1/Imin; up is inf where Imin is within
    1 sigma of zero."""
    r = abs(sig_imin / imin) if imin else np.inf
    q = abs(sig_imax / imax) if imax else 0.0
    if r >= 1:
        return er * (1 - 1 / (1 + r)), np.inf
    lo = er - er * (1 - q) / (1 + r)
    hi = er * (1 + q) / (1 - r) - er
    return float(lo), float(hi)


def direction(segs, t_s):
    """'away' on a ramp out (an 'up' segment), 'back' on a ramp back
    ('down'), 'static' in rest, holds and after. From the segment, not the
    local slope: within ~0.2 deg of rest the rotation rings through zero and
    a slope there called the end of a ramp back 'away' (7 Oct 2026, test-6
    and test-7)."""
    for s in segs:
        if s["t0"] <= t_s <= s["t1"]:
            return {"up": "away", "down": "back"}.get(s["base"], "static")
    return "static"


ER_COLUMNS = ["method", "leg", "direction", "segment", "t_ms", "rotation_deg",
              "analyzer_deg", "rate_deg_per_ms", "er", "er_sigma_lo", "er_sigma_hi",
              "lower_bound", "imin_mV", "imin_sigma_mV", "imax_V", "n", "note"]


def er_table(res, fit_bin_deg=2.0):
    """Every extinction-ratio value of an analysed scan (the GUI result
    dict) as rows of ER_COLUMNS, in time order within each method:
    'crossing' and 'static' (both intensities measured), 'dip' (Imin
    fitted around a crossing), 'refine' (null refine), then 'er_fit' (the
    per-sample Malus fit's ER: its median in `fit_bin_deg` rotation bins per
    leg and direction while moving, and per static stretch, with the 16th /
    84th percentiles as the spread). Static points whose angle sat too far
    from crossed (offset_limited) are kept and marked in 'note'.
    Returns (rows, facts): facts is a dict of numbers for the header."""
    d, pol = res["d"], res["pol"]
    segs = segments(pol["t"], pol["rotation"])
    downs = [x["t1"] for x in segs if x["base"] == "down"]
    t_end = (max(downs) + 1e-3) if downs else float(pol["t"][-1])
    rows = []

    def direction_(t_s, moving=True):
        return direction(segs, t_s) if moving else "static"

    def add(method, t_s, rotation, theta, rate, er, sig, lower, imin, simin, imax,
            n="", note="", moving=True, seg=None):
        lo, hi = sig
        rows.append({"method": method, "leg": leg_of(segs, t_s),
                     "direction": direction_(t_s, moving),
                     "segment": seg or next((x["kind"] for x in segs
                                             if x["t0"] <= t_s <= x["t1"]), ""),
                     "t_ms": t_s * 1e3, "rotation_deg": rotation, "analyzer_deg": theta,
                     "rate_deg_per_ms": rate, "er": er, "er_sigma_lo": lo,
                     "er_sigma_hi": hi, "lower_bound": int(bool(lower)),
                     "imin_mV": imin, "imin_sigma_mV": simin, "imax_V": imax, "n": n,
                     "note": note})
    for p in sorted(res.get("direct", []), key=lambda p: (p["kind"] != "crossing", p["t_ms"])):
        sig = (np.nan, np.nan) if p["lower"] else er_sigma(
            p["er"], p["imin_mV"], p["sig_mV"], p["imax_V"] * 1e3,
            p.get("sig_imax_V", 0.0) * 1e3)
        note = ""
        if p.get("offset_limited"):
            note = (f"offset-limited: the angle sat {p['off_deg']:+.2f} deg from crossed, "
                    f"which alone gives {p['imin_from_offset_mV']:.2f} mV - a useless lower "
                    f"bound, not the light's ER; measure analyzer {p['crossed_deg']:.1f} deg")
        elif p["kind"] == "static":
            note = f"angle {p['off_deg']:+.2f} deg from crossed"
            if p.get("bound_from") == "offset":
                note += (f" (its {p['imin_from_offset_mV']:.2f} mV is above the noise: "
                         f"lower bound)")
        add(p["kind"], p["t_ms"] * 1e-3, p["rotation"], float(wrap_deg(p["theta"])),
            p["rate"], p["er"], sig, p["lower"], p["imin_mV"], p["sig_mV"], p["imax_V"],
            note=note, moving=p["kind"] == "crossing", seg=p.get("seg"))
    for p in res.get("dips", []):
        sig = (np.nan, np.nan) if p["er_lower"] else er_sigma(p["er"], p["imin"], p["sig_imin"])
        add("dip", p["t"], p["rotation"], float(wrap_deg(p["theta"])), p["rate"] * 1e3,
            p["er"], sig, p["er_lower"], p["imin"] * 1e3, p["sig_imin"] * 1e3, p["imax"],
            n=p["n"])
    for r in res.get("refine", []):
        if "er" not in r:
            continue
        m = (pol["t"] >= r["t0"]) & (pol["t"] <= r["t1"])
        tm = 0.5 * (r["t0"] + r["t1"])
        sig = (np.nan, np.nan) if r["er_lower"] else er_sigma(r["er"], r["imin"], r["sig_imin"])
        add("refine", tm, float(np.mean(pol["rotation"][m])), float(wrap_deg(r["theta_null"])),
            0.0, r["er"], sig, r["er_lower"], r["imin"] * 1e3, r["sig_imin"] * 1e3,
            float(np.median(pol["imax"][m])), note=r.get("label", ""), moving=False)
    # the per-sample fit, binned
    t, rot, er = pol["t"], pol["rotation"], pol["er"]
    drift = pol.get("drift_resid")
    lim = 1 / drift if drift else None
    for sg in segs:
        m = (t >= sg["t0"]) & (t <= sg["t1"])
        moving = sg["base"] in ("up", "down")
        if not moving and sg["t0"] > t_end:
            continue                    # after the intensity lock switches off
        if moving:
            lo_, hi_ = np.min(rot[m]), np.max(rot[m])
            edges = np.arange(np.floor(lo_ / fit_bin_deg) * fit_bin_deg,
                              hi_ + fit_bin_deg, fit_bin_deg)
            groups = [(m & (rot >= a) & (rot < b)) for a, b in zip(edges[:-1], edges[1:])]
        else:
            groups = [m]
        for g in groups:
            if g.sum() < 5:
                continue
            e = er[g]
            med, p16, p84 = np.percentile(e, [50, 16, 84])
            tm = float(np.median(t[g]))
            notes = []
            if lim and med > lim:
                notes.append(f"above ER_fit's drift limit {lim:.0f}")
            nl = int(np.sum(pol["er_lower"][g]))
            if nl:
                notes.append(f"{nl} of {int(g.sum())} samples lower bounds")
            add("er_fit", tm, float(np.median(rot[g])), "", float(np.median(
                np.abs(np.gradient(rot, t))[g]) * 1e-3) if moving else 0.0,
                float(med), (float(med - p16), float(p84 - med)), False,
                float(np.median(pol["imin"][g]) * 1e3), float(np.median(pol["sig_imin"][g]) * 1e3),
                float(np.median(pol["imax"][g])), n=int(g.sum()), note="; ".join(notes),
                moving=moving, seg=sg["kind"])
    order = {"crossing": 0, "static": 1, "dip": 2, "refine": 3, "er_fit": 4}
    rows.sort(key=lambda r_: (order[r_["method"]], r_["t_ms"]))
    cr = [r_ for r_ in rows if r_["method"] == "crossing"]
    facts = {"segments": [(x["kind"], x["t0"] * 1e3, x["t1"] * 1e3) for x in segs],
             "drift_limit": lim, "t_end_ms": t_end * 1e3}
    if cr:
        im = np.array([r_["imin_mV"] for r_ in cr])
        facts.update(n_crossing=len(cr), n_crossing_lower=sum(r_["lower_bound"] for r_ in cr),
                     imin_median_mV=float(np.median(im)), imin_sd_mV=float(np.std(im)),
                     imin_range_mV=(float(im.min()), float(im.max())),
                     imax_median_V=float(np.median([r_["imax_V"] for r_ in cr])))
    ups = [x["t0"] for x in segs if x["base"] == "up"]
    if len(ups) >= 2:
        facts["leg_spacing_ms"] = (ups[1] - ups[0]) * 1e3
    return rows, facts


def wrap_deg(a):
    """An analyzer angle in [0, 180)."""
    return float(np.mod(a, 180.0))


def er_csv(res, extra=()):
    """The ER table as CSV text with a '#' header saying what the scan was,
    what was subtracted and corrected, and what limits the numbers.
    Read it with pandas.read_csv(path, comment='#')."""
    import csv
    import io
    rows, facts = er_table(res)
    d = res["d"]
    man = d.manifest
    plan = man.get("plan", {})
    corr = (res.get("corr") or {}).get("text", "")
    H = [f"extinction ratio along the ramp - scan {d.name}",
         f"measured {man.get('created', '')} - preset '{plan.get('preset', '')}', "
         f"{(res.get('pol') or {}).get('n_angles', '?')} analyzer angles, "
         f"{plan.get('shots', '?')} shots each, PD at {_pd_vdiv(d, 'scan')} V/div"]
    seq = plan.get("sequence")
    if seq:
        H.append("sequence (recorded): " + ", ".join(f"{k} {v:g}" for k, v in seq.items()))
    if facts.get("leg_spacing_ms"):
        H.append(f"legs {facts['leg_spacing_ms']:.3f} ms apart (measured from the light)")
    if man.get("notes"):
        H.append("notes: " + man["notes"].replace("\n", " "))
    H.append("corrections applied: " + (corr or "none"))
    H.append("segments (ms): " + "; ".join(f"{k} {a:.2f}..{b:.2f}" for k, a, b in facts["segments"]))
    if facts.get("n_crossing"):
        im0, im1 = facts["imin_range_mV"]
        H.append(f"crossings: {facts['n_crossing']} ({facts['n_crossing_lower']} lower bounds); "
                 f"Imin there {im0:+.2f}..{im1:+.2f} mV across analyzer angles (median "
                 f"{facts['imin_median_mV']:+.2f}, SD {facts['imin_sd_mV']:.2f}) against Imax "
                 f"{facts['imax_median_V']:.2f} V: an Imin of 2 SD would be ER "
                 f"{facts['imax_median_V'] / max(2e-3 * facts['imin_sd_mV'], 1e-12):.0f}")
    if facts.get("drift_limit"):
        H.append(f"ER_fit is drift-limited above ~{facts['drift_limit']:.0f} (reference returns)")
    H += list(extra)
    H += ["",
          "method: crossing = Imin read off the trace of the analyzer angle the light sweeps "
          "through crossed (4 us boxcar), Imax = the trace 90 deg away at the same instant; "
          "static = the angle nearest crossed averaged over a still stretch, Imax its 90-deg "
          "partner; dip = Imin fitted to the dip within +-8 deg of crossed, Imax from the "
          "per-sample fit; refine = null refine at a sensitive V/div; er_fit = the per-sample "
          "Malus fit's ER, median per rotation bin (moving) or per stretch (static), "
          "er_sigma_lo/hi = 16th/84th percentile spread",
          "leg: 1 = first transport, 2 = second; direction: away = on a ramp out from rest, "
          "back = on the ramp back, static = rest, hold or after",
          "rotation_deg: polarization rotation from rest (this scan's sign); analyzer_deg: "
          "the analyzer angle at crossed, in its own frame [0, 180)",
          "er_sigma_lo / er_sigma_hi: 1-sigma down / up (asymmetric: ER goes as 1/Imin); "
          "lower_bound = 1: Imin < 2 sigma, er = Imax / (2 sigma) and the true ER is above it",
          "rate_deg_per_ms: how fast the rotation swept through crossed"]
    buf = io.StringIO()
    for h in H:
        buf.write(f"# {h}\n" if h else "#\n")
    w = csv.DictWriter(buf, fieldnames=ER_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r_ in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) and np.isfinite(v)
                        else "" if isinstance(v, float) else v) for k, v in r_.items()})
    return buf.getvalue()


def leg_of(segs, t):
    """The transport (leg) a time belongs to, from segments(): 'up 2' ->
    2; a record with one transport, or a time before it, is leg 1."""
    for s in segs:
        if s["t0"] <= t <= s["t1"]:
            return max(1, int(s.get("leg", 1) or 1))
    return 1


# -- the fit's residual, and the polarization state ---------------------------

def malus_residual(pol):
    """What the Malus law does not explain, per angle and sample: (theta [K],
    resid [K, N] in V, z [K, N] = resid / that step's standard error). The
    model is a0 + c2 cos 2theta + s2 sin 2theta (the 1- and 4-theta
    diagnostics are left IN the residual, so they show). Bad angles, clipping,
    a missed lock or slow drift show up as rows or patches."""
    th = np.asarray(pol["theta"], float)
    r = np.deg2rad(th)[:, None]
    model = pol["a0"][None, :] + pol["c2"][None, :] * np.cos(2 * r) \
        + pol["s2"][None, :] * np.sin(2 * r)
    resid = pol["I"] - model
    g = pol.get("angle_gain")
    sem = np.array([np.asarray(s["sem"]["PD"], float) for s in pol["steps"]])
    if g is not None:
        sem = sem / np.asarray(g)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sem > 0, resid / sem, np.nan)
    return th, resid, z


def stokes(pol, a0=None, c2=None, s2=None):
    """The polarization state the rotating analyzer measures, per sample, in
    the REST frame (azimuth measured from the rest azimuth). s1, s2: the
    normalised linear Stokes parameters; p: their length (the degree of
    LINEAR polarization, B / a0); chi_deg: the ellipticity angle a FULLY
    polarized beam with this p would have (tan chi = sqrt(Imin / Imax)); s3:
    sqrt(1 - p^2) under the same assumption, sign unknown. A linear analyzer
    alone cannot tell ellipticity from depolarization, nor the handedness:
    s3 and chi are upper bounds on the circular part. a0/c2/s2 may be passed
    pre-smoothed."""
    a0 = pol["a0"] if a0 is None else a0
    c2 = pol["c2"] if c2 is None else c2
    s2 = pol["s2"] if s2 is None else s2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.clip(np.hypot(c2, s2) / a0, 0.0, 1.0)
    two_psi = np.arctan2(s2, c2) - np.deg2rad(2 * pol["psi_rest"])
    s3 = np.sqrt(np.maximum(1 - p * p, 0.0))
    chi = 0.5 * np.rad2deg(np.arcsin(s3))
    return {"s1": p * np.cos(two_psi), "s2": p * np.sin(two_psi), "s3": s3, "p": p,
            "chi_deg": chi, "azimuth_deg": 0.5 * np.rad2deg(np.unwrap(two_psi))}


def scan_summary(res, direct=None):
    """The key numbers of an analysed scan (GUI result dict) for the lab log
    and the brief: plain values, JSON-able."""
    d, pol = res["d"], res.get("pol")
    man = d.manifest
    out = {"name": d.name, "created": man.get("created", ""),
           "updated": man.get("updated", ""), "folder": d.folder,
           "steps_done": res.get("n_done"), "steps_total": res.get("n_total"),
           "complete": res.get("n_done") == res.get("n_total"),
           "pd_vdiv": _pd_vdiv(d, "scan"),
           "shots_per_angle": man.get("plan", {}).get("shots"),
           "preset": man.get("plan", {}).get("preset", ""),
           "sequence": man.get("plan", {}).get("sequence"),
           "notes": man.get("notes", ""), "drive": man.get("drive")}
    cs = res.get("corr") or {}
    out["subtracted"] = cs.get("subtracted_kind")
    out["subtracted_mV"] = None if cs.get("subtracted") is None else cs["subtracted"] * 1e3
    out["corrections"] = cs.get("text", "")
    if cs.get("dropped"):
        out["shots_dropped"], out["shots_total"] = cs["dropped"]
    if pol is not None:
        rot = pol["rotation"]
        out.update(n_angles=int(pol["n_angles"]), psi_rest_deg=float(pol["psi_rest"]),
                   rotation_max_deg=float(np.max(rot)), rotation_min_deg=float(np.min(rot)),
                   drift_resid=pol.get("drift_resid"),
                   fit_residual_mV_median=float(np.nanmedian(pol["rms"]) * 1e3)
                   if np.any(np.isfinite(pol["rms"])) else None)
        segs = segments(pol["t"], rot)
        out["segments"] = []
        for s in segs:
            m = (pol["t"] >= s["t0"]) & (pol["t"] <= s["t1"])
            out["segments"].append({"kind": s["kind"], "t0_ms": s["t0"] * 1e3,
                                    "t1_ms": s["t1"] * 1e3,
                                    "rotation_deg": float(np.median(rot[m])),
                                    "er_fit_median": float(np.median(pol["er"][m])),
                                    "fit_rms_mV": float(np.nanmedian(pol["rms"][m]) * 1e3)
                                    if np.any(np.isfinite(pol["rms"][m])) else None})
        out["malus_notes"] = malus_check(pol)[1]
    if direct:
        res_pts = [p for p in direct if not p["lower"] and not p.get("offset_limited")]
        if res_pts:
            lo = min(res_pts, key=lambda p: p["er"])
            out["direct_er_min"] = {k: lo[k] for k in ("er", "kind", "seg", "t_ms", "theta",
                                                       "rotation", "imin_mV", "imax_V")}
        out["direct_er_n"] = len(direct)
        out["direct_er_lower_bounds"] = sum(p["lower"] for p in direct)
        out["direct_er_offset_limited"] = sum(bool(p.get("offset_limited")) for p in direct)
    out["provenance"] = man.get("provenance")
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
    peak = float(np.max(np.abs(rotation))) if len(rotation) else 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (t[b - 1] - t[a]) < min_ms * 1e-3:
            continue
        rmean = float(np.mean(rotation[a:b]))
        if static[a]:
            # A static stretch is named by the motion before it: after an
            # "up" it is the hold, after a "down" the after-ramp rest - unless
            # the ramp came down only part of the way (a hold at another
            # level). Until 7 Oct 2026 "hold" meant |rotation| > 45 deg, so
            # the 10 ms holds of the 15 and 30 deg ramps were called "after".
            last = out[-1]["base"] if out else None
            if last is None:
                base = "rest"
            elif last == "up" or abs(rmean) > 0.5 * peak:
                base = "hold"
            else:
                base = "after"
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
               "dark_vdiv_matched": bool(dstep) and bool(dstep.get("vdiv")) and
               abs(np.log(dstep["vdiv"] / vdiv)) < 1e-6 if vdiv else False,
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
    over the scan steps. Its sign comes from the light (least squares); its
    offset makes it zero at rest before the first motion, where the light's
    rotation is zero by definition - so light - monitors starts at zero and
    shows where they part. Returns (pred_rotation, residual, sign) or None
    without monitors.

    Before 7 Oct 2026 the offset was the whole record's mean difference.
    With long holds the light creeps away from the monitors (-20..-33
    mdeg/ms at -180 deg) and stays off for ~100 ms after the ramp back, and
    that mean moved the rest by 0.35 deg (20 ms holds) to 0.8 deg (50 ms)."""
    roles = [r for r in ("MonX1", "MonX2") if r in d.roles]
    if not roles:
        return None
    steps = pol["steps"]
    pred = np.zeros(len(d.t))
    for r in roles:
        k = float(deg_per_mon_v.get(r, 0.0))
        pred += k * np.mean([s["v"][r] for s in steps if r in s["v"]], axis=0)
    segs = segments(pol["t"], pol["rotation"])
    t = pol["t"]
    first = next((s for s in segs if s["base"] in ("up", "down")), None)
    rest = (t < first["t0"] - 0.2e-3) if first else np.ones(len(t), bool)
    if rest.sum() < 10:
        rest = np.ones(len(t), bool)       # no rest before the first motion
    best = None
    for sign in (1.0, -1.0):
        p = sign * pred
        c = float(np.mean(pol["rotation"][rest] - p[rest]))
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

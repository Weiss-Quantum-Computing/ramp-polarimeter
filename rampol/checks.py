"""Is the scope set up for the data that is about to arrive?

Two halves. `settings_checks` reads nothing new: it judges the settings
snapshot (Scope.read_settings) against the channel roles and the scan plan -
trigger sweep, channels displayed, coupling, the time window, the trigger
wait. `shot_checks` judges one real acquisition: clipping and screen use per
channel, whether the light and the ramps are there, and whether the ramps sit
inside the record. Every finding is (level, text) with level OK, INFO, WARN
or FAIL; a FAIL means the scan would record something unusable.

Why a single shot is enough for the PD: the ramp sweeps the polarization
through 180 deg, so at ANY analyzer angle it passes maximum transmission at
some point in the record - one shot shows the brightest the PD will get.
"""
import numpy as np

from .config import ROLE_NAMES, record_span

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"
SCREEN_DIVS = 4.0        # the MSO-X shows offset +- 4 div
CLIP_DIVS = 4.9          # its converter runs out a little past the screen


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _on(x):
    return str(x).strip().upper() in ("1", "ON")


def settings_checks(settings, prof, roles, plan):
    """Judge a settings snapshot. `roles` is {ch: role} for every recorded
    channel; `plan` the scan settings (mode, rep_s, wait_s, points,
    dither_codes)."""
    out = []
    s = settings.get
    sweep = str(s(":TRIGger:SWEep", "")).upper()
    if sweep.startswith("AUTO"):
        out.append((FAIL, "trigger sweep is AUTO: the scope triggers itself when no "
                          "trigger comes, so shots between real triggers are noise - "
                          "set NORMAL"))
    elif sweep:
        out.append((OK, f"trigger sweep {sweep}"))
    src = str(s(":TRIGger:EDGE:SOURce", "")).upper()
    lvl = s(":TRIGger:EDGE:LEVel", "?")
    slope = s(":TRIGger:EDGE:SLOPe", "?")
    if src.startswith("LINE"):
        out.append((WARN, "trigger source is LINE (mains): only right for a "
                          "deliberate mains-locked measurement"))
    elif src.startswith("CHAN"):
        ch = int(src[-1]) if src[-1].isdigit() else None
        role = roles.get(ch)
        if ch is not None and ch not in roles:
            out.append((WARN, f"trigger source {src} is a channel this scan does not record"))
        else:
            out.append((INFO, f"trigger on {src} ({ROLE_NAMES.get(role, role)}), "
                              f"level {lvl} V, slope {slope}"))
    elif src:
        out.append((INFO, f"trigger on {src}, level {lvl} V, slope {slope}"))
    acq = str(s(prof.acq_type, "")).upper()
    mode = plan.get("mode", "single")
    if mode == "single" and not acq.startswith("HRES"):
        out.append((INFO, f"acquisition is {acq}; the scan switches it to HRES for "
                          f"single shots and puts it back afterwards"))
    t0, t1 = record_span(s(":TIMebase:SCALe", "nan"), s(":TIMebase:POSition", "nan"),
                         s(":TIMebase:REFerence", "LEFT"))
    if np.isfinite(t0):
        npts = plan.get("points") if mode == "single" else 7680
        dt = (t1 - t0) / npts if npts else float("nan")
        out.append((INFO, f"record {t0 * 1e3:+.2f} to {t1 * 1e3:+.2f} ms from the "
                          f"trigger, {npts} points = {dt * 1e6:.2f} us spacing"))
        if t0 >= 0:
            out.append((WARN, "the record starts at or after the trigger: there is no "
                              "pre-trigger stretch for the rest level and the "
                              "missed-lock check"))
    rep, wait = _f(plan.get("rep_s")), _f(plan.get("wait_s"))
    if np.isfinite(rep) and np.isfinite(wait) and wait <= rep:
        out.append((FAIL, f"trigger wait {wait:g} s is not longer than the repetition "
                          f"period {rep:g} s: shots will time out"))
    for ch, role in sorted(roles.items()):
        name = f"CH{ch} ({ROLE_NAMES.get(role, role)})"
        disp = s(prof.ch_display.format(ch=ch))
        if disp is not None and not _on(disp):
            out.append((FAIL, f"{name} is switched off on the scope: it has no record "
                              f"to read out"))
        coup = str(s(f":CHANnel{ch}:COUPling", "")).upper()
        if coup.startswith("AC") and role in ("PD", "MonX1", "MonX2", "CmdX1", "CmdX2", "Ref"):
            out.append((WARN, f"{name} is AC coupled: its high-pass turns level "
                              f"steps into slow decays (seen 1 Oct 2026) - use DC"))
        sc, off = _f(s(prof.ch_scale.format(ch=ch))), _f(s(prof.ch_offset.format(ch=ch)))
        if np.isfinite(sc) and np.isfinite(off):
            out.append((INFO, f"{name}: {sc:g} V/div, screen {off - SCREEN_DIVS * sc:+.3f} "
                              f"to {off + SCREEN_DIVS * sc:+.3f} V"))
    return out


def _activity(t, v, frac=0.05):
    """(first, last) time the trace leaves its starting level by more than
    `frac` of its swing, or None if it never does."""
    base = np.median(v[:max(10, len(v) // 50)])
    dev = np.abs(v - base)
    swing = dev.max()
    if swing <= 0:
        return None
    idx = np.flatnonzero(dev > frac * swing)
    return (t[idx[0]], t[idx[-1]]) if len(idx) else None


def shot_checks(t, traces, settings, prof, roles, plan):
    """Judge one acquisition. `traces` is {ch: volts}, `roles` {ch: role}."""
    out = []
    s = settings.get
    span = t[-1] - t[0]
    edge = 0.02 * span
    codes = float(plan.get("dither_codes", 0) or 0)
    for ch, role in sorted(roles.items()):
        v = traces.get(ch)
        if v is None or not len(v):
            continue
        name = f"CH{ch} ({ROLE_NAMES.get(role, role)})"
        sc, off = _f(s(prof.ch_scale.format(ch=ch))), _f(s(prof.ch_offset.format(ch=ch)))
        if not (np.isfinite(sc) and np.isfinite(off)):
            continue
        # the dither moves the offset by up to half its span either way
        half = 0.5 * codes * prof.adc_code_per_vdiv * sc
        lo, hi = float(np.min(v)), float(np.max(v))
        top_div = (hi - off + half) / sc
        bot_div = (off + half - lo) / sc
        worst = max(top_div, bot_div)
        if worst >= CLIP_DIVS:
            out.append((FAIL, f"{name} reaches {hi:+.3f} / {lo:+.3f} V: past the "
                              f"converter's range at {sc:g} V/div, offset {off:+.3f} - "
                              f"clipped. Raise V/div or move the offset"))
        elif worst > SCREEN_DIVS:
            out.append((WARN, f"{name} goes {worst - SCREEN_DIVS:.2f} div off screen "
                              f"({lo:+.3f} to {hi:+.3f} V, dither included): outside the "
                              f"calibrated range - move the offset"))
        else:
            used = (hi - lo) / (2 * SCREEN_DIVS * sc)
            level = OK
            note = ""
            if role == "PD" and used < 0.25:
                level = WARN
                note = " - a smaller V/div would resolve it better"
            out.append((level, f"{name}: {lo:+.3f} to {hi:+.3f} V, {used:.0%} of the "
                               f"screen{note}"))
        if role == "PD" and hi < 0.05:
            out.append((WARN, f"{name} never rises above {hi * 1e3:.0f} mV: is the light on "
                              f"and the beam unblocked?"))
        if role in ("MonX1", "MonX2", "CmdX1", "CmdX2"):
            act = _activity(t, v)
            if hi - lo < 0.3:
                out.append((WARN, f"{name} swings only {(hi - lo) * 1e3:.0f} mV: is the "
                                  f"ramp running?"))
            elif act is not None:
                a, b = act
                msg = (f"{name} ramp activity {a * 1e3:+.2f} to {b * 1e3:+.2f} ms "
                       f"(record {t[0] * 1e3:+.2f} to {t[-1] * 1e3:+.2f})")
                if a - t[0] < edge or t[-1] - b < edge:
                    out.append((WARN, msg + ": it reaches the edge of the record - "
                                            "a ramp may be cut off; move the position "
                                            "or lengthen the timebase"))
                else:
                    out.append((OK, msg))
    return out


def summary(findings):
    worst = max((("OK", "INFO", "WARN", "FAIL").index(lv) for lv, _ in findings), default=0)
    return ("OK", "INFO", "WARN", "FAIL")[worst]

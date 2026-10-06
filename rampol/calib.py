"""The EOM voltage calibration: AWG volts -> Trek monitor volts -> HV -> rotation.

One chain per crystal, every step a number you can see and edit:

    monitor V = gain x (AWG V - idle)          gain: AWG -> monitor, V/V
    HV (kV)   = monitor V / mon_per_kv          the monitor's scale, V per kV (1)
    rotation  = 90 deg x HV / v90_kv            v90_kv: HV for 90 deg (V_pi/2 x ...)

and the rotation of the pair is the sum of the two crystals'. Everything
that turns volts into degrees reads it from here: the AWG waveforms
(rotation -> AWG volts), the bias run's monitor prediction, the scan
analysis's rotation-from-monitors, the ILC-target comparison.

Sources: the 1 Sep 2026 optical calibration (the defaults; EOM-ILC's
eomilc.config holds the same numbers: cmd_hv_gain_meas, v90_hv), EOM-ILC's
config as it is now, or a fit to a bias run (gains from monitor vs AWG per
point; V90 of both crystals scaled by the light/monitors gain of the
static transfer curve - the two crystals are driven together there, so
their V90s are not separable from one run).
"""
import copy
import datetime

import numpy as np

NAMES = ("EO1", "EO2")
MON_ROLE = {"EO1": "MonX1", "EO2": "MonX2"}

DEFAULT = {
    "EO1": {"gain": 0.5594, "mon_per_kv": 1.0, "v90_kv": 5.1283},
    "EO2": {"gain": 0.5924, "mon_per_kv": 1.0, "v90_kv": 5.1374},
    "source": "1 Sep 2026 optical calibration (V90 at the monitor 5128.3 / 5137.4 V; "
              "AWG -> monitor 0.5594 / 0.5924)",
    "date": "2026-09-01",
}


def get(cfg):
    """The calibration in the config, filled from DEFAULT where missing."""
    cal = copy.deepcopy(DEFAULT)
    for k, v in (cfg.get("calibration") or {}).items():
        if k in NAMES and isinstance(v, dict):
            cal[k].update({kk: float(vv) for kk, vv in v.items()})
        else:
            cal[k] = v
    return cal


def validate(cal):
    for n in NAMES:
        c = cal[n]
        for k in ("gain", "mon_per_kv", "v90_kv"):
            v = float(c[k])
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{n} {k} = {v}: must be a positive number")
        if not 0.2 < c["gain"] < 2.0:
            raise ValueError(f"{n} gain {c['gain']} V/V is far from the ~0.56-0.59 measured")
        if not 1.0 < c["v90_kv"] < 10.0:
            raise ValueError(f"{n} V90 {c['v90_kv']} kV is far from the ~5.1 kV measured")
    return cal


def v90_mon(cal, n):
    """Monitor volts for 90 deg on crystal n."""
    return cal[n]["v90_kv"] * cal[n]["mon_per_kv"]


def deg_per_awg_v(cal, n):
    return 90.0 * cal[n]["gain"] / v90_mon(cal, n)


def deg_per_mon_v(cal, n):
    return 90.0 / v90_mon(cal, n)


def convert(cal, n, awg=None, mon=None, kv=None, deg=None):
    """Any one of AWG V (above idle), monitor V, kV, deg for crystal n ->
    all four."""
    c = cal[n]
    if awg is not None:
        mon = c["gain"] * awg
    elif kv is not None:
        mon = kv * c["mon_per_kv"]
    elif deg is not None:
        mon = deg / deg_per_mon_v(cal, n)
    if mon is None:
        raise ValueError("give one of awg, mon, kv, deg")
    return {"awg": mon / c["gain"], "mon": mon, "kv": mon / c["mon_per_kv"],
            "deg": mon * deg_per_mon_v(cal, n)}


def chan(cal):
    """bias.CHAN's shape: {EO1: {gain, v90 (monitor V), awg, mon}, ...}."""
    return {n: {"gain": cal[n]["gain"], "v90": v90_mon(cal, n), "awg": i + 1,
                "mon": MON_ROLE[n]} for i, n in enumerate(NAMES)}


def apply(cal, cfg=None):
    """Make every consumer use `cal`: bias.CHAN (AWG waveforms, bias runs),
    ilc_target.V90, and the config's analysis deg_per_mon_v (the scan's
    rotation from the monitors)."""
    from . import bias, ilc_target
    validate(cal)
    for n, c in chan(cal).items():
        bias.CHAN[n].update(c)
    for n in NAMES:
        ilc_target.V90[n] = v90_mon(cal, n)
    if cfg is not None:
        cfg["calibration"] = {k: (dict(v) if isinstance(v, dict) else v)
                              for k, v in cal.items()}
        cfg["analysis"]["deg_per_mon_v"] = {MON_ROLE[n]: deg_per_mon_v(cal, n)
                                            for n in NAMES}
    return cal


def from_eomilc(eomilc_mod=None):
    """EOM-ILC's eomilc.config as it is now."""
    from eomilc.config import CHANNELS
    cal = copy.deepcopy(DEFAULT)
    for n in NAMES:
        ch = CHANNELS[n]
        mon_scale = float(getattr(ch, "mon_scale", 1000.0))     # HV volts per monitor volt
        cal[n] = {"gain": float(ch.cmd_hv_gain_meas),
                  "mon_per_kv": 1000.0 / mon_scale,
                  "v90_kv": float(ch.v90_hv) / 1000.0}
    cal["source"] = "EOM-ILC eomilc.config (cmd_hv_gain_meas, v90_hv, mon_scale)"
    cal["date"] = datetime.date.today().isoformat()
    return cal


def from_bias(man, base):
    """A fit to a bias run (rampol.bias.load): per crystal the AWG -> monitor
    gain from the points' hold-window monitor volts against the AWG volts
    (slope through the origin, both relative to idle), and both V90s scaled
    by the static transfer curve's light / monitors gain. Returns (cal,
    report lines)."""
    pts = [p for p in man.get("points", []) if p.get("mon_V") and p.get("awg")]
    if len(pts) < 2:
        raise ValueError("the bias run has fewer than 2 points with monitor readings")
    cal = copy.deepcopy(base)
    rep = []
    for n in NAMES:
        role = MON_ROLE[n]
        a = np.array([p["awg"][n] for p in pts if role in p["mon_V"]])
        m = np.array([p["mon_V"][role] for p in pts if role in p["mon_V"]])
        if len(a) < 2 or np.ptp(a) <= 0:
            rep.append(f"{n}: no monitor readings - gain left at {base[n]['gain']:.4f}")
            continue
        g = float(a @ m / (a @ a))
        res = m - g * a
        rep.append(f"{n}: gain {g:.4f} V/V (was {base[n]['gain']:.4f}; residual "
                   f"{np.std(res)*1e3:.1f} mV rms over {len(a)} points)")
        cal[n]["gain"] = g
    tf = man.get("transfer") or {}
    lg = tf.get("gain")
    if lg:
        for n in NAMES:
            cal[n]["v90_kv"] = base[n]["v90_kv"] / float(lg)
        rep.append(f"V90 of both crystals / {lg:.4f} (light per monitor-predicted degree, "
                   f"{tf.get('rms_resid', 0)*1e3:.0f} mdeg rms left)")
    cal["source"] = f"fit to bias run {man.get('name', '?')}"
    cal["date"] = datetime.date.today().isoformat()
    return cal, rep


def summary(cal):
    parts = []
    for n in NAMES:
        parts.append(f"{n}: {cal[n]['gain']:.4f} V/V, V90 {cal[n]['v90_kv']:.4f} kV "
                     f"-> {deg_per_awg_v(cal, n):.3f} deg/V at the AWG")
    return "; ".join(parts) + f" ({cal.get('source', '')})"

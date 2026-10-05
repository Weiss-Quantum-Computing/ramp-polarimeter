"""Settings that persist between sessions, and the presets for the two kinds of
ramp this measures.

The file lives in %APPDATA%\\RampPolarimeter\\config.json. Tests point
CONFIG_PATH at a temporary file before anything reads or writes it - the same
rule Scope Grab learned the hard way (a test once rewrote the live config).
"""
import copy
import json
import os

CONFIG_PATH = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")),
                           "RampPolarimeter", "config.json")

PROJECTS = os.path.join(os.path.expanduser("~"), "Desktop", "Python Projects")

# What a scope channel carries. The analysis looks channels up by role, never
# by number, so the wiring can change without touching anything else.
ROLES = ("off", "PD", "MonX1", "MonX2", "Ref", "Marker", "Other")
ROLE_NAMES = {
    "off": "not recorded",
    "PD": "analyzer photodiode",
    "MonX1": "Trek monitor X1",
    "MonX2": "Trek monitor X2",
    "Ref": "reference photodiode (before the analyzer)",
    "Marker": "trigger / sequence marker",
    "Other": "recorded, not analysed",
}

# Rotation per monitor volt, from the optically measured V_pi (31 Aug 2026):
# with the QWP making the EOM pair a rotator, phi = 90 deg * V_HV / V_pi per
# crystal, and the monitors read 1 V per kV. 90e3 / 5128.3 and 90e3 / 5137.4.
DEG_PER_MON_V = {"MonX1": 17.550, "MonX2": 17.519}

# Where a record starts relative to the trigger, in divisions before
# :TIMebase:POSition, by :TIMebase:REFerence. MEASURED for LEFT on the MSO-X
# (5 Oct 2026 dry run: position -2 ms at 1.5 ms/div started at -3.49 ms -
# LEFT is one division in from the edge, not the edge); CENTer/RIGHt follow.
REF_DIVS = {"LEFT": 1.0, "CENT": 5.0, "RIGH": 9.0}


def record_span(scale, position, reference, divs=10.0):
    """(start, stop) in seconds of the record the timebase settings give."""
    k = REF_DIVS.get(str(reference).strip().upper()[:4], 5.0)
    t0 = float(position) - k * float(scale)
    return t0, t0 + divs * float(scale)


# Presets: scope settings to write (Apply to scope) and scan settings to fill
# in (on picking one). Built-ins here; ones saved from the Scope settings
# window go to the config under "user_presets" and override these by name.
PRESETS = {
    "Spin echo 16.7 ms (2 legs)": {
        "note": "Experiment-control sequence, both legs in one record: ramps "
                "4.5 ms up / 0.5 ms hold / 4.5 ms down, legs 16.667 ms apart, "
                "trigger before leg 1, ~10 s repetition. 5 ms/div = 50 ms "
                "record from -2 ms, room for the after-ramp relaxation. Trigger "
                "sweep NORMAL: AUTO would self-trigger in a 10 s gap.",
        "scope": {":ACQuire:TYPE": "HRESolution",
                  ":TIMebase:SCALe": "5.0E-03", ":TIMebase:REFerence": "LEFT",
                  ":TIMebase:POSition": "3.0E-03",
                  ":TRIGger:MODE": "EDGE", ":TRIGger:SWEep": "NORMal"},
        "scan": {"mode": "single", "shots": 8, "points": 100000,
                 "wait_s": 30.0, "rep_s": 10.0},
    },
    "AWG bench ramp": {
        "note": "target_PARX1-style ramp, 11 ms, DS345 trigger at 3.6997 Hz "
                "on EXT. 1.5 ms/div = 15 ms record from -3.5 ms.",
        "scope": {":ACQuire:TYPE": "HRESolution",
                  ":TIMebase:SCALe": "1.5E-03", ":TIMebase:REFerence": "LEFT",
                  ":TIMebase:POSition": "-2.0E-03",
                  ":TRIGger:MODE": "EDGE", ":TRIGger:SWEep": "NORMal",
                  ":TRIGger:EDGE:SOURce": "EXT"},
        "scan": {"mode": "single", "shots": 32, "points": 20000,
                 "wait_s": 10.0, "rep_s": 0.27},
    },
}


def all_presets(cfg):
    out = dict(PRESETS)
    out.update(cfg.get("user_presets") or {})
    return out


DEFAULTS = {
    "scope_grab_path": os.path.join(PROJECTS, "scope-grab-multi", "scope_grab.py"),
    "scope_model": "msox2014a",
    "scope_addr": "",
    "ell_port": "COM3",
    "ell_address": "0",
    # Mount angle (deg) that is analyzer 0. By the campaign's convention 0 is
    # aligned with the polarization at the EO zero - set it from a scan with
    # "Set zero from rest", never from the engraving on the mount.
    "ell_zero_deg": 0.0,
    "simulate": False,
    "channels": {
        "1": {"role": "PD", "name": "Analyzer PD"},
        "2": {"role": "off", "name": ""},
        "3": {"role": "MonX1", "name": "Trek monitor X1"},
        "4": {"role": "MonX2", "name": "Trek monitor X2"},
    },
    "outdir": os.path.join(PROJECTS, "scope_data", "polarimetry"),
    "scan_name": "scan",
    "preset": "Spin echo 16.7 ms (2 legs)",
    "user_presets": {},
    "scan": {
        # Malus repeats every 180 deg, so 0-170 is a full set; 360 deg of
        # coverage only adds the 1- and 4-theta diagnostics
        "start": 0.0, "stop": 170.0, "step": 10.0,
        "order": "forward",       # forward | bidirectional | shuffled
        "mode": "single",         # single (HRES shots, averaged here) | average (:DIGitize)
        "shots": 8,               # per angle; single: one file per shot
        "blocks": 4,              # average mode only: dither positions (one file each)
        "dither_codes": 3,
        "ref_every": 6,           # return to ref_angle after every N angles (0 = never)
        "ref_angle": 45.0,
        "backoff_deg": 3.0,       # approach every angle from below by this much
        "points": 100000,         # single-shot readout points
        "wait_s": 30.0,           # trigger stall limit (> the repetition period)
        "rep_s": 10.0,            # trigger period, for the time estimate only
    },
    "refine": {
        "offsets": "-4,-2,-1,-0.5,0,0.5,1,2,4",
        "pd_vdiv": 0.02,          # PD V/div for the null captures
        "windows": "auto",        # auto, or "t1-t2,t3-t4" in ms
        "shots": 64,
        "blocks": 4,
    },
    "analysis": {
        "trim": 10,               # samples dropped from each record start (sample-0 artefact)
        # single-shot scans: drop a shot whose pre-trigger level is off its
        # step's median by more than this fraction of the brightest level -
        # the intensity lock missed 11 of 60 records on 1 Oct 2026 (0 = keep all)
        "lock_tol": 0.006,
        "polarizer_er": 1.0e4,    # analyzer's own extinction ratio (LPVIS100 spec floor)
        "deg_per_mon_v": dict(DEG_PER_MON_V),
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load():
    """DEFAULTS with whatever the file holds laid over them. A missing or
    unreadable file gives the defaults."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            return _merge(DEFAULTS, json.load(fh))
    except (OSError, ValueError):
        return copy.deepcopy(DEFAULTS)


def replace_retrying(tmp, path, tries=20, wait=0.05):
    """os.replace, retried: on Windows a virus scanner, the indexer or
    OneDrive can hold the target open for a moment and the rename then fails
    with Access is denied (seen 5 Oct 2026 on a scan manifest)."""
    import time
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait)


def save(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    tmp = CONFIG_PATH + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    replace_retrying(tmp, CONFIG_PATH)


def channel_roles(cfg):
    """{ch: (role, name)} for every recorded channel, in channel order."""
    out = {}
    for ch in sorted(cfg["channels"], key=int):
        c = cfg["channels"][ch]
        if c.get("role", "off") != "off":
            out[int(ch)] = (c["role"], c.get("name") or ROLE_NAMES[c["role"]])
    return out

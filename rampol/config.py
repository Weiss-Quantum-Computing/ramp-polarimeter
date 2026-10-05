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

# Scope settings a preset may write before a scan. Values are SCPI strings for
# the MSO-X profile; a preset is applied only when asked (the Apply preset
# button), so a scan otherwise runs on whatever the scope is set to.
PRESETS = {
    "AWG bench ramp": {
        "note": "target_PARX1-style ramp, 11 ms, DS345 trigger at 3.6997 Hz "
                "on EXT. 1.5 ms/div puts the whole 15 ms record on screen.",
        "scope": {":TIMebase:SCALe": "1.5E-03", ":TIMebase:REFerence": "LEFT",
                  ":TIMebase:POSition": "-2.0E-03",
                  ":TRIGger:EDGE:SOURce": "EXT"},
        "shots": 64,
    },
    "Spin-echo sequence": {
        "note": "Experiment-control sequence: both legs in one record, lattice "
                "presaturation before leg 1. 10 ms/div = 100 ms record; the "
                "trigger is the sequence marker.",
        "scope": {":TIMebase:SCALe": "1.0E-02", ":TIMebase:REFerence": "LEFT",
                  ":TIMebase:POSition": "-5.0E-03"},
        "shots": 32,
    },
}

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
    "preset": "AWG bench ramp",
    "scan": {
        "start": 0.0, "stop": 350.0, "step": 10.0,
        "order": "forward",       # forward | bidirectional | shuffled
        "mode": "average",        # average (scope :DIGitize) | single (HRES shots)
        "shots": 64,              # per angle, split over the dither blocks
        "blocks": 4,              # dither positions per angle (one file each)
        "dither_codes": 3,
        "ref_every": 6,           # return to ref_angle after every N angles (0 = never)
        "ref_angle": 45.0,
        "backoff_deg": 3.0,       # approach every angle from below by this much
        "points": 20000,          # single-shot readout points
        "wait_s": 10.0,           # trigger stall limit
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

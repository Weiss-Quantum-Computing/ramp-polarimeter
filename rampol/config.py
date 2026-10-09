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
ROLES = ("off", "PD", "MonX1", "MonX2", "CmdX1", "CmdX2", "Ref", "Marker", "Other")
ROLE_NAMES = {
    "off": "not recorded",
    "PD": "analyzer photodiode",
    "MonX1": "Trek monitor X1",
    "MonX2": "Trek monitor X2",
    "CmdX1": "Trek X1 command",       # the drive into the Trek input (AWG or NI card)
    "CmdX2": "Trek X2 command",
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


# Channel settings by ROLE, so a preset follows the wiring: written to
# whichever channel carries that role when the preset is applied. Offsets put
# 0 V about 2.7 div below centre for signals that run 0 to ~5 V (PD, monitors)
# and the ~8.5 V commands on 2 V/div.
ROLE_CHANNELS = {
    "PD": {"scale": 1.0, "offset": 2.7},
    "MonX1": {"scale": 1.0, "offset": 2.5},
    "MonX2": {"scale": 1.0, "offset": 2.5},
    "CmdX1": {"scale": 2.0, "offset": 4.0},
    "CmdX2": {"scale": 2.0, "offset": 4.0},
    "Marker": {"scale": 2.0, "offset": 2.0},
    "Ref": {"scale": 1.0, "offset": 2.5},
}

# Presets: scope settings to write (Apply to scope) and scan settings to fill
# in (on picking one). "scope" is {SCPI root: value}; "roles" is channel
# settings by role (V/div, offset; DC coupling and display on are always
# written for every recorded channel). Built-ins here; ones saved from the
# Scope settings window go to the config under "user_presets" and override
# these by name.
SPIN_ECHO = "Spin echo"
OLD_SPIN_ECHO = "Spin echo 16.7 ms (2 legs)"      # its name before 7 Oct 2026

PRESETS = {
    SPIN_ECHO: {
        "note": "Experiment-control spin-echo sequence, both legs in one record, "
                "trigger at the start of the first motion, ~10 s repetition. The "
                "record runs from 'before' ms before the trigger to 'after' ms after "
                "the second motion ends (spacing + motion): the timebase is "
                "computed from the sequence fields (defaults: legs 16.667 ms apart, "
                "9.5 ms motions - 4.5 up / 0.5 hold / 4.5 down - 12 ms before and "
                "after: about 5 ms/div). The intensity lock switches off at the end "
                "of leg 2, so much later data is not wanted. Trigger sweep NORMAL: "
                "AUTO would self-trigger in a 10 s gap.",
        # the timebase comes from cfg["sequence"] (sequence_timebase)
        "sequence": True,
        "scope": {":ACQuire:TYPE": "HRESolution",
                  ":TIMebase:SCALe": "5.0E-03", ":TIMebase:REFerence": "LEFT",
                  # LEFT starts the record one division before the position
                  ":TIMebase:POSition": "-7.0E-03",
                  ":TRIGger:MODE": "EDGE", ":TRIGger:SWEep": "NORMal"},
        "roles": ROLE_CHANNELS,
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
        "roles": ROLE_CHANNELS,
        "scan": {"mode": "single", "shots": 32, "points": 20000,
                 "wait_s": 10.0, "rep_s": 0.27},
    },
}


def sequence_timebase(seq):
    """(s/div, position) for REFerence LEFT covering a spin-echo record:
    from `before_ms` before the trigger to `after_ms` after the second
    motion ends (spacing + motion), on 10 divisions; the scale rounded UP to
    two significant figures (the MSO-X takes fine values: 1.5 ms/div has run
    on it)."""
    import math
    before = float(seq.get("before_ms", 12.0)) * 1e-3
    end = (float(seq.get("spacing_ms", 16.667)) + float(seq.get("motion_ms", 9.5))
           + float(seq.get("after_ms", 12.0))) * 1e-3
    span = before + end
    raw = span / 10.0
    e = math.floor(math.log10(raw)) - 1
    scale = math.ceil(raw / 10 ** e - 1e-9) * 10 ** e
    return scale, -before + scale          # LEFT: the record starts 1 div before


def all_presets(cfg):
    """Every preset by name, the built-ins with a sequence's timebase worked
    out from cfg['sequence'], then the user's."""
    out = copy.deepcopy(PRESETS)
    seq = cfg.get("sequence") or DEFAULTS["sequence"]
    for p in out.values():
        if p.get("sequence"):
            scale, pos = sequence_timebase(seq)
            p["scope"][":TIMebase:SCALe"] = f"{scale:.4E}"
            p["scope"][":TIMebase:POSition"] = f"{pos:.4E}"
    out.update(cfg.get("user_presets") or {})
    return out


DEFAULTS = {
    "scope_grab_path": os.path.join(PROJECTS, "scope-grab-multi", "scope_grab.py"),
    # EOM-ILC: the target-correction format, the Trek limits and the line-
    # ripple fit come from its eomilc package; the AWG driver from the 4063B
    # repo (bias-point measurements). Both loaded by path, like Scope Grab.
    "eomilc_path": os.path.join(PROJECTS, "EOM-ILC"),
    "awg_path": os.path.join(PROJECTS, "BK4063B-AWG-GUI", "bk4063b.py"),
    # the SRS DS345 (light gate): its Ds345 class from the DS345 panel, by path
    "ds345_path": os.path.join(PROJECTS, "DS345-AWG-GUI", "ds345_awg_gui.py"),
    "scope_model": "msox2014a",
    "scope_addr": "",
    # the BK Precision 4063B's VISA resource; blank = the first 4063B found
    "awg_addr": "",
    # the DS345's GPIB resource; blank = the remembered one, else a bus scan
    "ds345_addr": "",
    "ell_port": "COM3",
    "ell_address": "0",
    # Mount angle (deg) that is analyzer 0. By the campaign's convention 0 is
    # aligned with the polarization at the EO zero - set it from a scan with
    # "Set zero from rest", never from the engraving on the mount.
    "ell_zero_deg": 0.0,
    "simulate": False,
    # connect the scope and the ELL14 when the window opens (never the AWG)
    "autoconnect": True,
    "channels": {
        "1": {"role": "PD", "name": "Analyzer PD"},
        "2": {"role": "off", "name": ""},
        "3": {"role": "MonX1", "name": "Trek monitor X1"},
        "4": {"role": "MonX2", "name": "Trek monitor X2"},
    },
    "outdir": os.path.join(PROJECTS, "scope_data", "polarimetry"),
    # bias points (rampol.bias): the AWG holds the EOMs at fixed rotations
    "bias": {"biases": "0:180:15", "order": "up", "split": 0.5, "shots": 8,
             "null_half_deg": 3.0, "null_points": 9, "hold_ms": 8.0,
             "settle_ms": 4.0, "name": "bias",
             # X1 / X2 held separately (pairs or a grid) - these win over
             # biases x split when x1 is given; the null predicted from the
             # point before; the azimuth tracked at null +- 45 deg for
             # track_ms after the fall (rampol.bias.PLAN)
             "x1": "", "x2": "", "how": "pairs", "predict_null": True,
             "track": True, "track_ms": 150.0},
    # AWG mode (rampol.awg): the 4063B drives the Treks, CH1 -> X1, CH2 -> X2.
    # idle1/2 blank = from the ILC state files' first sample (the learned trim)
    "awg": {"source": "ramp", "rotation": 45.0, "split": 0.5, "edge": "cosine",
            # the record is lead + rise + hold + fall + after (tail_ms): the
            # defaults make it the ILC's own 11 ms (5501 points at 2 us)
            "lead_ms": 0.5, "rise_ms": 1.0, "hold_ms": 8.0, "fall_ms": 1.0,
            "tail_ms": 0.5, "dt_us": 2.0, "idle1": "", "idle2": "",
            # the scope while the AWG plays (Load, Find in the hold): from
            # scope_before_ms before the trigger to scope_after_ms after the
            # AWG record ends - set apart from the AWG record itself
            "scope_before_ms": 0.2, "scope_after_ms": 0.0,
            # a sequence of ramps with their own X1 / X2 end points (deg),
            # one ramp scan each, interleaved with the analyzer angles (the
            # Ramp scan tab's); settle after each live change of waveform
            "seq_x1": "45", "seq_x2": "0:90:45", "seq_how": "pairs",
            "seq_order": "interleaved (per angle)", "seq_settle_s": 1.0,
            # each member also measures its hold's null: seq_null_points
            # angles across +-seq_null_half_deg around the hold's crossed
            # angle (and the bright angle), skipped where the grid has one
            # within 1 deg. Crossed at rest = seq_crossed_deg (0: the Find
            # zero; the fit's psi_rest + 90 says what it really is);
            # the hold is crossed at seq_crossed_deg + seq_sense x (X1 + X2)
            # (sense -1 measured 7 Oct 2026). See scan.hold_angles.
            "seq_null_angles": True, "seq_null_half_deg": 3.0, "seq_null_points": 3,
            "seq_crossed_deg": 0.0, "seq_sense": -1.0,
            "file1": "", "file2": "", "trig_hz": 3.7,
            "settle_ms": 4.0, "fit_timebase": True, "shots": 8,
            # BOTH outputs: never switched off by the program (the X2 path's
            # FPGA/buffer stage drives high on a floating input; which output
            # meets it depends on the cabling) - the end of anything is 'park'
            "never_float": True,
            # a waveform must have been played into the scope (dry run) before
            # it may drive the Treks; where the AWG BNCs go for that
            "require_dry_run": True, "dry_ch1": "3", "dry_ch2": "4", "dry_shots": 4},
    # the EOM voltage chain (rampol.calib): AWG V -> monitor V -> kV -> deg
    "calibration": {"EO1": {"gain": 0.5594, "mon_per_kv": 1.0, "v90_kv": 5.1283},
                    "EO2": {"gain": 0.5924, "mon_per_kv": 1.0, "v90_kv": 5.1374},
                    "source": "1 Sep 2026 optical calibration", "date": "2026-09-01"},
    # Find angle: min/max transmission in a window, or held by the AWG
    "find": {"kind": "min", "light": "record window", "window": "-10:-0.5",
             "line_hz": 60.0, "step": 10.0, "half": "", "points": "", "shots": "8",
             # write the selected preset and run the scan's settings check first
             "use_preset": True,
             # the timebase zoomed onto the window while measuring, then put back
             "zoom": True,
             # measure after / reuse latest / none, at every V/div the readings used
             "dark_mode": "none", "background_mode": "measure after"},
    # the ILC target comparison (rampol.ilc_target)
    "ilc": {"x1": os.path.join(PROJECTS, "EOM-ILC", "run", "drive_P92PX1H.state.npz"),
            "x2": os.path.join(PROJECTS, "EOM-ILC", "run", "drive_P92PX2A.state.npz"),
            "f_cut": 2000.0, "pd_delay_us": 0.0, "split": 0.5, "line_ref": ""},
    "scan_name": "scan",
    "preset": "Spin echo",
    # the spin-echo sequence the "Spin echo" preset's record covers (ms):
    # the legs' spacing (the echo time), one motion's length, and how much of
    # the record to keep before the trigger and after the second motion ends.
    # The ILC-target comparison takes its leg gap from here too.
    "sequence": {"spacing_ms": 16.667, "motion_ms": 9.5, "before_ms": 12.0,
                 "after_ms": 12.0},
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
        # before the angles: measure / reuse latest (same PD V/div and offset,
        # from another scan) / none. Background = beam blocked (subtracted when
        # there is one); dark = PD covered (shows the stray light apart)
        "dark_mode": "none", "bg_mode": "measure",
        # with both measured: each also read at this fine PD V/div, where
        # background - dark is the stray light to ~0.05 mV (at 1 V/div each
        # is +-1.3 mV); with the background reused, the latest such pair is
        "stray_on": True, "stray_vdiv": 0.005,
        "backoff_deg": 3.0,       # approach every angle from below by this much
        "points": 100000,         # single-shot readout points
        "wait_s": 30.0,           # trigger stall limit (> the repetition period)
        "rep_s": 10.0,            # trigger period, for the time estimate only
    },
    # The DS345 gating the light (rampol/ds345.py). Volts AT THE MODULATOR
    # (Hi-Z: the DS345 is programmed half). load_min_v / load_max_v: what the
    # modulator input may see - set them to its rating; a gate outside is
    # refused. idle: the light's level outside the windows and between
    # bursts; windows (ms from the trigger) take the other level.
    # This bench (9 Oct 2026): the DS345 drives the home-built VCO/VCA AOM
    # controller's input, which takes up to ~4.2 V (2.1 V programmed).
    "ds345": {"on_v": 4.2, "off_v": 0.0, "load_min_v": 0.0, "load_max_v": 4.2,
              "idle": "on", "windows": "", "edge_us": 20.0, "record_ms": 0.0,
              "dry_ch": 2, "dry_shots": 4, "margin_ms": 0.3, "recover_ms": 0.5,
              "gate_refine": False},
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
        # the analyzer's own extinction ratio at 843 nm: Thorlabs' LPVIS100
        # data (lpvis_te_xls.xls) gives 1.37e8 at 840 nm and 1.46e8 at 844 nm,
        # transmission 80.6 %. The 1e4 used before 6 Oct 2026 was the sheet's
        # minimum over 550-1500 nm, not the value at our wavelength.
        "polarizer_er": 1.4e8,
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
            cfg = _merge(DEFAULTS, json.load(fh))
    except (OSError, ValueError):
        return copy.deepcopy(DEFAULTS)
    # the spin-echo preset was renamed 7 Oct 2026 (it is generic now)
    if cfg.get("preset") == OLD_SPIN_ECHO:
        cfg["preset"] = SPIN_ECHO
    # a config saved before 6 Oct 2026 carries the old wide-band floor
    if cfg["analysis"].get("polarizer_er") == 1.0e4:
        cfg["analysis"]["polarizer_er"] = DEFAULTS["analysis"]["polarizer_er"]
    return cfg


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

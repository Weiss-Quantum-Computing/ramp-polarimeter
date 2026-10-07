"""The B&K 4063B driving the Treks: waveforms in AWG volts per channel, the
checks before anything is played, and a session that knows which outputs it
switched on.

CH1 -> Trek X1 -> EO1, CH2 -> Trek X2 -> EO2, the same wiring and the same
upload path as EOM-ILC (its ilc_bench.upload_drive and check_awg_channel, its
Trek limit check). What this module adds, and why (review of 6 Oct 2026
against EOM-ILC and bk4063b.py):

* Idle level. Between bursts the AWG holds the record's FIRST sample. File
  zero parks the EOMs at -9 V (X1) / -41 V (X2) because of the generator's
  zero-code error and the network's offsets; EOM-ILC's keepers start at a
  learned idle trim (X1 +20..+26 mV, X2 +78..+81 mV). A waveform here is
  idle + amplitude x shape with BOTH ends exactly at idle, idle capped at
  +-100 mV (EOM-ILC's Limits.idle_awg).
* Record length. In DDS mode the whole record is resampled into one FRQ
  period, FRQ = 1/(N dt). A ramp's record is lead + rise + hold + fall +
  after; the defaults make it EOM-ILC's: 11 ms at 2 us, 5501 points, 90.893
  Hz. N <= 16384 (datasheet; 5501 is the most proven). A different length
  needs the channels set up again for the new FRQ, and setting up stops
  the burst for a moment (BSWV switches burst off; the mode block puts it
  back), so the channel free-runs whatever it holds. With the outputs live
  (never-float) the session therefore first plays a flat idle waveform on
  the CURRENT grid - free-running, it is still idle - then sets the channels
  up, then uploads. Park and the end of anything go back to the ILC's 11 ms
  grid, so the ILC panel's FRQ check passes afterwards. This live change has
  not been tried on the bench (7 Oct 2026): the dry run does it first, into
  the scope.
* Names. The 4063B cannot delete stored waveforms over SCPI and locks its
  panel past 11-character names. A waveform's name is a hash of its samples
  (RP<ch><8 hex>): the same waveform is selected again, not stored again.
* Outputs. Both channels are checked before anything is written; ON is set
  one channel at a time with the bookkeeping first, and read back; every
  exit switches off whatever was touched. A channel that is ON but was not
  switched on here belongs to someone else (the ILC panel): refused.
* Never float (default ON, BOTH channels - which output reaches which
  stage depends on the day's cabling). The X2 drive path's FPGA/buffer
  stage drives its output high (-4 to -5.7 kV) when its input floats (3 Sep
  2026; pull-down not fitted). Under the rule an output, once on, is never
  switched off by the program: waveform changes are made live, and the end
  of anything is 'park' - an idle-level waveform with the outputs ON.
  Switching OFF is then an explicit, confirmed act. With the rule off the
  outputs go OFF for changes and at the end.
* Dry run. Before a waveform drives the Treks it is played into two scope
  channels (the AWG's BNCs moved from the Trek inputs to the scope) and
  compared with what was meant: which output is which (each played alone,
  the other at idle), delay, time scale (a wrong FRQ plays the record
  stretched), gain, offset, shape, the idle level, and that every shot is
  triggered. A waveform that passed is remembered by name (the hash of its
  samples); with 'require a dry run' on, nothing else is allowed onto live
  outputs or switched on.
* Limits. A synthetic ramp is checked as check_limits(u, u x gain); an ILC
  drive against its own state's target (the ILC's way - a keeper fails the
  2 mA current check computed from u x gain but passes against its target).
"""
import hashlib
import os
import re
import threading

import numpy as np

from . import bias as biasmod

FULL_SCALE = 10.0          # V: AMP 20 Vpp, OFST 0, samples u / FULL_SCALE uploaded as-is
AWG_CAP = 9.6              # V: this program's cap per channel (the 4063B gives +-10 V;
                           # a margin under that rail)
MAX_PTS = 16384            # 4063B arb memory (datasheet; unprobed above 5501)
PROVEN_PTS = 5501
IDLE_CAP = 0.100           # V: EOM-ILC Limits.idle_awg
CHANNELS = {"EO1": 1, "EO2": 2}

from .config import DEFAULTS as _CFG  # noqa: E402

DEFAULTS = _CFG["awg"]


class Wave:
    """One waveform per channel on a common time grid. u: {EO1, EO2: AWG
    volts}. target: {name: monitor volts} when known (an ILC drive's state),
    for the limit check. hold: (t0, t1) of the hold in s, or None. rotation:
    the commanded rotation of a ramp, deg."""

    def __init__(self, t, u, dt, label, hold=None, target=None, rotation=None,
                 source="ramp", files=None):
        self.t, self.u, self.dt = np.asarray(t, float), u, float(dt)
        self.label, self.hold, self.rotation = label, hold, rotation
        self.target = target or {}
        self.source, self.files = source, files or {}
        self.ends = None          # a sequence ramp: {EO1, EO2} end points, deg

    @property
    def n(self):
        return len(self.t)

    @property
    def period(self):
        return self.n * self.dt

    def idle(self):
        return {k: float(v[0]) for k, v in self.u.items()}


# ------------------------------------------------------------------ building
def edge(n, kind="cosine"):
    """0 -> 1 over n samples (the first 0, the last just under 1)."""
    k = np.arange(n) / max(n, 1)
    if kind == "linear":
        return k
    return 0.5 - 0.5 * np.cos(np.pi * k)


ILC_RECORD_MS, ILC_DT_US = 11.0, 2.0      # EOM-ILC's record: 5501 points, 90.893 Hz


def record_ms(p):
    """A ramp's record length (ms): lead + rise + hold + fall + after."""
    p = dict(DEFAULTS, **(p or {}))
    return sum(float(p[k]) for k in ("lead_ms", "rise_ms", "hold_ms", "fall_ms", "tail_ms"))


def ramp_hold(rotation_deg, p, idle=None, chan=None, ends=None):
    """Idle -> the commanded rotation -> idle: `lead` at idle, `rise` edge,
    hold, `fall` edge, `after` (tail_ms) at idle - the record is their sum,
    on a `dt_us` grid (N = record / dt + 1, EOM-ILC's convention: 11 ms ->
    5501). The rotation is split split : 1 - split between EO1 and EO2
    (awg_volts) and is relative to idle. Both ends are exactly idle.
    ends={'EO1': deg, 'EO2': deg}: each crystal's own end point instead
    (rotation and split are then ignored)."""
    p = dict(DEFAULTS, **(p or {}))
    idle = idle or {"EO1": 0.0, "EO2": 0.0}
    split = float(p["split"])
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"split {split:g} must be between 0 and 1")
    dt = float(p["dt_us"]) * 1e-6
    if dt <= 0:
        raise ValueError("dt must be > 0")
    seg = [float(p[k]) * 1e-3 for k in ("lead_ms", "rise_ms", "hold_ms", "fall_ms", "tail_ms")]
    if min(seg) < 0:
        raise ValueError("lead, rise, hold, fall and after must be >= 0")
    if seg[0] <= 0 or seg[4] <= 0:
        raise ValueError("lead and after must be > 0: the record starts and ends at idle")
    nl, nr, nh, nf, na = (int(round(x / dt)) for x in seg)
    n = nl + nr + nh + nf + na + 1
    prof = np.zeros(n)
    prof[nl:nl + nr] = edge(nr, p["edge"])
    prof[nl + nr:nl + nr + nh] = 1.0
    if nf:
        # the fall's first sample is the hold level, its last just above 0
        prof[nl + nr + nh:nl + nr + nh + nf] = 1.0 - edge(nf, p["edge"])
    prof[[0, -1]] = 0.0
    kw = {"chan": chan} if chan else {}
    if ends is not None:
        e1, e2 = float(ends["EO1"]), float(ends["EO2"])
        amps = {"EO1": biasmod.awg_volts(e1, 1.0, **kw)["EO1"],
                "EO2": biasmod.awg_volts(e2, 0.0, **kw)["EO2"]}
        what = f"X1 {e1:g} / X2 {e2:g} deg"
        rotation_deg = e1 + e2
    else:
        amps = biasmod.awg_volts(rotation_deg, split, **kw)
        what = f"{rotation_deg:g} deg"
    u = {k: float(idle.get(k, 0.0)) + amps[k] * prof for k in CHANNELS}
    t = np.arange(n) * dt
    hold = ((nl + nr) * dt, (nl + nr + nh) * dt)
    w = Wave(t, u, dt, f"ramp to {what} ({p['edge']} edges "
             f"{p['rise_ms']:g}/{p['fall_ms']:g} ms, hold {p['hold_ms']:g} ms)",
             hold=hold, rotation=float(rotation_deg), source="ramp")
    w.ends = None if ends is None else {"EO1": e1, "EO2": e2}
    return w


def parse_ends(x1, x2, how="pairs"):
    """The (X1, X2) end points of a sequence, deg. Each list is
    start:stop:step or comma-separated. 'pairs': taken together (a list of
    one goes with every entry of the other); 'grid': every X1 with every
    X2, X1 the outer loop."""
    a, b = biasmod.parse_biases(x1), biasmod.parse_biases(x2)
    if not a or not b:
        raise ValueError("give X1 and X2 end points (deg)")
    if how == "grid":
        return [(p, q) for p in a for q in b]
    if len(a) == 1:
        a = a * len(b)
    if len(b) == 1:
        b = b * len(a)
    if len(a) != len(b):
        raise ValueError(f"{len(a)} X1 and {len(b)} X2 end points: pairs need as many of "
                         f"each (or one of either), or choose 'grid'")
    return list(zip(a, b))


def idle_flat(idle, like):
    """The park waveform: idle on both channels, on `like`'s grid, so the
    channel's FRQ stays right for the next real waveform."""
    return Wave(like.t, {k: np.full(like.n, float(idle.get(k, 0.0))) for k in CHANNELS},
                like.dt, "park at idle", source="park")


def load_drive(path):
    """An EOM-ILC drive CSV (run/drive_<stem>_iNN.csv): '#' comment lines, a
    header 'time_us,voltage_V', then AWG volts. Header-less files are
    refused (EOM-ILC's rule: a bare column is ambiguous). Returns (t s, u V)."""
    rows, header = [], None
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if header is None:
                header = [x.strip().lower() for x in s.split(",")]
                if not any(h.startswith("time") for h in header):
                    raise ValueError(f"{os.path.basename(path)}: no 'time_us,voltage_V' "
                                     f"header - not a drive file")
                continue
            rows.append([float(x) for x in s.split(",")[:2]])
    a = np.asarray(rows, float)
    if a.ndim != 2 or len(a) < 2:
        raise ValueError(f"{os.path.basename(path)}: no data")
    tcol = header[0]
    t = a[:, 0] * (1e-6 if "us" in tcol else (1e-3 if "ms" in tcol else 1.0))
    u = a[:, 1]
    if "voltage" not in header[1] and "v" not in header[1]:
        raise ValueError(f"{os.path.basename(path)}: second column {header[1]!r} is not volts")
    if np.max(np.abs(u)) > FULL_SCALE:
        raise ValueError(f"{os.path.basename(path)} peaks at {np.max(np.abs(u)):.2f} V: "
                         f"past the AWG's {FULL_SCALE:g} V - a TARGET file (EOM volts)?")
    return t, u


def drive_state(path):
    """The state file beside an ILC drive (drive_<stem>_iNN.csv ->
    drive_<stem>.state.npz): (target monitor volts, stem) or (None, stem)."""
    base = os.path.basename(path)
    m = re.match(r"^drive_(.+?)(?:_i\d+)?\.csv$", base)
    stem = m.group(1) if m else None
    if stem:
        sp = os.path.join(os.path.dirname(path), f"drive_{stem}.state.npz")
        if os.path.exists(sp):
            z = np.load(sp, allow_pickle=True)
            return np.asarray(z["target"], float), stem
    return None, stem


def from_files(path1, path2):
    """Two ILC drives (X1, X2) as one Wave. Same N and dt on both (the
    channels share a trigger and play side by side)."""
    out, tgt, files = {}, {}, {}
    grid = None
    for name, path in (("EO1", path1), ("EO2", path2)):
        if not path:
            raise ValueError(f"no drive file for {name}")
        t, u = load_drive(path)
        dt = float(np.median(np.diff(t)))
        if grid is None:
            grid = (t, dt)
        elif len(t) != len(grid[0]) or abs(dt - grid[1]) > 1e-9:
            raise ValueError(f"the two drives differ: {len(grid[0])} points at "
                             f"{grid[1]*1e6:.3f} us vs {len(t)} at {dt*1e6:.3f} us")
        out[name] = u
        target, stem = drive_state(path)
        if target is not None and len(target) == len(u):
            tgt[name] = target
        files[name] = os.path.abspath(path)
    return Wave(grid[0] - grid[0][0], out, grid[1],
                f"ILC drives {os.path.basename(path1)} + {os.path.basename(path2)}",
                target=tgt, source="files", files=files)


def idle_from_states(paths):
    """{EO1, EO2: idle V} from EOM-ILC state files' u[0] (the learned trim),
    clipped to the idle cap; 0 where a file is missing."""
    out = {}
    for name, path in zip(CHANNELS, paths):
        v = 0.0
        try:
            if path and os.path.exists(path):
                v = float(np.load(path, allow_pickle=True)["u"][0])
        except Exception:
            v = 0.0
        out[name] = float(np.clip(v, -IDLE_CAP, IDLE_CAP))
    return out


# ------------------------------------------------------------------- checks
def predict(wave, chan=None):
    """Monitor volts (relative to idle) and the rotation they give, deg:
    rotation per crystal = 90 x monitor / V90, summed."""
    chan = chan or biasmod.CHAN
    mon = {k: chan[k]["gain"] * (wave.u[k] - wave.u[k][0]) for k in CHANNELS}
    rot = sum(90.0 * mon[k] / chan[k]["v90"] for k in CHANNELS)
    return mon, rot


def max_deg(idle=None, chan=None, cap=AWG_CAP):
    """{EO1, EO2}: the most rotation each crystal can be driven to (deg)
    with its AWG channel at most `cap` volts - its idle level allowed for
    (the rotation is on top of idle). With the 1 Sep 2026 calibration that
    is ~94 deg on X1 (9.82 deg/V) and ~99.6 deg on X2 (10.38 deg/V): a
    rotation of 180 deg needs both crystals, and no single one gets there."""
    chan = chan or biasmod.CHAN
    idle = idle or {}
    out = {}
    for k in CHANNELS:
        per_deg = abs(biasmod.awg_volts(1.0, 1.0 if k == "EO1" else 0.0, chan)[k])
        out[k] = max(cap - abs(float(idle.get(k, 0.0))), 0.0) / per_deg
    return out


def share_range(rotation, idle=None, chan=None):
    """(lo, hi): the X1 shares that keep both crystals within reach for
    `rotation` deg, or None when the pair cannot reach it at all."""
    m = max_deg(idle, chan)
    r = abs(float(rotation))
    if r == 0:
        return 0.0, 1.0
    lo, hi = max(0.0, 1.0 - m["EO2"] / r), min(1.0, m["EO1"] / r)
    return (lo, hi) if lo <= hi + 1e-12 else None


def within_reach(ends, idle=None, chan=None):
    """True when each crystal's end point (deg) is within its reach."""
    m = max_deg(idle, chan)
    return abs(float(ends["EO1"])) <= m["EO1"] + 1e-9 and abs(float(ends["EO2"])) <= m["EO2"] + 1e-9


def check(wave, eomilc=None, trig_hz=None, chan=None):
    """[(level, text)], level FAIL | WARN | INFO. A FAIL stops an upload."""
    chan = chan or biasmod.CHAN
    out = []
    if wave.n > MAX_PTS:
        out.append(("FAIL", f"{wave.n} points: past the 4063B's {MAX_PTS}"))
    elif wave.n > PROVEN_PTS:
        out.append(("WARN", f"{wave.n} points: more than the {PROVEN_PTS} played so far"))
    for k, u in wave.u.items():
        pk = float(np.max(np.abs(u)))
        if pk > AWG_CAP:
            reach = max_deg(wave.idle(), chan)[k]
            out.append(("FAIL", f"{k}: {pk:.3f} V at the AWG, past the {AWG_CAP:g} V cap (the "
                                f"4063B gives +-10 V) - {k} reaches at most {reach:.1f} deg; "
                                f"put more of the rotation on the other crystal"))
        for end, v in (("first", u[0]), ("last", u[-1])):
            if abs(v) > IDLE_CAP:
                out.append(("FAIL", f"{k}: {end} sample {v*1e3:+.0f} mV - past the "
                                    f"{IDLE_CAP*1e3:.0f} mV idle cap (the AWG holds the "
                                    f"first sample between bursts)"))
        if abs(u[-1] - u[0]) > 0.005 and wave.source == "ramp":
            out.append(("WARN", f"{k}: ends differ by {(u[-1]-u[0])*1e3:.1f} mV"))
    if trig_hz:
        gap = 1.0 / float(trig_hz)
        if wave.period > 0.8 * gap:
            out.append(("FAIL", f"record {wave.period*1e3:.2f} ms is over 80 % of the "
                                f"{gap*1e3:.0f} ms trigger period: bursts would skip "
                                f"triggers the scope still takes"))
        _, rot = predict(wave, chan)
        hot = float(np.sum(np.abs(rot) > 10.0)) * wave.dt
        if hot / gap > 0.05:
            out.append(("WARN", f"{hot*1e3:.1f} ms of every {gap*1e3:.0f} ms above 10 deg "
                                f"({hot/gap:.0%} duty at kV): far more than the "
                                f"experiment's ramps; crystal memory and heating were "
                                f"seen with less. Watch the rest level drift."))
    if eomilc is not None:
        from eomilc.config import CHANNELS as ECH
        from eomilc.ilc import check_limits
        for k, u in wave.u.items():
            ch = ECH[k]
            if k in wave.target:
                v = wave.target[k]
                how = "against its ILC target"
            else:
                v = u * chan[k]["gain"]
                how = "as u x gain"
            rep = check_limits(u, v, wave.dt, ch, ch.limits)
            msg = "; ".join(rep.messages) if rep.messages else "within the Trek limits"
            out.append(("INFO" if rep.ok else "FAIL", f"{k} ({how}): {msg}"))
            # the gains and V90 this program assumes, against EOM-ILC's
            g = getattr(ch, "cmd_hv_gain_meas", None)
            v90 = getattr(ch, "v90_hv", None)
            if g and abs(g / chan[k]["gain"] - 1) > 1e-3:
                out.append(("WARN", f"{k}: gain {chan[k]['gain']} here, {g} in EOM-ILC"))
            if v90 and abs(v90 / 1000.0 / chan[k]["v90"] - 1) > 1e-3:
                out.append(("WARN", f"{k}: V90 {chan[k]['v90']} V here, {v90/1000:.4f} in EOM-ILC"))
    else:
        out.append(("WARN", "Trek limit check not run (EOM-ILC not loaded)"))
    return out


def worst(found):
    levels = [lv for lv, _ in found]
    return "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "OK")


def wave_name(u, ch):
    """RP<ch><8 hex of the 16-bit codes>: 11 characters, the 4063B's cap; the
    same samples give the same name."""
    codes = np.round(np.clip(np.asarray(u, float) / FULL_SCALE, -1, 1) * 32767).astype("<i2")
    return f"RP{ch}{hashlib.sha1(codes.tobytes()).hexdigest()[:8]}"


def timebase_for(wave, before_ms=0.2, after_ms=0.0):
    """(scale s/div, position s) for REFerence LEFT: from `before_ms` before
    the trigger to `after_ms` after the record ends (defaults: the whole
    record from 0.2 ms before, as the bias run sets it)."""
    import math
    before, after = float(before_ms) * 1e-3, float(after_ms) * 1e-3
    # rounded UP to two significant figures, as the spin-echo preset's: a
    # 1-2-5 step would nearly double a 29 ms span (the MSO-X takes 2.9 ms/div)
    raw = (before + wave.period * 1.02 + after) / 10
    e = math.floor(math.log10(raw)) - 1
    div = math.ceil(raw / 10 ** e - 1e-9) * 10 ** e
    return div, -before + div


def ilc_grid(idle):
    """The park waveform on EOM-ILC's grid (11 ms at 2 us): parked there, the
    channels are at the ILC's FRQ again."""
    dt = ILC_DT_US * 1e-6
    n = int(round(ILC_RECORD_MS * 1e-3 / dt)) + 1
    return Wave(np.arange(n) * dt, {k: np.full(n, float(idle.get(k, 0.0))) for k in CHANNELS},
                dt, "park at idle (ILC record)", source="park")


# ------------------------------------------------------------------- session
def names(wave):
    """(CH1 name, CH2 name) of a wave: its identity for the dry-run record."""
    return (wave_name(wave.u["EO1"], 1), wave_name(wave.u["EO2"], 2))


class NotVerified(RuntimeError):
    pass


class Session:
    """The AWG as this window holds it. One lock serialises every AWG call
    (Outputs OFF can be pressed while a measurement runs on the worker).
    `owned` is the set of channels this session switched ON. never_float and
    require_dry_run: see the module docstring. `verified` holds the names of
    waveforms that passed a dry run; `dry` is set while one runs."""

    def __init__(self, awg, ib=None, log=print, never_float=True, require_dry_run=True):
        self.awg, self.ib, self.log = awg, ib, log
        self.lock = threading.RLock()
        self.owned = set()
        self.wave = None          # what is loaded (None: unknown / replaced)
        self.parked = False
        self._period = {}         # without ilc_bench (simulator): the FRQ set here
        self.never_float = never_float
        self.require_dry_run = require_dry_run
        self.verified = {}        # names(wave) -> dry-run summary
        self.dry = False

    def is_verified(self, wave):
        return wave.source == "park" or names(wave) in self.verified

    def _gate(self, wave, what):
        if self.require_dry_run and not self.dry and not self.is_verified(wave):
            raise NotVerified(
                f"{what}: '{wave.label}' has not passed a dry run on the scope. Dry-run "
                f"it first (AWG outputs to the scope, Treks disconnected), or untick "
                f"'require a dry run'.")

    # -- state --------------------------------------------------------------
    def outputs(self):
        with self.lock:
            return {ch: bool(self.awg.is_on(ch)) for ch in CHANNELS.values()}

    def foreign_on(self):
        return [ch for ch, on in self.outputs().items() if on and ch not in self.owned]

    # -- loading a waveform -----------------------------------------------
    def setup_ok(self, ch, period):
        """[] when the channel is already set up for this record (FRQ, AMP,
        OFST, DDS), else the problems (needs EOM-ILC's ilc_bench)."""
        if self.ib is None:
            return [] if abs(self._period.get(ch, -1.0) - period) < 1e-12 else ["FRQ"]
        problems, _notes = self.ib.check_awg_channel(self.awg, ch, full_scale=FULL_SCALE,
                                                     expect_period=period)
        return problems

    def _setup(self, ch, period):
        blocks = {"OUTP": {"LOAD": "HZ"}, "SRATE": {"MODE": "DDS"},
                  "BSWV": {"WVTP": "ARB", "FRQ": 1.0 / period, "AMP": 2 * FULL_SCALE,
                           "OFST": 0},
                  "MODE": ("Burst", {"GATE_NCYC": "NCYC", "TIME": 1, "TRSR": "EXT"})}
        missed = self.awg.apply_channel(ch, blocks, log=lambda m: None)
        self._period[ch] = period
        if missed:
            raise RuntimeError(f"AWG CH{ch} did not take {missed} - without burst the "
                               f"channel free-runs; nothing uploaded")
        if self.ib is not None:
            problems = self.setup_ok(ch, period)
            if problems:
                raise RuntimeError(f"AWG CH{ch} setup: " + "; ".join(problems))

    def _put(self, ch, u):
        """Select the waveform if the generator already holds it, else
        upload it (EOM-ILC's fixed mapping)."""
        name = wave_name(u, ch)
        try:
            stored = set(self.awg.list_waveforms(user_only=True))
        except Exception:
            stored = set()
        if name in stored:
            self.awg.write(f"C{ch}:ARWV NAME,{name}")
            if not self.dry:
                self.log(f"  CH{ch}: {name} (already stored) selected")
            return name
        if self.ib is not None:
            self.ib.upload_drive(self.awg, ch, name, u, FULL_SCALE)
        else:
            self.awg.upload_arb(ch, name, np.asarray(u) / FULL_SCALE, normalize=False)
        self.log(f"  CH{ch}: {name} uploaded ({len(u)} points)")
        return name

    def load(self, wave, keep_on=False):
        """Set the channels up (only where they are not already right for
        this record length) and put the waveform on both. A channel ON that
        this session did not switch on is refused. Live outputs are switched
        OFF for the change unless keep_on (park policy: the X2 stage must not
        float) - then the upload happens live, as EOM-ILC's does."""
        with self.lock:
            foreign = self.foreign_on()
            if foreign:
                raise RuntimeError(
                    f"AWG CH{', CH'.join(map(str, foreign))} is ON and this window did not "
                    f"switch it on - another program (the ILC panel?) may be driving the "
                    f"Treks. Switch it off there first.")
            live = [ch for ch, on in self.outputs().items() if on]
            keep_on = keep_on or self.never_float
            if live:
                self._gate(wave, "Not loaded onto live outputs")
            if live and not keep_on:
                self.off(force=True)
                live = []
            need = [ch for ch in CHANNELS.values() if self.setup_ok(ch, wave.period)]
            if need and live:
                # Setting up stops the burst for a moment and the channel
                # free-runs what it holds: make that a flat idle first, on
                # the grid the channel is at now (module docstring)
                cur = self.wave
                if cur is None:
                    raise RuntimeError(
                        "the record length changes with the outputs live, and what the AWG "
                        "holds now is unknown to this window - so it cannot be put to idle "
                        "first. Park (or Load something) on the current record first.")
                if cur.source != "park":
                    flat = idle_flat(cur.idle(), cur)
                    for name, ch in CHANNELS.items():
                        self._put(ch, flat.u[name])
                    self.wave, self.parked = flat, True
                if not self.dry:
                    self.log(f"  record {cur.period*1e3:.3f} -> {wave.period*1e3:.3f} ms with "
                             f"the outputs live: held at idle while the channels are set up")
            for ch in need:
                self._setup(ch, wave.period)
            names = {}
            if live and not self.dry:
                self.log("  outputs live: the change can land mid-burst (as EOM-ILC's uploads)")
            for name, ch in CHANNELS.items():
                names[name] = self._put(ch, wave.u[name])
            self.wave = wave
            self.parked = wave.source == "park"
            return names

    # -- outputs --------------------------------------------------------------
    def on(self):
        """Both outputs ON, one at a time, bookkeeping first, read back."""
        with self.lock:
            if self.wave is None:
                raise RuntimeError("nothing loaded on the AWG by this window - Load first")
            self._gate(self.wave, "Outputs not switched on")
            try:
                for ch in CHANNELS.values():
                    self.owned.add(ch)
                    self.awg.set_output(ch, True)
                state = self.outputs()
            except Exception:
                # CH1 may be live with CH2 refused: never leave half of it on
                self.off(force=True)
                raise
            if not all(state.values()):
                bad = [ch for ch, on in state.items() if not on]
                self.off(force=True)
                raise RuntimeError(f"CH{bad} did not switch on - both switched OFF")
            self.log("AWG outputs ON (CH1 -> X1, CH2 -> X2)")

    def off(self, force=False):
        """Both outputs OFF, each tried whatever the other does. Under the
        never-float rule only with force=True (the window asks first)."""
        if self.never_float and not force:
            raise RuntimeError("never-float rule: switching OFF leaves the inputs floating - "
                               "park instead, or confirm OFF")
        errs = []
        with self.lock:
            for ch in CHANNELS.values():
                try:
                    self.awg.set_output(ch, False)
                except Exception as exc:
                    errs.append(f"CH{ch}: {exc}")
            self.owned.clear()
        if errs:
            self.log("  could not switch off " + "; ".join(errs))
        else:
            self.log("AWG outputs OFF")
        return not errs

    def park(self, like=None, ilc=True):
        """An idle-level waveform with the outputs ON (the X2 FPGA stage must
        not see a floating input) - on EOM-ILC's 11 ms grid (ilc=True), so
        the ILC panel finds its FRQ, else on `like`'s grid."""
        with self.lock:
            base = like or self.wave
            if base is None:
                raise RuntimeError("no waveform to take the grid and idle from")
            self.load(ilc_grid(base.idle()) if ilc else idle_flat(base.idle(), base),
                      keep_on=True)
            if not all(self.outputs().values()):
                self.on()
            self.log("AWG parked: idle level, outputs ON")

    def end(self, policy=None):
        """The end of an operation: park under the never-float rule (or
        policy 'park'), else OFF. Under the rule a failed park is reported
        and the outputs are left as they are - switching off is the act the
        rule forbids."""
        park = self.never_float or policy == "park"
        if park and self.wave is not None:
            try:
                self.park()
                return
            except Exception as exc:
                if self.never_float:
                    self.log(f"  PARK FAILED ({exc}) - outputs left as they are (never-"
                             f"float rule); switch off by hand only with the stage bypassed")
                    return
                self.log(f"  park failed ({exc}) - switching OFF instead")
        if park and self.wave is None and self.never_float:
            return                 # nothing of ours loaded: leave the outputs alone
        self.off(force=True)

    def forget(self):
        """Something else (a bias run) replaced the waveform."""
        self.wave = None


# ------------------------------------------------------------------- dry run
TOL = {"gain": 0.04,        # the MSO-X's own DC gain accuracy is +-3 % of full scale
       "stretch": 5e-4,     # a record played at the wrong FRQ is stretched
       "delay_us": 20.0,
       "shape": 0.015,      # rms residual / peak-to-peak (+ 3 mV for the scope)
       "idle_V": 0.10,      # the generator's zero-code error is -12 / -40 mV at 20 Vpp
       "jitter_us": 5.0}    # shot-to-shot delay spread: more = not triggered


def _model(t, u, dt, delay, stretch):
    """The AWG's output at scope time t: sample k at delay + k dt stretch, the
    first sample held outside the burst."""
    tu = (np.asarray(t) - delay) / stretch
    return np.interp(tu, np.arange(len(u)) * dt, u, left=u[0], right=u[0])


def _xcorr_delay(t, v, u, dt, search=(-100e-6, 300e-6)):
    """Coarse delay from the cross-correlation of the derivatives."""
    tt = np.arange(search[0], search[1], 0.5e-6)
    dv = np.gradient(v)
    best, arg = -np.inf, 0.0
    for d in tt[::4]:
        m = _model(t, u, dt, d, 1.0)
        c = float(np.dot(dv, np.gradient(m)))
        if c > best:
            best, arg = c, d
    for d in np.arange(arg - 2e-6, arg + 2e-6, 0.25e-6):
        m = _model(t, u, dt, d, 1.0)
        c = float(np.dot(dv, np.gradient(m)))
        if c > best:
            best, arg = c, d
    return arg


def compare(t, v, u, dt, shots=None):
    """Fit the captured trace v(t) (mean of the shots) to the intended u:
    v = gain x u((t - delay) / stretch) + offset. `shots` (k x n) gives the
    per-shot delay spread. Returns a dict of numbers and 'problems'."""
    from scipy.optimize import least_squares
    t, v, u = np.asarray(t, float), np.asarray(v, float), np.asarray(u, float)
    d0 = _xcorr_delay(t, v, u, dt)

    def lin(p):
        m = _model(t, u, dt, p[0], p[1])
        A = np.column_stack([m, np.ones_like(m)])
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return coef, v - A @ coef

    r = least_squares(lambda p: lin(p)[1], [d0, 1.0], x_scale=[1e-6, 1e-4],
                      bounds=([d0 - 50e-6, 0.5], [d0 + 50e-6, 2.0]))
    (gain, off), res = lin(r.x)
    delay, stretch = float(r.x[0]), float(r.x[1])
    ptp = float(np.ptp(u))
    rms = float(np.sqrt(np.mean(res ** 2)))
    pre = t < delay - 20e-6
    idle_meas = float(np.mean(v[pre])) if pre.sum() > 20 else None
    jit = None
    if shots is not None and len(shots) > 1 and ptp > 0.05:
        ds = [_xcorr_delay(t, s_, u, dt, (delay - 50e-6, delay + 50e-6)) for s_ in shots]
        jit = float(np.ptp(ds))
    out = {"delay_us": delay * 1e6, "stretch": stretch, "gain": float(gain),
           "offset_mV": float(off) * 1e3, "rms_mV": rms * 1e3, "ptp_V": ptp,
           "peak_err_mV": float(np.max(np.abs(res))) * 1e3,
           "idle_meant_V": float(u[0]), "idle_meas_V": idle_meas,
           "jitter_us": None if jit is None else jit * 1e6}
    probs = []
    if ptp > 0.05:
        if abs(gain - 1) > TOL["gain"]:
            probs.append(f"gain {gain:.4f} (x{gain:.2f}: a load or scaling error?)")
        if abs(stretch - 1) > TOL["stretch"]:
            probs.append(f"played {stretch:.5f} x as long as meant (FRQ wrong for this "
                         f"record?)")
        if abs(delay) * 1e6 > TOL["delay_us"]:
            probs.append(f"starts {delay*1e6:.1f} us after the trigger")
        if rms > TOL["shape"] * ptp + 3e-3:
            probs.append(f"shape off by {rms*1e3:.1f} mV rms ({rms/ptp:.1%} of the swing)")
        if jit is not None and jit * 1e6 > TOL["jitter_us"]:
            probs.append(f"shots start {jit:.1f} us apart: not triggered (free-running?)")
    if idle_meas is not None and abs(idle_meas - u[0]) > TOL["idle_V"]:
        probs.append(f"idle {idle_meas*1e3:+.0f} mV, meant {u[0]*1e3:+.0f} mV")
    out["problems"] = probs
    out["model"] = (delay, stretch, float(gain), float(off))
    return out


def solo(wave, which):
    """`wave` on channel `which` (EO1/EO2), the other at its idle level -
    to see which scope channel each output reaches without letting either
    float."""
    u = {k: (wave.u[k] if k == which else np.full(wave.n, float(wave.u[k][0])))
         for k in CHANNELS}
    return Wave(wave.t, u, wave.dt, f"{wave.label} on {which} only", hold=wave.hold,
                source="dry-solo")


def capture(link, scope_chs, shots, points, wait_s, cancelled):
    """{scope ch: (t, mean, shots k x n)} of `shots` single shots."""
    acc, tt = {ch: [] for ch in scope_chs}, {}

    def on_block(k, recs, hits):
        for ch, rec in recs.items():
            acc[ch].append(rec.v())
            tt[ch] = rec.t()
    link.acquire_blocks(list(scope_chs), "single", shots, shots, dither_codes=0,
                        points=points, wait_s=wait_s, cancelled=cancelled,
                        on_block=on_block)
    out = {}
    for ch in scope_chs:
        n = min(len(x) for x in acc[ch])
        stack = np.array([x[:n] for x in acc[ch]])
        out[ch] = (tt[ch][:n], stack.mean(axis=0), stack)
    return out


def _nice(x):
    return biasmod._nice_up(max(x, 1e-3))


def dry_run(sess, link, wave, wiring, shots=4, points=20000, wait_s=10.0,
            cancelled=None, log=print, identify=True):
    """Play `wave` into the scope and check it. wiring: {EO1: scope ch, EO2:
    scope ch} - where the AWG's CH1 / CH2 BNCs go for the dry run. The scope
    channels' V/div, offset and the timebase are set to show the record and
    put back afterwards; the trigger is left alone (the bench trigger the AWG
    bursts on). With `identify`, each output is first played alone (the
    other at idle) to see which scope channel it reaches. Returns the report;
    a pass is entered in sess.verified."""
    cancelled = cancelled or (lambda: False)
    sc = link.scope
    chs = [wiring["EO1"], wiring["EO2"]]
    if chs[0] == chs[1]:
        raise ValueError("the two AWG outputs need two different scope channels")
    saved_ch = link.channel_state(chs)
    tb_keys = (":TIMebase:SCALe", ":TIMebase:POSition", ":TIMebase:REFerence")
    saved_tb = {k: sc.get(k) for k in tb_keys}
    trig = (sc.get(":TRIGger:EDGE:SOURce") or "").strip().upper()
    report = {"label": wave.label, "names": list(names(wave)), "wiring": dict(wiring),
              "shots": shots, "trigger": trig, "steps": {}, "problems": []}
    if trig.startswith("LINE"):
        report["problems"].append("the scope triggers on LINE: the AWG bursts on the bench "
                                  "trigger, so the two are not in step - set EXT")
    sess.dry = True
    try:
        div, pos = timebase_for(wave)
        sc.put(":TIMebase:REFerence", "LEFT")
        sc.put(":TIMebase:SCALe", f"{div:.6g}")
        sc.put(":TIMebase:POSition", f"{pos:.6g}")
        # one setting for both channels, covering both outputs: a swapped
        # cable then shows the other output on screen instead of clipped
        allu = np.concatenate([wave.u[n] for n in CHANNELS])
        lo, hi = float(min(allu.min(), 0.0)), float(max(allu.max(), 0.0))
        vd = _nice((hi - lo) / 6.0)
        for ch in chs:
            link.set_channel(ch, vd, (hi + lo) / 2.0)
        steps = ([("EO1", solo(wave, "EO1")), ("EO2", solo(wave, "EO2"))] if identify else [])
        steps.append(("both", wave))
        for key, w in steps:
            if cancelled():
                from .hw import Cancelled
                raise Cancelled()
            sess.load(w, keep_on=True)
            if not all(sess.outputs().values()):
                sess.on()
            got = capture(link, chs, shots, points, wait_s, cancelled)
            res = {}
            for name, ch in zip(CHANNELS, chs):
                t, mean, stack = got[ch]
                res[name] = compare(t, mean, w.u[name], w.dt, stack)
                res[name]["trace"] = (t, mean)
            report["steps"][key] = res
            log(f"  dry run {key}: " + "; ".join(
                f"{n} -> scope CH{c}: gain {res[n]['gain']:.3f}, delay "
                f"{res[n]['delay_us']:.1f} us, {res[n]['rms_mV']:.1f} mV rms"
                for n, c in zip(CHANNELS, chs)))
        # which output reached which scope channel
        if identify:
            for name, other in (("EO1", "EO2"), ("EO2", "EO1")):
                st = report["steps"][name]
                ptp_meant = float(np.ptp(wave.u[name]))
                if ptp_meant < 0.05:
                    continue
                seen_here = float(np.ptp(st[name]["trace"][1]))
                seen_there = float(np.ptp(st[other]["trace"][1]))
                swapped = seen_here < 0.5 * ptp_meant and seen_there > 0.5 * ptp_meant
                if seen_here < 0.5 * ptp_meant:
                    if seen_there > 0.5 * ptp_meant:
                        report["problems"].append(
                            f"AWG CH{CHANNELS[name]} ({name}) shows on scope CH{wiring[other]}, "
                            f"not CH{wiring[name]}: the outputs are swapped against the "
                            f"wiring given - the Treks would get each other's drive")
                    else:
                        report["problems"].append(
                            f"AWG CH{CHANNELS[name]} ({name}) not seen on scope "
                            f"CH{wiring[name]} ({seen_here*1e3:.0f} mV p-p of "
                            f"{ptp_meant*1e3:.0f} meant): not connected, or output off?")
                if seen_there > 0.1 * ptp_meant + 0.02 and not swapped:
                    report["problems"].append(
                        f"scope CH{wiring[other]} moved {seen_there*1e3:.0f} mV p-p while only "
                        f"{name} played: cross-talk or the wrong cable")
        for name in CHANNELS:
            for pr in report["steps"]["both"][name]["problems"]:
                report["problems"].append(f"{name}: {pr}")
    finally:
        sess.dry = False
        try:
            for ch, (vd, off) in saved_ch.items():
                link.set_channel(ch, vd, off)
            for k, v in saved_tb.items():
                if v is not None:
                    sc.put(k, v)
        except Exception as exc:
            log(f"  could not restore the scope ({exc})")
    report["ok"] = not report["problems"]
    if report["ok"]:
        both = report["steps"]["both"]
        sess.verified[names(wave)] = {
            "label": wave.label, "gain": [both[n]["gain"] for n in CHANNELS],
            "delay_us": [both[n]["delay_us"] for n in CHANNELS]}
    return report

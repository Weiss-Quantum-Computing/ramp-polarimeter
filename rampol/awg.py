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
  period, FRQ = 1/(N dt). The default record is EOM-ILC's: 11 ms at 2 us,
  5501 points, 90.893 Hz - so switching between an ILC drive and a ramp here
  never needs the channel set up again, and the ILC's own FRQ check passes
  afterwards. N <= 16384 (datasheet; 5501 is the most proven).
* Names. The 4063B cannot delete stored waveforms over SCPI and locks its
  panel past 11-character names. A waveform's name is a hash of its samples
  (RP<ch><8 hex>): the same waveform is selected again, not stored again.
* Outputs. Both channels are checked before anything is written; ON is set
  one channel at a time with the bookkeeping first, and read back; every
  exit switches off whatever was touched. A channel that is ON but was not
  switched on here belongs to someone else (the ILC panel): refused.
* End state. 'off' (the default, fine with the X2 FPGA/buffer stage
  bypassed) or 'park': an idle-level waveform with the outputs left ON, for
  when the drive goes THROUGH that stage, whose output goes high on a
  floating input (3 Sep 2026; pull-down not fitted).
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
AWG_CAP = 9.6              # V: this program's cap per channel (a margin under the rail)
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


def ramp_hold(rotation_deg, p, idle=None, chan=None):
    """Idle -> the commanded rotation -> idle: lead at idle, `rise` edge,
    hold, `fall` edge, idle to the end of a `record_ms` record on a `dt_us`
    grid (N = record / dt + 1, EOM-ILC's convention: 11 ms -> 5501). The
    rotation is split split : 1 - split between EO1 and EO2 (awg_volts) and
    is relative to idle. Both ends are exactly idle."""
    p = dict(DEFAULTS, **(p or {}))
    idle = idle or {"EO1": 0.0, "EO2": 0.0}
    split = float(p["split"])
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"split {split:g} must be between 0 and 1")
    dt = float(p["dt_us"]) * 1e-6
    n = int(round(float(p["record_ms"]) * 1e-3 / dt)) + 1
    seg = [float(p[k]) * 1e-3 for k in ("lead_ms", "rise_ms", "hold_ms", "fall_ms")]
    if min(seg) < 0:
        raise ValueError("lead, rise, hold and fall must be >= 0")
    nl, nr, nh, nf = (int(round(x / dt)) for x in seg)
    if nl + nr + nh + nf + 1 > n:
        raise ValueError(f"lead + rise + hold + fall = {sum(seg)*1e3:.3f} ms does not fit "
                         f"the {p['record_ms']:g} ms record")
    prof = np.zeros(n)
    prof[nl:nl + nr] = edge(nr, p["edge"])
    prof[nl + nr:nl + nr + nh] = 1.0
    if nf:
        # the fall's first sample is the hold level, its last just above 0
        prof[nl + nr + nh:nl + nr + nh + nf] = 1.0 - edge(nf, p["edge"])
    prof[[0, -1]] = 0.0
    amps = biasmod.awg_volts(rotation_deg, split, **({"chan": chan} if chan else {}))
    u = {k: float(idle.get(k, 0.0)) + amps[k] * prof for k in CHANNELS}
    t = np.arange(n) * dt
    hold = ((nl + nr) * dt, (nl + nr + nh) * dt)
    return Wave(t, u, dt, f"ramp to {rotation_deg:g} deg ({p['edge']} edges "
                f"{p['rise_ms']:g}/{p['fall_ms']:g} ms, hold {p['hold_ms']:g} ms)",
                hold=hold, rotation=float(rotation_deg), source="ramp")


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
            out.append(("FAIL", f"{k}: {pk:.3f} V at the AWG, past the {AWG_CAP:g} V cap"))
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


def timebase_for(wave):
    """(scale s/div, position s) for REFerence LEFT: the record from 0.2 ms
    before the trigger, as the bias run sets it."""
    div = biasmod._nice_up(wave.period * 1.05 / 10)
    return div, -0.2e-3 + div


# ------------------------------------------------------------------- session
class Session:
    """The AWG as this window holds it. One lock serialises every AWG call
    (Outputs OFF can be pressed while a measurement runs on the worker).
    `owned` is the set of channels this session switched ON."""

    def __init__(self, awg, ib=None, log=print):
        self.awg, self.ib, self.log = awg, ib, log
        self.lock = threading.RLock()
        self.owned = set()
        self.wave = None          # what is loaded (None: unknown / replaced)
        self.parked = False
        self._period = {}         # without ilc_bench (simulator): the FRQ set here

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
            if live and not keep_on:
                self.off()
                live = []
            for name, ch in CHANNELS.items():
                if self.setup_ok(ch, wave.period):
                    if ch in live:
                        raise RuntimeError(
                            f"CH{ch} needs setting up for a {wave.period*1e3:.3f} ms record "
                            f"(FRQ), which needs its output OFF - use the same record "
                            f"length, or the 'off' end policy for this change")
                    self._setup(ch, wave.period)
            names = {}
            for name, ch in CHANNELS.items():
                if ch in live:
                    self.log(f"  CH{ch} is live: the change can land mid-burst")
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
            try:
                for ch in CHANNELS.values():
                    self.owned.add(ch)
                    self.awg.set_output(ch, True)
                state = self.outputs()
            except Exception:
                # CH1 may be live with CH2 refused: never leave half of it on
                self.off()
                raise
            if not all(state.values()):
                bad = [ch for ch, on in state.items() if not on]
                self.off()
                raise RuntimeError(f"CH{bad} did not switch on - both switched OFF")
            self.log("AWG outputs ON (CH1 -> X1, CH2 -> X2)")

    def off(self):
        """Both outputs OFF, each tried whatever the other does."""
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

    def park(self, like=None):
        """An idle-level waveform with the outputs ON (the X2 FPGA stage must
        not see a floating input)."""
        with self.lock:
            base = like or self.wave
            if base is None:
                raise RuntimeError("no waveform to take the grid and idle from")
            self.load(idle_flat(base.idle(), base), keep_on=True)
            if not all(self.outputs().values()):
                self.on()
            self.log("AWG parked: idle level, outputs ON")

    def end(self, policy="off"):
        if policy == "park" and self.wave is not None:
            try:
                self.park()
                return
            except Exception as exc:
                self.log(f"  park failed ({exc}) - switching OFF instead")
        self.off()

    def forget(self):
        """Something else (a bias run) replaced the waveform."""
        self.wave = None

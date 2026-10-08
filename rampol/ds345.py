"""The SRS DS345 as a light gate: a waveform, played once per bench trigger,
into the light modulator, so the light is on only where a measurement wants
it - e.g. only during the hold when the analyzer sits at the hold's null, so
the bright rest no longer overdrives the scope (8 Oct 2026: at 20 mV/div the
trace stayed pinned at the screen top for ~5.5 ms after the light went dark).

The instrument (SRS DS345 manual, read 8 Oct 2026):

* Arbitrary waveforms: 8 to 16,300 points, 12-bit (-2047..+2047), one point
  per 1/FSMP with FSMP = 40 MHz / N, N = 1 .. 2^34 - 1. The limit is real
  (unlike the 4063B's old 16384, which was the wrong model's): a gate fits by
  choosing N (fit_clock), so a 300 ms gate plays at ~18 us per point.
* Burst modulation (MTYP 5) bursts an ARB too: BCNT 1 plays the record once
  per trigger, always from its first point; TSRC 2 = rising edge at the
  rear-panel TRIGGER input (TTL, 10 kOhm). A trigger during a burst is
  ignored. What the output holds between bursts is not stated: every gate
  starts and ends at its idle level, and the dry run reads it.
* The output has a 50 Ohm source impedance and its levels are specified into
  50 Ohm: into a high-impedance load (this bench: the modulator input, the
  scope) the voltage is TWICE what is programmed. Everything here is in
  volts AT THE LOAD; program() halves it. Programmed |AMPL/2| + |OFFS| <=
  5 V, so +-10 V at a Hi-Z load. AMPL 0VP gives a DC-only output (OFFS).
* There is no output switch: the BNC always drives. 'Park' is a DC level
  (the light's normal state), never off - nothing floats.

The instrument code (GPIB find/connect, *ESR? after each write, the LDWF?
handshake and checksum) is the DS345-AWG-GUI's Ds345 class, loaded by path
as the 4063B driver is; this module only adds the gate and its session.
"""
import math
import threading

import numpy as np

CLOCK_HZ = 40e6
MAX_PTS = 16300
MIN_PTS = 8
N_MAX = 2 ** 34 - 1
DAC = 2047
HIZ = 2.0                   # load volts per programmed volt (50 Ohm source, Hi-Z load)
PROG_LIMIT = 5.0            # programmed |AMPL/2| + |OFFS|
FUNC_ARB, FUNC_SINE = 5, 0
MTYP_BURST, TSRC_EXT_POS = 5, 2

TOL = {"gain": 0.03, "idle_V": 0.05, "delay_us": 50.0, "shape": 0.03}


def fit_clock(record_s, dt_s=None):
    """(N, dt s): the sample-clock divider that puts `record_s` in at most
    MAX_PTS points (or the given dt rounded up to a divider)."""
    want = max(float(dt_s or 0.0), float(record_s) / (MAX_PTS - 1))
    n = max(1, int(math.ceil(want * CLOCK_HZ - 1e-6)))
    if n > N_MAX:
        raise ValueError(f"{record_s:g} s does not fit the DS345 at any sample clock")
    return n, n / CLOCK_HZ


def parse_windows(text):
    """'2.2-11.8, 20-30' (ms) -> [(t0, t1)] in s, sorted; '' -> []."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("-"):
            a, b = part[1:].split("-", 1)
            a = "-" + a
        else:
            a, b = part.split("-", 1)
        t0, t1 = float(a) * 1e-3, float(b) * 1e-3
        if t1 <= t0:
            raise ValueError(f"window {part}: the end must come after the start")
        out.append((t0, t1))
    return sorted(out)


class Gate:
    """The gate as the load sees it: v (load volts) on a dt grid from the
    trigger. idle: 'on' or 'off' (the level outside the windows, and between
    bursts); windows [(t0, t1)] s at the other level."""

    def __init__(self, t, v, dt, n_div, label, idle, windows, on_v, off_v, edge_s):
        self.t, self.v, self.dt, self.n_div = np.asarray(t, float), np.asarray(v, float), dt, n_div
        self.label, self.idle, self.windows = label, idle, list(windows)
        self.on_v, self.off_v, self.edge_s = float(on_v), float(off_v), float(edge_s)

    @property
    def n(self):
        return len(self.v)

    @property
    def period(self):
        return self.n * self.dt

    @property
    def idle_v(self):
        return self.on_v if self.idle == "on" else self.off_v

    def light(self, t):
        """0..1: how much of the light the gate lets through at t (s from
        the trigger), linear between off_v and on_v; idle outside the record."""
        v = np.interp(np.asarray(t, float), self.t, self.v, left=self.idle_v,
                      right=self.idle_v)
        span = self.on_v - self.off_v
        return np.clip((v - self.off_v) / span, 0.0, 1.0) if span else np.ones_like(v)

    def describe(self):
        ws = ", ".join(f"{a*1e3:g}-{b*1e3:g}" for a, b in self.windows) or "none"
        other = "off" if self.idle == "on" else "on"
        return (f"light {self.idle}, {other} in {ws} ms ({self.edge_s*1e6:g} us edges); "
                f"{self.n} points at {self.dt*1e6:g} us = {self.period*1e3:.3f} ms; "
                f"on {self.on_v:+.3f} V / off {self.off_v:+.3f} V at the modulator")


def build(windows, idle="on", on_v=1.0, off_v=0.0, edge_us=20.0, record_ms=None,
          tail_ms=0.5, label=None):
    """A gate: `idle` level outside `windows` (s), the other level inside,
    cosine edges of `edge_us`, starting and ending at idle. The record runs
    to the last window's end + tail (or `record_ms`); the clock divider is
    chosen so it fits the DS345's memory."""
    if idle not in ("on", "off"):
        raise ValueError("idle is 'on' or 'off'")
    windows = sorted(windows)
    if any(a < 0 for a, _b in windows):
        raise ValueError("a gate window cannot start before the trigger: the DS345 starts "
                         "its burst on it (before that the output is at idle)")
    for (a0, a1), (b0, b1) in zip(windows, windows[1:]):
        if b0 < a1:
            raise ValueError("gate windows overlap")
    edge = max(float(edge_us), 0.0) * 1e-6
    end = (windows[-1][1] if windows else 0.0) + edge + float(tail_ms) * 1e-3
    rec = max(end, float(record_ms or 0.0) * 1e-3, 1e-3)
    n_div, dt = fit_clock(rec)
    n = max(MIN_PTS, int(round(rec / dt)) + 1)
    t = np.arange(n) * dt
    lvl = {"on": float(on_v), "off": float(off_v)}
    a, b = lvl[idle], lvl["off" if idle == "on" else "on"]
    frac = np.zeros(n)                       # 0 = idle level, 1 = the other
    for w0, w1 in windows:
        if edge > 0:
            up = np.clip((t - w0) / edge, 0, 1)
            dn = np.clip((w1 - t) / edge, 0, 1)
            f = np.minimum(0.5 - 0.5 * np.cos(np.pi * up), 0.5 - 0.5 * np.cos(np.pi * dn))
        else:
            f = ((t >= w0) & (t < w1)).astype(float)
        frac = np.maximum(frac, f)
    frac[[0, -1]] = 0.0
    v = a + (b - a) * frac
    g = Gate(t, v, dt, n_div, label or "", idle, windows, on_v, off_v, edge)
    g.label = label or g.describe()
    return g


def program(gate):
    """(OFFS, AMPL Vpp, codes) to program for the gate's load volts: half of
    each (Hi-Z), the codes spanning the gate's range. Raises past the
    output's limit."""
    v = gate.v / HIZ
    lo, hi = float(v.min()), float(v.max())
    offs = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if abs(offs) + half > PROG_LIMIT + 1e-9:
        raise ValueError(f"{gate.v.min():+.2f}..{gate.v.max():+.2f} V at the load needs "
                         f"{lo:+.2f}..{hi:+.2f} V programmed: past the DS345's +-5 V "
                         f"(+-10 V into Hi-Z)")
    if half <= 0:
        codes = np.zeros(gate.n)
    else:
        codes = (v - offs) / half
    return offs, 2 * half, codes


def check(gate, load_min_v, load_max_v, trig_hz=None):
    """[(level, text)] like awg.check. A FAIL stops a load."""
    out = []
    lo, hi = float(gate.v.min()), float(gate.v.max())
    if lo < load_min_v - 1e-9 or hi > load_max_v + 1e-9:
        out.append(("FAIL", f"{lo:+.3f}..{hi:+.3f} V at the modulator is outside the "
                            f"{load_min_v:+.3f}..{load_max_v:+.3f} V it is set to take "
                            f"(DS345 Settings)"))
    try:
        offs, ampl, _c = program(gate)
        out.append(("INFO", f"programmed OFFS {offs:+.4f} V, AMPL {ampl:.4f} Vpp (into 50 "
                            f"Ohm terms; the Hi-Z load sees twice that)"))
        if 0 < ampl < 0.01:
            out.append(("FAIL", f"a {ampl*1e3:.1f} mVpp swing is under the DS345's 10 mVpp"))
    except ValueError as exc:
        out.append(("FAIL", str(exc)))
    if gate.n > MAX_PTS:
        out.append(("FAIL", f"{gate.n} points: past the DS345's {MAX_PTS}"))
    out.append(("INFO", f"{gate.n} points at {gate.dt*1e6:g} us (40 MHz / {gate.n_div}) = "
                        f"{gate.period*1e3:.3f} ms; 12-bit: steps of "
                        f"{(hi - lo) / (2 * DAC) * 1e3:.2f} mV at the load"))
    if trig_hz:
        gap = 1.0 / float(trig_hz)
        if gate.period > 0.95 * gap:
            out.append(("FAIL", f"the gate ({gate.period*1e3:.1f} ms) is not over before the "
                                f"next trigger ({gap*1e3:.0f} ms): the DS345 ignores triggers "
                                f"during a burst, so it would skip every other one"))
    return out


def worst(found):
    levels = [lv for lv, _ in found]
    return "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "OK")


def windows_for(pol, kind, margin_s=0.3e-3, recover_s=0.5e-3):
    """Gate windows from a scan's segments (analysis.segments of its fit):
    kind 'hold' - light only inside each hold, `margin_s` in from its ends
    (idle off); kind 'rest' - light off from the first motion to the last
    one's end + `recover_s` (idle on). Returns (idle, [(t0, t1)])."""
    from . import analysis as an
    segs = an.segments(pol["t"], pol["rotation"])
    if kind == "hold":
        ws = [(s["t0"] + margin_s, s["t1"] - margin_s) for s in segs
              if s["base"] == "hold" and s["t1"] - s["t0"] > 2 * margin_s]
        return "off", [(max(a, 0.0), b) for a, b in ws if b > max(a, 0.0)]
    moving = [s for s in segs if s["base"] not in ("rest", "after")]
    if not moving:
        return "on", []
    t0 = max(min(s["t0"] for s in moving) - margin_s, 0.0)
    t1 = max(s["t1"] for s in moving) + recover_s
    return "on", [(t0, t1)]


# ------------------------------------------------------------------- session
class Session:
    """The DS345 as this window holds it: `dev` is the GUI module's Ds345
    (or the simulator's FakeDS345), already connected. One lock serialises
    every call. `gate` is what plays (None: parked or unknown); `verified`
    holds the labels of gates that passed a dry run, keyed by their samples."""

    def __init__(self, dev, log=print):
        self.dev, self.log = dev, log
        self.lock = threading.RLock()
        self.gate = None
        self.parked_v = None
        self.verified = {}

    def _w(self, cmd):
        missed = self.dev.checked_write(cmd)
        if missed:
            raise RuntimeError("DS345 refused: " + "; ".join(missed))

    def park(self, load_v):
        """A DC level at the load (the light's normal state), burst off. The
        output keeps driving - there is no off."""
        with self.lock:
            prog = float(load_v) / HIZ
            if abs(prog) > PROG_LIMIT:
                raise ValueError(f"{load_v:+.2f} V at the load is past the DS345's +-10 V")
            self._w("MENA 0")
            self._w(f"FUNC {FUNC_SINE}")
            self._w("AMPL 0VP")
            self._w(f"OFFS {prog:.4f}")
            self.gate, self.parked_v = None, float(load_v)
            self.log(f"DS345 parked: DC {load_v:+.3f} V at the modulator (programmed "
                     f"{prog:+.4f} V)")

    def load(self, gate):
        """Play `gate` once per rising edge at the TRIGGER input. Through a
        DC level at the gate's idle first, so nothing in between plays
        free-running: burst armed on DC, the record uploaded, ARB selected
        (amplitude still 0: DC), then the amplitude."""
        offs, ampl, codes = program(gate)
        with self.lock:
            self.park(gate.idle_v)
            self._w(f"MTYP {MTYP_BURST}")
            self._w("BCNT 1")
            self._w(f"TSRC {TSRC_EXT_POS}")
            self._w("MENA 1")
            self._w(f"FSMP {CLOCK_HZ / gate.n_div:.6f}")
            self.dev.upload_arb(codes, normalize=False)
            self._w(f"FUNC {FUNC_ARB}")
            self._w(f"OFFS {offs:.4f}")
            self._w(f"AMPL {ampl:.4f}VP")
            self.gate, self.parked_v = gate, None
            self.log(f"DS345: {gate.label} - armed on the rising edge at TRIGGER IN")

    def key(self, gate):
        return (gate.n, gate.n_div, round(float(gate.v.sum()), 6), round(gate.on_v, 6),
                round(gate.off_v, 6))

    def is_verified(self, gate):
        return self.key(gate) in self.verified


def dry_run(sess, link, gate, scope_ch, shots=4, points=20000, wait_s=10.0,
            cancelled=None, log=print):
    """Play `gate` into scope channel `scope_ch` (the DS345 output teed
    there; the scope is Hi-Z, as the modulator is) and compare it with the
    load volts meant: gain (a 50 Ohm / Hi-Z mistake reads 0.5 or 2), delay
    after the trigger, shape, and the idle level between bursts. The scope's
    channel and timebase are put back. Returns the report."""
    from . import awg as awgmod
    sc = link.scope
    saved_ch = link.channel_state([scope_ch])
    tb_keys = (":TIMebase:SCALe", ":TIMebase:POSition", ":TIMebase:REFerence")
    saved_tb = {k: sc.get(k) for k in tb_keys}
    rep = {"label": gate.label, "scope_ch": scope_ch, "shots": shots, "problems": []}
    try:
        div, pos = awgmod.timebase_for(gate, before_ms=0.2, after_ms=0.2)
        sc.put(":TIMebase:REFerence", "LEFT")
        sc.put(":TIMebase:SCALe", f"{div:.6g}")
        sc.put(":TIMebase:POSition", f"{pos:.6g}")
        lo, hi = float(min(gate.v.min(), 0.0)), float(max(gate.v.max(), 0.0))
        vd = awgmod._nice(max(hi - lo, 0.02) / 6.0)
        link.set_channel(scope_ch, vd, (hi + lo) / 2.0)
        sess.load(gate)
        got = awgmod.capture(link, [scope_ch], shots, points, wait_s, cancelled)
        t, mean, stack = got[scope_ch]
        res = awgmod.compare(t, mean, gate.v, gate.dt, stack)
        res["trace"] = (t, mean)
        rep["result"] = res
        if abs(res["gain"] - 1) > TOL["gain"] and np.ptp(gate.v) > 0.05:
            what = (" - 0.5: a 50 Ohm load (or terminator) where Hi-Z was assumed"
                    if abs(res["gain"] - 0.5) < 0.1 else
                    " - 2: the levels doubled (programmed as if into 50 Ohm twice?)"
                    if abs(res["gain"] - 2) < 0.2 else "")
            rep["problems"].append(f"gain {res['gain']:.3f}{what}")
        if abs(res["delay_us"]) > TOL["delay_us"]:
            rep["problems"].append(f"starts {res['delay_us']:.1f} us after the trigger")
        if res.get("idle_meas_V") is not None and abs(res["idle_meas_V"] - gate.idle_v) \
                > TOL["idle_V"]:
            rep["problems"].append(f"between bursts the output sits at "
                                   f"{res['idle_meas_V']:+.3f} V, not the idle "
                                   f"{gate.idle_v:+.3f} V")
        ptp = float(np.ptp(gate.v))
        if ptp > 0.05 and res["rms_mV"] * 1e-3 > TOL["shape"] * ptp + 5e-3:
            rep["problems"].append(f"shape off by {res['rms_mV']:.1f} mV rms")
        if res.get("jitter_us") is not None and res["jitter_us"] > 20:
            rep["problems"].append(f"shots start {res['jitter_us']:.0f} us apart: not "
                                   f"triggered by the bench trigger?")
        log(f"  DS345 dry run: gain {res['gain']:.3f}, delay {res['delay_us']:.1f} us, "
            f"{res['rms_mV']:.1f} mV rms, idle "
            + ("?" if res.get("idle_meas_V") is None else f"{res['idle_meas_V']:+.3f} V"))
    finally:
        try:
            for ch, (v_, off) in saved_ch.items():
                link.set_channel(ch, v_, off)
            for k, v in saved_tb.items():
                if v is not None:
                    sc.put(k, v)
        except Exception as exc:
            log(f"  could not restore the scope ({exc})")
    rep["ok"] = not rep["problems"]
    if rep["ok"]:
        sess.verified[sess.key(gate)] = {"label": gate.label,
                                        "delay_us": rep["result"]["delay_us"]}
    return rep

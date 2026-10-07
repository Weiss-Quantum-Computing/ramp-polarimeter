"""A bench in software: an EOM ramp, an analyzer on a simulated ELL14, and a
scope that digitizes it the way the MSO-X does - so the scan, the files, the
analysis and the window can all be run and tested with no instrument.

The scope is a subclass of scope_grab.Scope with a fake VISA session under it,
so settings reads, the offset dither, metadata and capture files go through
Scope Grab's real code. Only the acquisition itself is replaced.

What the model includes, because each one is a trap on the real bench:
  * the analyzer's mechanical zero is NOT the optical zero (23.7 deg here)
  * extinction ratio worst mid-ramp, best at 0/180 deg rotation, capped by the
    analyzer's own (1e4)
  * slow laser-intensity drift on the scale of a scan (0.3 % over minutes)
  * an 8-bit converter with a per-code error pattern (the offset dither's
    reason to exist) and clipping at the edge of the screen
  * a crystal "memory": a small rotation that decays for tens of ms after a ramp
  * dark offset on the PD
Time is virtual: an acquisition of N triggers advances the clock by N / 3.7 s
without sleeping, so a full scan runs in seconds under test.
"""
import math

import numpy as np

TRIG_HZ = 3.6997


class Bench:
    """The optics and the clock, shared by the fake scope and fake mount."""

    def __init__(self, seed=1, mount_of_rest_pol=23.7, imax=5.0, dark=-0.012,
                 er_rest=5000.0, er_mid=300.0, er_polarizer=1.0e4,
                 pd_noise=0.012, drift=3e-3, drift_period_s=600.0,
                 ramp_up_ms=4.61, hold_ms=1.24, swing_deg=180.0,
                 memory_deg=0.15, memory_tau_ms=35.0, deg_per_mon_v=(17.550, 17.519),
                 legs_ms=(0.0,), lock_miss=0.0, rotator_err_deg=0.0,
                 rotator_err_period_deg=90.0, ambient=0.0):
        self.rng = np.random.default_rng(seed)
        self.mount_of_rest_pol = mount_of_rest_pol
        self.imax, self.dark = imax, dark
        self.er_rest, self.er_mid, self.er_pol = er_rest, er_mid, er_polarizer
        self.pd_noise, self.drift, self.drift_period = pd_noise, drift, drift_period_s
        self.up, self.hold, self.swing = ramp_up_ms * 1e-3, hold_ms * 1e-3, swing_deg
        self.mem, self.mem_tau = memory_deg, memory_tau_ms * 1e-3
        self.k1, self.k2 = deg_per_mon_v
        self.legs = [x * 1e-3 for x in legs_ms]   # leg start times (s)
        self.lock_miss = lock_miss                  # fraction of shots 2.5 % dim
        self.clock = 0.0               # virtual seconds since the bench started
        self.mount = 300.0             # mechanical angle of the analyzer (deg)
        # a static error of the rotator: the light turns by rot + this, the
        # monitors report rot (the 5 Oct ramps' +-2-3 deg, ~90 deg period)
        self.rot_err, self.rot_err_period = rotator_err_deg, rotator_err_period_deg
        # the AWG, when a FakeAWG plays into the bench: {ch: (period s, u V)}
        # and which outputs are on. While any is on it replaces the legs.
        self.awg_drive = {}
        self.awg_on = {1: False, 2: False}
        self.awg_gain = {1: 0.5594, 2: 0.5924}     # AWG V -> monitor V
        # where the AWG's BNCs go: 'treks' (the bench) or 'scope' (a dry run:
        # awg_scope {AWG ch: scope ch}); the generator's zero-code error at
        # 20 Vpp (measured 24 Aug 2026) and its burst delay after the trigger
        self.wiring = "treks"
        self.awg_scope = {}
        self.awg_zero = {1: -0.012, 2: -0.040}
        self.awg_delay = 1.0e-6
        # stray light on the PD that does not come through the analyzer (V),
        # there with the beam blocked; only covering the PD removes it
        self.ambient = ambient
        self.covered = False

    # -- the ramp ---------------------------------------------------------
    def monitors(self, t):
        """(X1, X2) monitor volts at t: the AWG's plateaus when it is playing,
        otherwise the ramp legs shared equally."""
        t = np.asarray(t, float)
        if any(self.awg_on.values()) and self.awg_drive:
            if self.wiring != "treks":          # the AWG is on the scope, not the Treks
                return np.zeros_like(t), np.zeros_like(t)
            out = []
            for ch in (1, 2):
                if not self.awg_on.get(ch) or ch not in self.awg_drive:
                    out.append(np.zeros_like(t))
                    continue
                period, u = self.awg_drive[ch]
                tu = np.arange(len(u)) * period / len(u)
                v = np.interp(t, tu, u, left=u[0], right=u[-1])
                out.append(v * self.awg_gain[ch])
            return out[0], out[1]
        rot = sum(self._leg(t - t0) for t0 in self.legs)
        m = rot / (self.k1 + self.k2)
        return m, m

    def awg_out(self, ch, t):
        """The AWG output's voltage at t (what a scope channel cabled to it
        reads): the burst from the trigger + awg_delay, the first sample
        held around it, the zero-code error added; 0 V with the output off."""
        t = np.asarray(t, float)
        if not self.awg_on.get(ch) or ch not in self.awg_drive:
            return np.zeros_like(t)
        period, u = self.awg_drive[ch]
        tu = np.arange(len(u)) * period / len(u)
        return np.interp(t - self.awg_delay, tu, u, left=u[0], right=u[0]) + self.awg_zero[ch]

    def rotation(self, t):
        """Rotation from the rest polarization (deg) at time t (s) after the
        trigger, as the monitors report it: one transport per leg
        (raised-cosine up, hold, raised-cosine down), each followed by a
        decaying memory term - or the AWG's plateau."""
        m1, m2 = self.monitors(t)
        return self.k1 * m1 + self.k2 * m2

    def _leg(self, t):
        up, hold, sw = self.up, self.hold, self.swing
        r = np.zeros_like(t)
        a = (t >= 0) & (t < up)
        r[a] = sw * 0.5 * (1 - np.cos(np.pi * t[a] / up))
        b = (t >= up) & (t < up + hold)
        r[b] = sw
        c = (t >= up + hold) & (t < 2 * up + hold)
        r[c] = sw * 0.5 * (1 + np.cos(np.pi * (t[c] - up - hold) / up))
        end = 2 * up + hold
        d = t >= end
        r[d] = self.mem * np.exp(-(t[d] - end) / self.mem_tau)
        return r

    def er(self, rot):
        """Extinction ratio of the light at rotation `rot`, before the
        analyzer: best at 0 and 180, worst at 90."""
        inv = 1 / self.er_rest + (1 / self.er_mid - 1 / self.er_rest) * np.sin(
            np.deg2rad(rot)) ** 2
        return 1 / inv

    def intensity_gain(self):
        return 1 + self.drift * math.sin(2 * math.pi * self.clock / self.drift_period)

    def signals(self, t, shots, static=False):
        """Noise-free-ish channel voltages for one acquisition of `shots`
        triggers at the current mount angle: {role: volts}. static: a line-
        triggered record of the light as it is - no ramp in it."""
        m1, m2 = self.monitors(t)
        if static:
            m1, m2 = np.zeros_like(t), np.zeros_like(t)
        rot = self.k1 * m1 + self.k2 * m2
        light = rot + self.rot_err * np.sin(2 * np.pi * rot / self.rot_err_period)
        # analyzer angle relative to the polarization
        d = np.deg2rad(self.mount - self.mount_of_rest_pol - light)
        inv_er = 1 / self.er(rot) + 1 / self.er_pol
        imax = self.imax * self.intensity_gain()
        if shots == 1 and self.lock_miss and self.rng.random() < self.lock_miss:
            imax *= 0.975            # the lock did not catch on this shot
        pd = self.dark + imax * (np.cos(d) ** 2 + inv_er * np.sin(d) ** 2)
        if self.covered:
            pd = np.full_like(t, self.dark)
        elif self.ambient:
            pd = pd + self.ambient
        n = np.sqrt(max(shots, 1))
        pd = pd + self.rng.normal(0, self.pd_noise / n, t.size)
        x1 = m1 + self.rng.normal(0, 1e-3 / n, t.size)
        x2 = m2 + self.rng.normal(0, 1e-3 / n, t.size)
        marker = np.where((t >= 0) & (t < 20e-6), 5.0, 0.0)
        ref = 2.0 * self.intensity_gain() + self.rng.normal(0, 2e-3 / n, t.size)
        # the command into each Trek: ~8.5 V for 5.15 kV on the bench (2 Oct
        # 2026), i.e. ~1.65 x the monitor
        c1 = m1 / self.awg_gain[1] * 0.92 + self.rng.normal(0, 1e-3 / n, t.size)
        c2 = m2 / self.awg_gain[2] * 0.96 + self.rng.normal(0, 1e-3 / n, t.size)
        return {"PD": pd, "MonX1": x1, "MonX2": x2, "CmdX1": c1, "CmdX2": c2,
                "Marker": marker, "Ref": ref, "Other": np.zeros_like(t)}


class FakeELL14:
    """The ELL14 driver's interface over Bench.mount. Lands within a few mdeg,
    with 0.04 deg of backlash that depends on the direction of the last move -
    which is why the scan approaches every angle from below."""

    def __init__(self, bench, zero_offset_deg=0.0, port="SIM", backlash=0.04):
        self.bench = bench
        self.zero_offset = zero_offset_deg
        self.port = port
        self.serial_no = "SIM00001"
        self.pulses_per_rev = 143360
        self.backlash = backlash
        self._cmd = bench.mount        # commanded mechanical angle
        self._last_dir = 1

    def info(self):
        return {"type": 14, "serial": self.serial_no, "year": "2026",
                "firmware": "00", "thread": "metric", "hardware": 0,
                "travel": 360, "pulses_per_unit": self.pulses_per_rev}

    def status(self):
        return 0

    def _land(self, mech):
        step = (mech - self._cmd + 180) % 360 - 180
        if step:
            self._last_dir = 1 if step > 0 else -1
        self._cmd = mech % 360
        lag = -self.backlash / 2 * self._last_dir
        self.bench.mount = (self._cmd + lag
                            + self.bench.rng.normal(0, 0.003)) % 360
        self.bench.clock += 0.3 + abs(step) / 360 * 1.5
        return self.position()

    def position(self):
        return (self.bench.mount - self.zero_offset) % 360.0

    def home(self, ccw=False):
        return self._land(0.0)

    def move_to(self, angle_deg):
        return self._land(angle_deg + self.zero_offset)

    def move_by(self, delta_deg):
        return self._land(self._cmd + delta_deg)

    def goto(self, angle_deg, tol=0.05, retries=2, settle=0.05):
        return self.move_to(angle_deg)

    def close(self):
        pass


class FakeAWG:
    """The BK4063B calls a bias run makes, playing into a Bench: uploads
    become the bench's drive, outputs switch it, the burst plays on every
    trigger from the record's start."""

    def __init__(self, bench):
        self.bench = bench
        self.idn = "SIMULATED,4063B,SIM,0"
        self.frq = {1: 1000.0, 2: 1000.0}
        self.amp = {1: 20.0, 2: 20.0}
        self.stored = {}
        self.selected = {}

    def connect(self, resource=None):
        return self.idn

    def close(self):
        pass

    def is_on(self, ch):
        return bool(self.bench.awg_on.get(ch))

    def set_output(self, ch, on):
        self.bench.awg_on[ch] = bool(on)

    def apply_channel(self, ch, blocks, log=lambda s: None):
        b = blocks.get("BSWV", {})
        if "FRQ" in b:
            self.frq[ch] = float(b["FRQ"])
        if "AMP" in b:
            self.amp[ch] = float(b["AMP"])
        return []

    def upload_arb(self, ch, name, samples, freq=None, amp=None, offset=None,
                   phase=None, normalize=True):
        x = np.clip(np.asarray(samples, float), -1, 1)
        self.stored[name] = x
        self.selected[ch] = name
        self.bench.awg_drive[ch] = (1.0 / self.frq[ch], x * self.amp[ch] / 2)
        self.bench.clock += 0.8
        return len(x)

    def list_waveforms(self, user_only=False):
        return list(self.stored)

    def get_arb(self, ch):
        """As the 4063B: a user waveform reads back as NAME,<name>.bin."""
        name = self.selected.get(ch, "")
        return {"NAME": name + ".bin"} if name else {}

    def write(self, command):
        """Only what the session sends raw: C<ch>:ARWV NAME,<name>."""
        head, _, arg = command.partition(" ")
        if head.endswith(":ARWV") and arg.startswith("NAME,"):
            ch, name = int(head[1]), arg[5:]
            self.selected[ch] = name
            self.bench.awg_drive[ch] = (1.0 / self.frq[ch], self.stored[name] * self.amp[ch] / 2)


class FakeInst:
    """The VISA session under the fake scope: settings as a dict."""

    def __init__(self, state):
        self.state = state
        self.writes = []

    def query(self, text):
        text = text.strip()
        if text in ("*OPC?", ":TER?"):
            return "1\n"
        if text == ":SYSTem:ERRor?":
            return '+0,"No error"\n'
        root = text[:-1] if text.endswith("?") else text
        return str(self.state.get(root, "0")) + "\n"

    def write(self, text):
        self.writes.append(text)
        parts = text.strip().split(" ", 1)
        if len(parts) == 2:
            self.state[parts[0]] = parts[1]

    def close(self):
        pass


def make_scope_class(sg):
    """A scope_grab.Scope whose acquisitions come from a Bench."""

    class SimScope(sg.Scope):
        def __init__(self, prof, bench, roles):
            super().__init__(prof)
            self.bench = bench
            self.roles = roles          # {ch: role}, so CHn carries the right signal
            self._acq = None
            self._pending = None
            self.realtime = 0.0         # seconds actually slept per acquisition (demo)
            # the scope front end's noise on the PD channel, divisions rms per
            # shot. 0 here: Bench.pd_noise (12 mV) stands for all of it, which
            # is right at 1 V/div and what the analysis tests are tuned on. A
            # bench read at sensitive V/div wants the split - pd_noise 0.2 mV
            # (the PD) and 0.012 div here (12 uV at 1 mV/div) - which the
            # window's simulator uses (gui.ensure_sim) and test_bias sets.
            self.noise_per_div = 0.0
            # the scope's offset error as a fraction of the offset setting
            # (the bench's: about -1.3 %, -35 mV at a 2.7 V offset; 0 here
            # unless a test asks for it)
            self.offset_err = 0.0

        def connect(self, addr=None):
            p = self.prof
            state = {}
            for scpi in sg.setting_roots(p):
                state[scpi] = "0"
            state.update({
                ":TIMebase:SCALe": "1.5E-03", ":TIMebase:POSition": "-2.0E-03",
                ":TIMebase:REFerence": "LEFT", ":TIMebase:MODE": "MAIN",
                p.acq_type: "HRES", p.acq_count: "64",
                ":ACQuire:SRATe": "2.5E+07", ":ACQuire:POINts": "62500",
                ":TRIGger:MODE": "EDGE", ":TRIGger:SWEep": "NORM",
                ":TRIGger:EDGE:SOURce": "EXT", ":TRIGger:EDGE:LEVel": "1.0",
            })
            for ch in p.channels:
                role = self.roles.get(ch, "Other")
                # commands reach ~8.5 V: at 1 V/div they would clip at the
                # converter's edge (~offset + 5 div), so 2 V/div like the bench
                scale, off = {"PD": (1.0, 2.5), "Marker": (2.0, 2.0),
                              "CmdX1": (2.0, 4.0), "CmdX2": (2.0, 4.0),
                              "Ref": (0.5, 2.0)}.get(role, (1.0, 2.5))
                state[p.ch_scale.format(ch=ch)] = f"{scale:.6E}"
                state[p.ch_offset.format(ch=ch)] = f"{off:.6E}"
                state[p.ch_display.format(ch=ch)] = "1"
                state[f":CHANnel{ch}:COUPling"] = "DC"
                state[f":CHANnel{ch}:PROBe"] = "1.0E+00"
            self.inst = FakeInst(state)
            self.idn = "SIMULATED,MSO-X 2014A,SIM0000,0.00"
            self.addr = "SIM"
            return self.idn

        def close(self):
            self.inst = None

        def run(self):
            pass

        def _grid(self, n):
            g = self.inst.state
            from .config import record_span
            t0, t1 = record_span(g[":TIMebase:SCALe"], g[":TIMebase:POSition"],
                                 g.get(":TIMebase:REFerence", "LEFT"))
            return t0, (t1 - t0) / n

        def _arm(self, shots):
            """An acquisition of `shots` triggers. The record itself is made
            on the first readout, at the size it is read out at, and every
            channel comes from that one record."""
            import time
            if self.realtime:
                time.sleep(self.realtime)
            self.bench.clock += shots / TRIG_HZ
            self._pending, self._acq = shots, None

        def _make(self, n):
            t0, dt = self._grid(n)
            t = t0 + dt * np.arange(n)
            src = str(self.inst.state.get(":TRIGger:EDGE:SOURce", "EXT")).upper()
            sig = self.bench.signals(t, self._pending, static=src.startswith("LINE"))
            b = self.bench
            to_scope = {sch: ach for ach, sch in b.awg_scope.items()} if b.wiring == "scope" else {}
            chans = {}
            for ch in self.prof.channels:
                if ch in to_scope:
                    n_ = math.sqrt(max(self._pending, 1))
                    chans[ch] = b.awg_out(to_scope[ch], t) + b.rng.normal(0, 2e-3 / n_, t.size)
                else:
                    chans[ch] = sig[self.roles.get(ch, "Other")]
            self._acq = (t0, dt, chans, self._pending)

        def single(self, wait_s=10.0, cancelled=None):
            if cancelled is not None and cancelled():
                return None
            self._arm(1)
            return True

        def accumulate(self, count, wait_s=10.0, cancelled=None, progress=None,
                       channels=(1,)):
            if cancelled is not None and cancelled():
                return None
            self._arm(count)
            return count

        def transfer_plan(self, averaged, points):
            return ("MAXimum", None) if averaged else ("RAW", points or 20000)

        def hit_count(self):
            return str(self._acq[3]) if self._acq else None

        def record(self, channel, points_mode="RAW", points=None):
            if self._acq is None:
                if self._pending is None:
                    raise RuntimeError("no acquisition")
                self._make(7680 if points_mode == "MAXimum" else int(points or 20000))
            t0, dt, sig, shots = self._acq
            v = sig[channel]
            p = self.prof
            scale = float(self.inst.state[p.ch_scale.format(ch=channel)])
            off = float(self.inst.state[p.ch_offset.format(ch=channel)])
            code = p.adc_code_per_vdiv * scale or 0.04 * scale
            # the front end's own noise, a fraction of a division (0 by
            # default: the bench model's pd_noise stands for it at 1 V/div)
            if self.noise_per_div and self.roles.get(channel) == "PD":
                v = v + self.bench.rng.normal(0, self.noise_per_div * scale
                                              / math.sqrt(max(shots, 1)), v.size)
            v = v + self.offset_err * off
            # per-code error pattern, smeared by noise when averaged
            smear = math.exp(-0.5 * (2 * math.pi * 0.012 / code) ** 2) if shots > 1 else 1.0
            v = v + 0.0017 * scale * smear * np.sin(2 * np.pi * (v - off) / code)
            # 16-bit word over the converter's range, clipped at its edges
            half = 128 * code
            yinc = 2 * half / 65536
            codes = np.clip(np.round((v - off + half) / yinc), 0, 65535).astype(np.uint16)
            return sg.scope_profiles.Record((dt, t0, 0.0), codes=codes,
                                            y=(yinc, 32768.0, off))

        def screenshot(self):
            return None

    return SimScope


def make(sg, prof_key="msox2014a", roles=None, bench=None, zero_offset_deg=0.0):
    """(scope, ell14, bench) for a simulated bench. `roles` is {ch: role}."""
    bench = bench or Bench()
    prof = sg.scope_profiles.PROFILES[prof_key]
    SimScope = make_scope_class(sg)
    scope = SimScope(prof, bench, roles or {1: "PD", 3: "MonX1", 4: "MonX2"})
    scope.connect()
    return scope, FakeELL14(bench, zero_offset_deg), bench

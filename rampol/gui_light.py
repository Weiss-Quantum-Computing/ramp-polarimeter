"""The Light tab: the SRS DS345 gating the light (rampol/ds345.py), as a
mixin of the App so the main window file only carries the hooks.

    1 Gate      the light's level outside the windows (idle on / off), the
                windows (ms from the trigger) at the other level, edges;
                'From the shown scan': the hold, or everything but the rest
    2 Check     Preview (AWG plot tab, over the ramp) -> Dry run on the scope
                -> Load (armed on the bench trigger) / Park (DC, light on)
    Analyzer tab's refine: 'gate the light' loads, per window, the gate that
                keeps the light off wherever that analyzer angle would be bright

Levels are volts AT THE MODULATOR: the DS345 is specified into 50 Ohm and the
modulator is Hi-Z, so it is programmed half (ds345.program).
"""
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from . import config as cfgmod
from . import ds345 as dsmod
from . import hw

LIGHT_IDLE = ("on", "off")


class LightTab:
    # -- building -------------------------------------------------------------
    def build_light(self, f):
        self.ds_sess = None
        self.ds_gate = None          # the last gate built (preview / dry run / load)
        self.ds_dry = None
        self.light_view = None
        self.dv = {k: tk.StringVar() for k in
                   ("on_v", "off_v", "load_min_v", "load_max_v", "windows", "edge_us",
                    "record_ms", "dry_ch", "dry_shots", "margin_ms", "recover_ms")}
        self.ds_idle = tk.StringVar(value="on")
        if not hasattr(self, "ds_gate_refine"):          # made with the Analyzer tab
            self.ds_gate_refine = tk.BooleanVar(value=False)
        top = self._row(f, (2, 1))
        self.ds_lbl = ttk.Label(top, text="DS345: not connected", foreground="#666")
        self.ds_lbl.pack(side="left")
        ttk.Button(top, text="Settings...", command=self.open_light_settings).pack(side="right")
        self._btn(top, "Park", self.do_ds_park).pack_configure(side="right", padx=(0, 4))
        self.ds_note = ttk.Label(f, foreground="#666", wraplength=430, justify="left")
        self.ds_note.pack(anchor="w", padx=8)
        b = self._box(f, "1 Gate (ms from the bench trigger)")
        r = self._row(b)
        ttk.Label(r, text="light").pack(side="left")
        for v in LIGHT_IDLE:
            ttk.Radiobutton(r, text=v, value=v, variable=self.ds_idle,
                            command=self._ds_text).pack(side="left", padx=(4, 0))
        ttk.Label(r, text=", the other in").pack(side="left", padx=(4, 2))
        e = ttk.Entry(r, textvariable=self.dv["windows"], width=16)
        e.pack(side="left")
        ttk.Label(r, text="edges").pack(side="left", padx=(6, 2))
        ttk.Entry(r, textvariable=self.dv["edge_us"], width=4).pack(side="left")
        ttk.Label(r, text="us").pack(side="left", padx=(2, 0))
        r = self._row(b, (1, 3))
        ttk.Label(r, text="from the shown scan:").pack(side="left")
        ttk.Button(r, text="hold only", command=lambda: self.do_ds_from_scan("hold")).pack(
            side="left", padx=(4, 0))
        ttk.Button(r, text="all but the ramp", command=lambda: self.do_ds_from_scan("rest")).pack(
            side="left", padx=(4, 0))
        b = self._box(f, "2 Check, then play")
        r = self._row(b, (2, 1))
        ttk.Button(r, text="Preview", command=self.do_ds_preview).pack(side="left")
        self._btn(r, "Dry run on scope", self.do_ds_dry, padx=(4, 0))
        self._btn(r, "Load (play on trigger)", self.do_ds_load, padx=(4, 0))
        self.ds_check = ttk.Label(b, foreground="#666", wraplength=430, justify="left")
        self.ds_check.pack(anchor="w", padx=6, pady=(0, 3))
        for v in list(self.dv.values()) + [self.ds_idle]:
            v.trace_add("write", lambda *_: self._ds_text())

    def open_light_settings(self):
        f = self._dialog(self, "light_set_win", "DS345 light gate settings")
        if f is None:
            return
        dv = self.dv
        ttk.Label(f, text="Levels at the modulator (V, Hi-Z: the DS345 is programmed half)",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w", padx=6, pady=(0, 2))
        self._dline(f, ["light on", (dv["on_v"], 7), "V   light off", (dv["off_v"], 7), "V"])
        self._dline(f, ["the modulator takes", (dv["load_min_v"], 7), "to",
                        (dv["load_max_v"], 7), "V"],
                    "Set to the modulator input's rating: a gate outside it is refused. The "
                    "DS345 reaches +-10 V into Hi-Z (+-5 V programmed).")
        self._dline(f, ["record at least", (dv["record_ms"], 6), "ms (0: to the last window "
                        "+ 0.5 ms)"],
                    "The record is played once per trigger; the sample clock (40 MHz / N) is "
                    "chosen to fit 16,300 points.")
        ttk.Label(f, text="Dry run", font=("TkDefaultFont", 9, "bold")).pack(
            anchor="w", padx=6, pady=(6, 2))
        self._dline(f, ["DS345 output teed to scope CH", (dv["dry_ch"], 3),
                        (dv["dry_shots"], 3), "shots"])
        ttk.Label(f, text="Gating the null refine (Analyzer tab)",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        self._dline(f, ["light on from", (dv["margin_ms"], 5), "ms into a hold; back on",
                        (dv["recover_ms"], 5), "ms after the ramp"])
        ttk.Button(f, text="Close", command=f._close).pack(anchor="e", padx=6, pady=(6, 0))

    # -- settings --------------------------------------------------------------
    def _light_load(self, c):
        d = c.get("ds345", {})
        for k, v in self.dv.items():
            v.set(str(d.get(k, "")))
        self.ds_idle.set(d.get("idle", "on") if d.get("idle") in LIGHT_IDLE else "on")
        self.ds_gate_refine.set(bool(d.get("gate_refine", False)))
        self.ds_addr.set(c.get("ds345_addr", ""))
        self._ds_text()

    def _light_gather(self, c):
        d = c.setdefault("ds345", {})
        for k, v in self.dv.items():
            txt = v.get().strip()
            if k == "windows":
                d[k] = txt
                continue
            try:
                d[k] = int(float(txt)) if k in ("dry_ch", "dry_shots") else float(txt)
            except ValueError:
                pass
        d["idle"] = self.ds_idle.get()
        d["gate_refine"] = bool(self.ds_gate_refine.get())
        c["ds345_addr"] = self.ds_addr.get().strip()

    def _ds_levels(self, c=None):
        d = (c or self.gather())["ds345"]
        return (float(d.get("on_v", 1.0)), float(d.get("off_v", 0.0)),
                float(d.get("load_min_v", 0.0)), float(d.get("load_max_v", 1.0)))

    def _ds_text(self):
        try:
            on, off = float(self.dv["on_v"].get()), float(self.dv["off_v"].get())
            lo, hi = float(self.dv["load_min_v"].get()), float(self.dv["load_max_v"].get())
        except (ValueError, KeyError):
            return
        bad = not (lo <= min(on, off) and max(on, off) <= hi)
        self.ds_note.configure(
            text=f"on {on:+.3f} V, off {off:+.3f} V at the modulator (allowed "
                 f"{lo:+.3f}..{hi:+.3f} V){' - OUTSIDE: Settings...' if bad else ''}. "
                 f"Bench trigger -> DS345 rear TRIGGER IN.",
            foreground="#c00000" if bad else "#666")

    # -- the instrument --------------------------------------------------------
    def _ds_session(self, c):
        """Worker: the DS345 session, connecting on first use (GPIB; the
        Ds345 class of the DS345 panel, on this window's shared VISA RM)."""
        if self.ds_sess is not None:
            return self.ds_sess
        if self.bench is not None:
            from . import sim
            dev = getattr(self.bench, "ds345", None) or sim.FakeDS345(self.bench)
        else:
            mod = hw.load_module(c["ds345_path"], "ds345_awg_gui")
            dev = _SharedDs345(mod, hw.shared_rm(mod.pyvisa))
            remembered = ""
            try:
                import json
                p = os.path.join(os.environ.get("APPDATA", ""), "DS345-AWG-GUI", "config.json")
                with open(p, encoding="utf-8") as fh:
                    remembered = json.load(fh).get("address", "") or ""
            except (OSError, ValueError):
                pass
            dev.connect(c.get("ds345_addr") or None, remembered=remembered or None,
                        log=lambda s: None)
        self.log(f"DS345: {dev.idn} on {dev.addr}")
        self.ds_sess = dsmod.Session(dev, log=self.log)
        return self.ds_sess

    def do_connect_ds345(self):
        c = self.gather()
        roles = self.roles()

        def go():
            if c["simulate"] and self.bench is None:
                self.ensure_sim(roles)
            self._ds_session(c)
        self.worker(go, done=lambda _r: self._ds_status())

    def _ds_status(self):
        s = self.ds_sess
        if s is None:
            self.ds_lbl.configure(text="DS345: not connected (connects on first use)",
                                  foreground="#666")
            self.ds_hw.configure(text="not connected", foreground="#666")
            return
        where = ("simulated DS345" if self.bench is not None and
                 getattr(self.bench, "ds345", None) is s.dev else
                 f"{getattr(s.dev, 'idn', '')[:28]} on {getattr(s.dev, 'addr', '?')}")
        self.ds_hw.configure(text=where, foreground="#060")
        if s.gate is not None:
            txt = "gating: " + s.gate.label + ("" if s.is_verified(s.gate) else
                                               " - NOT dry-run")
            col = "#060" if s.is_verified(s.gate) else "#c60"
        elif s.parked_v is not None:
            txt, col = f"parked: DC {s.parked_v:+.3f} V (light on, ungated)", "#060"
        else:
            txt, col = "connected; nothing of this window's loaded", "#666"
        self.ds_lbl.configure(text="DS345 " + txt, foreground=col)

    # -- the gate ----------------------------------------------------------------
    def _ds_build(self, c):
        d = c["ds345"]
        on, off, lo, hi = self._ds_levels(c)
        g = dsmod.build(dsmod.parse_windows(d.get("windows", "")), d.get("idle", "on"),
                        on, off, float(d.get("edge_us", 20.0) or 0),
                        float(d.get("record_ms", 0.0) or 0))
        found = dsmod.check(g, lo, hi, trig_hz=c["awg"].get("trig_hz"))
        return g, found

    def do_ds_from_scan(self, kind):
        res = getattr(self, "result", None)
        if not res or not res.get("pol"):
            self.log("Light: load a scan first (its fit gives the ramp's timing).")
            return
        d = self.gather()["ds345"]
        idle, ws = dsmod.windows_for(res["pol"], kind,
                                     float(d.get("margin_ms", 0.3)) * 1e-3,
                                     float(d.get("recover_ms", 0.5)) * 1e-3)
        self.ds_idle.set(idle)
        self.dv["windows"].set(", ".join(f"{a*1e3:.2f}-{b*1e3:.2f}" for a, b in ws))
        self.log(f"Light gate from {res['d'].name}: light {idle}, "
                 + ("off" if idle == "on" else "on") + " in "
                 + (self.dv["windows"].get() or "nothing"))
        self.do_ds_preview()

    def do_ds_preview(self):
        c = self.gather()
        self.save_settings()
        try:
            g, found = self._ds_build(c)
        except ValueError as exc:
            self.log(f"Light: {exc}")
            return
        self.ds_gate = g
        self.report_checks(found, f"DS345 gate: {g.label}", popup=False)
        self.ds_check.configure(text=f"{dsmod.worst(found)}: "
                                + "; ".join(m for lv, m in found if lv != "INFO")[:300]
                                if dsmod.worst(found) != "OK" else
                                f"OK: {g.n} points at {g.dt*1e6:g} us", foreground=
                                "#c00000" if dsmod.worst(found) == "FAIL" else "#666")
        self.light_view = {"gate": g, "awg": (id(self.awg_wave), id(self.awg_set))}
        self.plot_dirty.add(self.fig_awg._frame)
        self.nb.select(self.fig_awg._frame)
        self.draw_visible()

    def draw_light(self, fig):
        """The gate (volts at the modulator) and, under it, the light it lets
        through, over the ramp's rotation when the AWG has one."""
        g = self.light_view["gate"]
        w = self.awg_wave or (self.awg_sess.wave if self.awg_sess is not None else None)
        ax = fig.add_subplot(211)
        tt = np.concatenate([[-1e-3], g.t, [g.period + 2e-3]]) * 1e3
        vv = np.concatenate([[g.idle_v], g.v, [g.idle_v]])
        ax.plot(tt, vv, color="#d62728", lw=1.0, label="DS345 output at the modulator")
        dry = self.ds_dry
        if dry is not None and dry.get("label") == g.label and dry.get("result"):
            t_, v_ = dry["result"]["trace"]
            ax.plot(np.asarray(t_) * 1e3, v_, color="0.4", lw=0.6,
                    label=f"dry run, scope CH{dry['scope_ch']}")
        ax.set_ylabel("V at the modulator")
        ax.set_title(f"Light gate: {g.label}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)
        ax2 = fig.add_subplot(212, sharex=ax)
        ax2.plot(tt, g.light(tt * 1e-3) * 100, color="#d62728", lw=1.0, label="light (%)")
        ax2.set_ylabel("light let through (%)")
        ax2.set_ylim(-5, 105)
        ax2.set_xlabel("time from the bench trigger (ms)")
        if w is not None:
            from . import awg as awgmod
            _mon, rot = awgmod.predict(w)
            a3 = ax2.twinx()
            a3.plot(w.t * 1e3, rot, color="k", lw=0.8, label=f"rotation: {w.label}")
            a3.set_ylabel("rotation (deg)")
            a3.legend(fontsize=7, loc="lower right")
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(alpha=0.3)

    def _ds_ready(self, c, what):
        """(gate, found) or None after saying why."""
        try:
            g, found = self._ds_build(c)
        except ValueError as exc:
            self.log(f"Light: {exc}")
            return None
        if self.report_checks(found, f"DS345 gate: {g.label}") == "FAIL":
            self.log(f"{what} not done.")
            return None
        return g, found

    def do_ds_dry(self):
        if not self.need(ell=False):
            return
        c = self.gather()
        self.save_settings()
        got = self._ds_ready(c, "Dry run")
        if got is None:
            return
        g = got[0]
        d = c["ds345"]
        ch = int(d.get("dry_ch", 2))
        roles = {ch_: r for ch_, (r, _n) in cfgmod.channel_roles(self.cfg).items()}
        busy = f" (it normally carries {roles[ch]}: its cable comes off for this)" \
            if ch in roles else ""
        if not messagebox.askokcancel(
                "DS345 dry run",
                f"The DS345 output teed to scope CH{ch}{busy}, the bench trigger into the "
                f"DS345's rear TRIGGER IN.\n\nThe gate plays on every trigger; the scope "
                f"reads it at Hi-Z, as the modulator does, so a 50 Ohm mistake shows as a "
                f"gain of 0.5. The light follows it meanwhile.\n\nStart?", parent=self.root):
            return
        sim_mode = self.bench is not None

        def go():
            sess = self._ds_session(c)
            if sim_mode:
                self.bench.ds_wiring, self.bench.ds_scope_ch = "scope", ch
            try:
                return dsmod.dry_run(sess, self.link, g, ch, shots=int(d.get("dry_shots", 4)),
                                     wait_s=float(c["scan"].get("wait_s", 10.0)),
                                     cancelled=self.stop_flag.is_set, log=self.log)
            finally:
                if sim_mode:
                    self.bench.ds_wiring = "modulator"

        def done(rep):
            self.ds_dry = rep
            if rep["ok"]:
                self.log(f"DS345 dry run PASSED: {g.label}")
            else:
                self.log("DS345 dry run FAILED:")
                for p in rep["problems"]:
                    self.log(f"  {p}")
                messagebox.showwarning("DS345 dry run failed", "\n\n".join(rep["problems"][:5]),
                                       parent=self.root)
            self._ds_status()
            self.light_view = {"gate": g, "awg": (id(self.awg_wave), id(self.awg_set))}
            self.plot_dirty.add(self.fig_awg._frame)
            self.nb.select(self.fig_awg._frame)
            self.draw_visible()
        self.worker(go, done=done)

    def do_ds_load(self):
        c = self.gather()
        self.save_settings()
        got = self._ds_ready(c, "Load")
        if got is None:
            return
        g = got[0]
        sess = self.ds_sess
        if (sess is None or not sess.is_verified(g)) and not messagebox.askyesno(
                "Not dry-run", "This gate has not passed a dry run on the scope this "
                "session. Load it onto the modulator anyway?", parent=self.root):
            return
        self.worker(lambda: self._ds_session(c).load(g), done=lambda _r: self._ds_status())

    def do_ds_park(self):
        c = self.gather()
        on = self._ds_levels(c)[0]
        self.worker(lambda: self._ds_session(c).park(on), done=lambda _r: self._ds_status())

    def _ds_close(self):
        """On close: the light back on (DC at on_v), ungated."""
        s = self.ds_sess
        if s is None:
            return
        try:
            if s.gate is not None:
                s.park(self._ds_levels(self.cfg)[0])
            s.dev.close()
        except Exception as exc:
            self.log(f"DS345 close: {exc}")
        self.ds_sess = None

    # -- gating the null refine -----------------------------------------------------
    def ds_refine_gates(self, plans, pol, c):
        """{window: gate dict} for refine `plans`: a window in a hold gets the
        light only in the hold, any other window the light off during the
        ramp. Raises ValueError when a gate is refused."""
        from . import analysis as an
        d = c["ds345"]
        on, off, lo, hi = self._ds_levels(c)
        segs = an.segments(pol["t"], pol["rotation"])
        out = {}
        for p in plans:
            mid = 0.5 * (p["t0"] + p["t1"])
            seg = next((s for s in segs if s["t0"] <= mid <= s["t1"]), None)
            kind = "hold" if seg is not None and seg["base"] == "hold" else "rest"
            idle, ws = dsmod.windows_for(pol, kind, float(d.get("margin_ms", 0.3)) * 1e-3,
                                         float(d.get("recover_ms", 0.5)) * 1e-3)
            spec = {"kind": kind, "idle": idle, "windows_ms": [[a * 1e3, b * 1e3] for a, b in ws],
                    "on_v": on, "off_v": off, "edge_us": float(d.get("edge_us", 20.0) or 0)}
            g = gate_of(spec)
            bad = [m for lv, m in dsmod.check(g, lo, hi, c["awg"].get("trig_hz")) if lv == "FAIL"]
            if bad:
                raise ValueError(f"window {p['label']}: " + "; ".join(bad))
            out[p["window"]] = spec
        return out

    def _run_gated(self, run, kinds, c):
        """Worker: the run's steps not done (of `kinds`), the DS345 loaded
        with each step's gate before it; parked (light on) at the end."""
        sess = self._ds_session(c)
        todo = [s for s in run.manifest["steps"] if s["status"] != "done" and s["kind"] in kinds]
        cur, n = None, 0
        try:
            for i, s in enumerate(todo):
                if self.stop_flag.is_set():
                    raise hw.Cancelled()
                spec = s.get("gate")
                key = None if spec is None else repr(sorted(spec.items()))
                if key != cur:
                    if spec is None:
                        sess.park(self._ds_levels(c)[0])
                    else:
                        sess.load(gate_of(spec))
                    cur = key
                self._progress(i, len(todo), f"{s['kind']} {s['target']:.2f} deg ("
                                             f"{i + 1}/{len(todo)}), light "
                                             + ("ungated" if spec is None else
                                                f"{spec['kind']} gate"))
                run.run_step(s, on_step=lambda _s: self.call(self.live_refresh, run.folder))
                n += 1
            self._progress(len(todo), len(todo), "done")
        except hw.Cancelled:
            self.log(f"Stopped after {n} step(s).")
        finally:
            try:
                sess.park(self._ds_levels(c)[0])
            except Exception as exc:
                self.log(f"  DS345 park at the end: {exc}")
            self.call(self._ds_status)
        return n


def gate_of(spec):
    """A ds345.Gate from a step's 'gate' record."""
    return dsmod.build([(a * 1e-3, b * 1e-3) for a, b in spec["windows_ms"]], spec["idle"],
                       spec["on_v"], spec["off_v"], spec.get("edge_us", 20.0))


class _SharedDs345:
    """The DS345 panel's Ds345 on this window's shared VISA resource manager:
    its own connect() opens (and close() closes) a manager of its own, and a
    closed manager takes every session on it down (the scope / AWG trap)."""

    def __init__(self, mod, rm):
        self._dev = mod.Ds345()
        self._rm = rm

    def connect(self, addr=None, remembered=None, log=lambda s: None):
        dev = self._dev
        orig = dev.close
        dev.close = lambda: None                      # do not close the shared manager
        try:
            import pyvisa
            real = pyvisa.ResourceManager
            pyvisa.ResourceManager = lambda *a, **k: self._rm
            try:
                return dev.connect(addr, remembered=remembered, log=log)
            finally:
                pyvisa.ResourceManager = real
        finally:
            dev.close = orig

    def close(self):
        try:
            if self._dev.inst is not None:
                self._dev.inst.close()
        except Exception:
            pass
        self._dev.inst = None

    def __getattr__(self, name):
        return getattr(self._dev, name)

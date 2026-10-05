"""The Ramp Polarimeter window.

Controls on the left (hardware, analyzer, channel roles, scan, null refine),
plots on the right in Scope Grab's arrangement: a plot bar above a notebook of
figure tabs, each with its matplotlib toolbar, and the log underneath.

Threading: every instrument operation runs on one worker thread at a time.
The worker never touches Tk; it hands results back through self.call(),
which the pump runs on the Tk thread.

    python polarimeter.py            (or python -m rampol, or Run in VS Code)
"""
import os
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from . import __version__
from . import analysis as an
from . import config as cfgmod
from . import hw, scan as scanmod, sim

ANGLE_CMAP = "hsv"                 # cyclic: 0 and 360 deg share a colour, none near white
CMP_COLOUR = "#ff7f0e"
PLOT_HINT = ("Click a time on the Map, Angle or Extinction tab to move the "
             "cursor the Malus tab shows.")


class App:
    def __init__(self, root):
        self.root = root
        self.msgs = queue.Queue()
        self.calls = queue.Queue()
        self.cfg = cfgmod.load()
        self.sg = None
        self.link = None            # hw.ScopeLink once connected
        self.rot = None             # hw.Rotator once connected
        self.bench = None           # sim.Bench in simulate mode
        self.busy = False
        self.stop_flag = threading.Event()
        self.run = None             # scan.ScanRun being measured
        self.result = None          # analysis of the scan on show
        self.compare = None         # analysis of the compare scan
        self.cursor_t = None
        self.plot_tabs = {}
        self.plot_dirty = set()
        self.busy_widgets = []

        root.title(f"Ramp Polarimeter {__version__}")
        win_w = min(1360, root.winfo_screenwidth() - 80)
        win_h = min(960, root.winfo_screenheight() - 120)
        root.geometry(f"{win_w}x{win_h}+40+20")
        matplotlib.rcParams["font.size"] = 8

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.build_hardware(left)
        self.build_analyzer(left)
        self.build_channels(left)
        self.build_scan(left)
        self.build_refine(left)
        self.build_right(right)
        self.load_settings()
        self.refresh_scan_list()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.pump()
        self.log(f"Ramp Polarimeter {__version__}. Config: {cfgmod.CONFIG_PATH}")
        self.load_sg(quiet=True)

    # -- plumbing -----------------------------------------------------------
    def log(self, text):
        self.msgs.put(text)

    def call(self, fn, *args):
        """Run fn(*args) on the Tk thread (from any thread)."""
        self.calls.put((fn, args))

    def pump(self):
        while not self.msgs.empty():
            self.logbox.insert("end", self.msgs.get() + "\n")
            self.logbox.see("end")
        while not self.calls.empty():
            fn, args = self.calls.get()
            try:
                fn(*args)
            except Exception as exc:
                self.log(f"ERROR (display): {exc}")
        self.root.after(80, self.pump)

    def worker(self, fn, *args, done=None):
        """Run fn on the worker thread. Refuses if one is already running."""
        if self.busy:
            self.log("Busy - wait for the current operation, or Stop it.")
            return False
        self.set_busy(True)
        self.stop_flag.clear()

        def body():
            ok, out = False, None
            try:
                out = fn(*args)
                ok = True
            except hw.Cancelled:
                self.log("Stopped.")
            except Exception as exc:
                self.log(f"ERROR: {exc}")
                # the whole chain: an ImportError's real cause is the one
                # BEFORE the last few lines
                self.log(traceback.format_exc().rstrip())
            finally:
                # not busy BEFORE the callback runs, so a callback can start
                # the next operation (dark -> unblock prompt -> scan)
                self.call(self._finish, ok, out, done)

        threading.Thread(target=body, daemon=True).start()
        return True

    def _finish(self, ok, out, done):
        self.set_busy(False)
        if ok and done:
            done(out)

    def set_busy(self, busy):
        self.busy = busy
        for w in self.busy_widgets:
            try:
                w.configure(state="disabled" if busy else "normal")
            except tk.TclError:
                pass
        self.stop_btn.configure(state="normal" if busy else "disabled")
        if not busy:
            self.progress_bar["value"] = 0

    def _btn(self, parent, text, cmd, busy=True, **pack):
        b = ttk.Button(parent, text=text, command=cmd)
        b.pack(side="left", **pack)
        if busy:
            self.busy_widgets.append(b)
        return b

    # -- left column ----------------------------------------------------------
    def build_hardware(self, left):
        f = ttk.LabelFrame(left, text="Hardware")
        f.pack(fill="x", padx=8, pady=(6, 3))
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Scope VISA:").pack(side="left")
        self.scope_addr = tk.StringVar()
        ttk.Entry(r, textvariable=self.scope_addr, width=22).pack(side="left", padx=4)
        self._btn(r, "Connect", self.do_connect_scope)
        # Fixed width + wrap: an instrument's identity or a VISA address is long
        # and a label sized to it widened the whole left column.
        self.scope_status = ttk.Label(f, text="scope: not connected", foreground="#666",
                                      width=48, wraplength=330)
        self.scope_status.pack(anchor="w", padx=6)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="ELL14 port:").pack(side="left")
        self.ell_port = tk.StringVar()
        ttk.Entry(r, textvariable=self.ell_port, width=8).pack(side="left", padx=4)
        ttk.Label(r, text="addr").pack(side="left")
        self.ell_addr = tk.StringVar()
        ttk.Entry(r, textvariable=self.ell_addr, width=3).pack(side="left", padx=4)
        self._btn(r, "Connect", self.do_connect_ell)
        self.ell_status = ttk.Label(f, text="analyzer: not connected", foreground="#666",
                                    width=48, wraplength=330)
        self.ell_status.pack(anchor="w", padx=6)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=(2, 4))
        self.simulate = tk.BooleanVar()
        ttk.Checkbutton(r, text="Simulate both (no hardware)",
                        variable=self.simulate).pack(side="left")
        self._btn(r, "Disconnect all", self.do_disconnect, padx=(12, 0))
        ttk.Button(r, text="Scope settings...", command=self.open_scope_settings).pack(
            side="left", padx=(8, 0))

    def build_analyzer(self, left):
        f = ttk.LabelFrame(left, text="Analyzer (ELL14)")
        f.pack(fill="x", padx=8, pady=3)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=2)
        self.pos_label = ttk.Label(r, text="position: -", width=24)
        self.pos_label.pack(side="left")
        self._btn(r, "Read", self.do_read_pos)
        self._btn(r, "Home", self.do_home, padx=4)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Go to").pack(side="left")
        self.goto_var = tk.StringVar(value="0")
        e = ttk.Entry(r, textvariable=self.goto_var, width=8)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _e: self.do_goto())
        self._btn(r, "Go", self.do_goto)
        for d in (-10, -1, 1, 10):
            self._btn(r, f"{d:+d}", lambda d=d: self.do_jog(d), padx=(4 if d == -10 else 1, 0))
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Label(r, text="Zero = mount").pack(side="left")
        self.zero_var = tk.StringVar()
        ttk.Entry(r, textvariable=self.zero_var, width=9).pack(side="left", padx=4)
        ttk.Label(r, text="deg").pack(side="left")
        self._btn(r, "Apply", self.do_apply_zero, padx=4)
        self._btn(r, "Set zero from rest", self.do_zero_from_rest)

    def build_channels(self, left):
        f = ttk.LabelFrame(left, text="Scope channels")
        f.pack(fill="x", padx=8, pady=3)
        self.ch_role, self.ch_name = {}, {}
        for ch in (1, 2, 3, 4):
            r = ttk.Frame(f)
            r.pack(fill="x", padx=6, pady=1)
            ttk.Label(r, text=f"CH{ch}").pack(side="left")
            self.ch_role[ch] = tk.StringVar()
            ttk.Combobox(r, textvariable=self.ch_role[ch], values=cfgmod.ROLES,
                         width=7, state="readonly").pack(side="left", padx=4)
            self.ch_name[ch] = tk.StringVar()
            ttk.Entry(r, textvariable=self.ch_name[ch], width=24).pack(side="left")
        ttk.Label(f, foreground="#666", justify="left", wraplength=330,
                  text="One PD is required. MonX1/MonX2 give the monitor "
                       "prediction; Ref (a pick-off before the analyzer) "
                       "normalises intensity per sample.").pack(anchor="w", padx=6, pady=(0, 4))

    def build_scan(self, left):
        f = ttk.LabelFrame(left, text="Scan")
        f.pack(fill="x", padx=8, pady=3)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Preset").pack(side="left")
        self.preset = tk.StringVar()
        cb = ttk.Combobox(r, textvariable=self.preset, values=list(cfgmod.all_presets(self.cfg)),
                          width=24, state="readonly")
        cb.pack(side="left", padx=4)
        self.preset_box = cb
        cb.bind("<<ComboboxSelected>>", lambda _e: self.preset_picked())
        self._btn(r, "Apply to scope", self.do_apply_preset)
        self.sv = {}

        def row(items):
            rr = ttk.Frame(f)
            rr.pack(fill="x", padx=6, pady=1)
            for label, key, w in items:
                ttk.Label(rr, text=label).pack(side="left", padx=(0, 2))
                self.sv[key] = tk.StringVar()
                ttk.Entry(rr, textvariable=self.sv[key], width=w).pack(side="left", padx=(0, 6))
            return rr

        row([("Angles from", "start", 6), ("to", "stop", 6), ("step", "step", 5)])
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        ttk.Label(rr, text="Order").pack(side="left")
        self.order = tk.StringVar()
        ttk.Combobox(rr, textvariable=self.order, width=13, state="readonly",
                     values=("forward", "bidirectional", "shuffled")).pack(side="left", padx=4)
        ttk.Label(rr, text="Mode").pack(side="left")
        self.mode = tk.StringVar()
        ttk.Combobox(rr, textvariable=self.mode, width=8, state="readonly",
                     values=("average", "single")).pack(side="left", padx=4)
        row([("Shots/angle", "shots", 5), ("blocks (avg)", "blocks", 3), ("dither codes", "dither_codes", 3)])
        row([("Ref every", "ref_every", 4), ("angles, at", "ref_angle", 6), ("backoff", "backoff_deg", 4)])
        row([("Points (single)", "points", 7), ("trig wait s", "wait_s", 4), ("rep s", "rep_s", 5)])
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        ttk.Label(rr, text="Folder").pack(side="left")
        self.outdir = tk.StringVar()
        ttk.Entry(rr, textvariable=self.outdir, width=30).pack(side="left", padx=4)
        self._btn(rr, "...", self.pick_outdir)
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        ttk.Label(rr, text="Scan name").pack(side="left")
        self.scan_name = tk.StringVar()
        ttk.Entry(rr, textvariable=self.scan_name, width=24).pack(side="left", padx=4)
        self.with_dark = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="dark first", variable=self.with_dark).pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(4, 2))
        self._btn(rr, "Start scan", self.do_start_scan)
        self.stop_btn = ttk.Button(rr, text="Stop", command=self.do_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.est_label = ttk.Label(rr, text="", foreground="#666")
        self.est_label.pack(side="left")
        self.progress_bar = ttk.Progressbar(f, mode="determinate", maximum=1)
        self.progress_bar.pack(fill="x", padx=6, pady=(2, 1))
        self.progress_text = ttk.Label(f, text="", foreground="#060", width=48,
                                       wraplength=330)
        self.progress_text.pack(anchor="w", padx=6, pady=(0, 4))
        for v in list(self.sv.values()) + [self.order, self.mode]:
            v.trace_add("write", lambda *_: self.update_estimate())

    def build_refine(self, left):
        f = ttk.LabelFrame(left, text="Null refine (static parts)")
        f.pack(fill="x", padx=8, pady=3)
        self.rv = {}
        for label, key, w in (("Windows (ms)", "windows", 22),
                              ("Offsets (deg)", "offsets", 22),
                              ("PD V/div at null", "pd_vdiv", 8)):
            rr = ttk.Frame(f)
            rr.pack(fill="x", padx=6, pady=1)
            ttk.Label(rr, text=label, width=15).pack(side="left")
            self.rv[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.rv[key], width=w).pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(2, 4))
        self._btn(rr, "Plan", self.do_plan_refine)
        self._btn(rr, "Run refine on shown scan", self.do_run_refine, padx=6)
        ttk.Label(f, foreground="#666", justify="left", wraplength=330,
                  text="'auto' = rest, hold and after-ramp windows. At a "
                       "sensitive V/div the bright part of the ramp is off "
                       "screen; a window right after it can carry the front "
                       "end's overdrive recovery.").pack(anchor="w", padx=6, pady=(0, 4))

    # -- right side -----------------------------------------------------------
    def build_right(self, right):
        bar = ttk.LabelFrame(right, text="Plot data")
        bar.pack(fill="x", padx=8, pady=(6, 0))
        r = ttk.Frame(bar)
        r.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(r, text="Scan:").pack(side="left")
        self.show_scan = tk.StringVar()
        self.scan_box = ttk.Combobox(r, textvariable=self.show_scan, width=34)
        self.scan_box.pack(side="left", padx=4)
        self.scan_box.bind("<<ComboboxSelected>>", lambda _e: self.do_load_shown())
        self.scan_box.bind("<Return>", lambda _e: self.do_load_shown())
        ttk.Button(r, text="Open...", command=self.do_open_scan).pack(side="left")
        ttk.Button(r, text="Rescan folder", command=self.refresh_scan_list).pack(side="left", padx=4)
        ttk.Label(r, text="Compare:").pack(side="left", padx=(12, 0))
        self.cmp_scan = tk.StringVar()
        self.cmp_box = ttk.Combobox(r, textvariable=self.cmp_scan, width=24)
        self.cmp_box.pack(side="left", padx=4)
        self.cmp_box.bind("<<ComboboxSelected>>", lambda _e: self.do_load_compare())
        ttk.Button(r, text="Clear", command=self.do_clear_compare).pack(side="left")
        r = ttk.Frame(bar)
        r.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(r, text="Cursor (ms):").pack(side="left")
        self.cursor_var = tk.StringVar()
        e = ttk.Entry(r, textvariable=self.cursor_var, width=8)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _e: self.set_cursor_text())
        ttk.Label(r, text="Smooth (us):").pack(side="left", padx=(10, 0))
        self.smooth_us = tk.StringVar(value="20")
        e = ttk.Entry(r, textvariable=self.smooth_us, width=5)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _e: self.mark_dirty())
        self.drift_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(r, text="drift-correct from refs", variable=self.drift_on,
                        command=self.reanalyse).pack(side="left", padx=(10, 0))
        self.plot_status = ttk.Label(r, text=PLOT_HINT, foreground="#666")
        self.plot_status.pack(side="left", padx=(12, 0))

        pane = ttk.PanedWindow(right, orient="vertical")
        pane.pack(fill="both", expand=True, padx=8, pady=4)
        self.nb = ttk.Notebook(pane)
        pane.add(self.nb, weight=5)
        logf = ttk.Frame(pane)
        pane.add(logf, weight=1)
        self.logbox = tk.Text(logf, height=8, font=("Consolas", 9), wrap="word")
        sb = ttk.Scrollbar(logf, command=self.logbox.yview)
        self.logbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.logbox.pack(fill="both", expand=True)

        self.fig_traces = self._fig_tab("Traces", self.draw_traces)
        self.fig_map = self._fig_tab("Map", self.draw_map, click=True)
        self.fig_malus = self._fig_tab("Malus", self.draw_malus)
        self.fig_angle = self._fig_tab("Angle", self.draw_angle, click=True)
        self.fig_ext = self._fig_tab("Extinction", self.draw_extinction, click=True)
        ctl = self.fig_ext._ctl
        ttk.Label(ctl, text="x axis:").pack(side="left")
        self.ext_x = tk.StringVar(value="time")
        cb = ttk.Combobox(ctl, textvariable=self.ext_x, values=("time", "rotation"),
                          width=9, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw(self.fig_ext))
        self.fig_diag = self._fig_tab("Diagnostics", self.draw_diagnostics)
        self.build_table_tab()
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self.draw_visible())

    def _fig_tab(self, name, draw, click=False):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=name)
        ctl = ttk.Frame(frame)
        ctl.pack(fill="x", padx=4, pady=(4, 0))
        fig = Figure(figsize=(7.2, 5.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._canvas, fig._toolbar, fig._ctl, fig._frame = canvas, toolbar, ctl, frame
        self.plot_tabs[frame] = (fig, draw)
        self.plot_dirty.add(frame)
        if click:
            canvas.mpl_connect("button_press_event",
                               lambda ev, fig=fig: self.on_click(ev, fig))
        return fig

    def build_table_tab(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="Table")
        ctl = ttk.Frame(frame)
        ctl.pack(fill="x", padx=4, pady=4)
        ttk.Label(ctl, text="Segments, then every direct extinction-ratio "
                            "measurement", foreground="#666").pack(side="left")
        ttk.Button(ctl, text="Save CSV...", command=self.save_table).pack(side="right")
        heads = ("what", "t (ms)", "rotation (deg)", "analyzer (deg)",
                 "rate (deg/us)", "ER", "ER light", "Imin (mV)", "sigma (mV)",
                 "visibility", "sigma psi (mdeg)", "note")
        self.tv = ttk.Treeview(frame, columns=heads, show="headings", height=20)
        for h, w in zip(heads, (110, 110, 90, 90, 80, 80, 80, 80, 80, 80, 90, 200)):
            self.tv.heading(h, text=h)
            self.tv.column(h, width=w, anchor="e" if h != "what" and h != "note" else "w")
        sb = ttk.Scrollbar(frame, command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(fill="both", expand=True, padx=4)
        self.table_heads = heads
        self.plot_tabs[frame] = (None, self.fill_table)
        self.plot_dirty.add(frame)

    # -- settings -------------------------------------------------------------
    def load_settings(self):
        # Setting a variable fires update_estimate, which gathers the window
        # back into the config - with half of it still unset. Hold it off
        # until everything is in.
        self._loading = True
        try:
            self._load_settings()
        finally:
            self._loading = False
        self.update_estimate()

    def _load_settings(self):
        c = self.cfg
        self.scope_addr.set(c["scope_addr"])
        self.ell_port.set(c["ell_port"])
        self.ell_addr.set(c["ell_address"])
        self.simulate.set(bool(c["simulate"]))
        self.zero_var.set(f"{float(c['ell_zero_deg']):.3f}")
        for ch in (1, 2, 3, 4):
            cc = c["channels"].get(str(ch), {"role": "off", "name": ""})
            self.ch_role[ch].set(cc.get("role", "off"))
            self.ch_name[ch].set(cc.get("name", ""))
        s = c["scan"]
        for k, v in self.sv.items():
            v.set(str(s.get(k, "")))
        self.order.set(s["order"])
        self.mode.set(s["mode"])
        self.preset.set(c["preset"])
        self.outdir.set(c["outdir"])
        self.scan_name.set(c["scan_name"])
        for k, v in self.rv.items():
            v.set(str(c["refine"].get(k, "")))

    def gather(self):
        """The window's values into self.cfg (validated where it matters)."""
        c = self.cfg
        c["scope_addr"] = self.scope_addr.get().strip()
        c["ell_port"] = self.ell_port.get().strip()
        c["ell_address"] = self.ell_addr.get().strip() or "0"
        c["simulate"] = bool(self.simulate.get())
        try:
            c["ell_zero_deg"] = float(self.zero_var.get())
        except ValueError:
            pass
        for ch in (1, 2, 3, 4):
            c["channels"][str(ch)] = {"role": self.ch_role[ch].get(),
                                      "name": self.ch_name[ch].get().strip()}
        s = c["scan"]
        for k, v in self.sv.items():
            txt = v.get().strip()
            try:
                s[k] = int(txt) if k in ("shots", "blocks", "dither_codes", "ref_every", "points") else float(txt)
            except ValueError:
                pass
        s["order"], s["mode"] = self.order.get(), self.mode.get()
        c["preset"] = self.preset.get()
        c["outdir"] = self.outdir.get().strip()
        c["scan_name"] = self.scan_name.get().strip()
        for k, v in self.rv.items():
            c["refine"][k] = float(v.get()) if k == "pd_vdiv" and _isnum(v.get()) else v.get().strip()
        return c

    def save_settings(self):
        try:
            cfgmod.save(self.gather())
        except OSError as exc:
            self.log(f"Could not save settings: {exc}")

    def update_estimate(self):
        if getattr(self, "_loading", True):
            return
        try:
            s = self.gather()["scan"]
            angles = scanmod.angle_list(s["start"], s["stop"], s["step"])
            refs = (len(angles) // max(s["ref_every"], 1) + 2) if s["ref_every"] > 0 else 0
            n = len(angles) + refs
            rep = max(float(s.get("rep_s", 0.27)), 0.01)
            # a single shot cannot be re-armed faster than it is read out (~0.6 s)
            per = (s["shots"] * rep + s["blocks"] * 0.8 if s["mode"] == "average"
                   else s["shots"] * max(rep, 0.6)) + 1.5
            mins = n * per / 60
            self.est_label.configure(
                text=f"{len(angles)} angles + {refs} refs, ~"
                     + (f"{mins:.0f} min" if mins < 90 else f"{mins / 60:.1f} h")
                     + f" at {rep:g} s/shot")
        except Exception:
            self.est_label.configure(text="")

    def preset_picked(self):
        p = cfgmod.all_presets(self.cfg).get(self.preset.get())
        if not p:
            return
        for k, v in (p.get("scan") or {}).items():
            if k == "mode":
                self.mode.set(v)
            elif k in self.sv:
                self.sv[k].set(str(v))
        self.log(f"Preset {self.preset.get()}: {p.get('note', '')}")
        if p.get("scope"):
            self.log("  'Apply to scope' writes: " + ", ".join(
                f"{k} {v}" for k, v in p["scope"].items()))

    def pick_outdir(self):
        d = filedialog.askdirectory(initialdir=self.outdir.get() or os.getcwd(),
                                    parent=self.root)
        if d:
            self.outdir.set(d)
            self.refresh_scan_list()

    # -- hardware -------------------------------------------------------------
    def load_sg(self, quiet=False):
        if self.sg is not None:
            return self.sg
        try:
            self.sg = hw.load_scope_grab(self.cfg["scope_grab_path"])
            if not quiet:
                self.log(f"Scope Grab loaded from {self.cfg['scope_grab_path']}")
        except Exception as exc:
            self.log(f"Scope Grab not loaded ({exc}) - set scope_grab_path in the config")
        return self.sg

    def roles(self):
        return {ch: self.ch_role[ch].get() for ch in (1, 2, 3, 4)
                if self.ch_role[ch].get() != "off"}

    def ensure_sim(self, roles):
        """The simulated scope and mount (worker thread: `roles` is read from
        the window beforehand, never here)."""
        if self.bench is None:
            sg = self.load_sg()
            scope, ell, self.bench = sim.make(sg, roles=roles,
                                              zero_offset_deg=float(self.cfg["ell_zero_deg"]))
            scope.realtime = 0.05
            self._sim_parts = (scope, ell)
        return self._sim_parts

    def do_connect_scope(self):
        c = self.gather()
        roles = self.roles()

        def go():
            sg = self.load_sg()
            if sg is None:
                raise RuntimeError("Scope Grab is not loaded")
            if c["simulate"]:
                scope, _ = self.ensure_sim(roles)
            else:
                prof = sg.scope_profiles.get_profile(c["scope_model"])
                scope = sg.Scope(prof)
                scope.connect(c["scope_addr"] or None)
            self.link = hw.ScopeLink(scope, log=self.log)
            self.log(f"Scope: {scope.idn.strip()} at {scope.addr}")
            return f"scope: {short_idn(scope.idn)}"

        self.worker(go, done=lambda txt: self.scope_status.configure(text=txt, foreground="#060"))

    def do_connect_ell(self):
        c = self.gather()
        roles = self.roles()

        def go():
            if c["simulate"]:
                _, dev = self.ensure_sim(roles)
            else:
                from .ell14 import ELL14
                dev = ELL14(c["ell_port"] or None, address=c["ell_address"],
                            zero_offset_deg=float(c["ell_zero_deg"]))
            self.rot = hw.Rotator(dev, log=self.log)
            info = dev.info()
            pos = dev.position()
            self.log(f"Analyzer: ELL{info['type']} S/N {info['serial']} on {dev.port}, "
                     f"{info['pulses_per_unit']} pulses/rev, firmware {info['firmware']}")
            return (f"analyzer: ELL{info['type']} S/N {info['serial']} on {dev.port}", pos)

        def done(out):
            self.ell_status.configure(text=out[0], foreground="#060")
            self.show_pos(out[1])
        self.worker(go, done=done)

    def do_disconnect(self):
        def go():
            if self.link is not None and self.bench is None:
                self.link.scope.close()
            if self.rot is not None and self.bench is None:
                self.rot.close()
            self.link = self.rot = None
            self.bench = None

        def done(_):
            self.scope_status.configure(text="scope: not connected", foreground="#666")
            self.ell_status.configure(text="analyzer: not connected", foreground="#666")
        self.worker(go, done=done)

    def need(self, scope=True, ell=True):
        if scope and self.link is None:
            self.log("Connect the scope first.")
            return False
        if ell and self.rot is None:
            self.log("Connect the ELL14 first.")
            return False
        return True

    def show_pos(self, pos):
        self.pos_label.configure(text=f"position: {pos:8.3f} deg")

    def do_read_pos(self):
        if self.need(scope=False):
            self.worker(self.rot.position, done=self.show_pos)

    def do_home(self):
        if self.need(scope=False):
            self.worker(lambda: (self.rot.home(), self.rot.position())[1], done=self.show_pos)

    def do_goto(self):
        if not self.need(scope=False):
            return
        try:
            a = float(self.goto_var.get())
        except ValueError:
            self.log("Go to: not a number")
            return
        back = float(self.cfg["scan"].get("backoff_deg", 3.0))
        self.worker(lambda: self.rot.approach(a, backoff=back), done=self.show_pos)

    def do_jog(self, d):
        if self.need(scope=False):
            self.worker(lambda: (self.rot.dev.move_by(d), self.rot.position())[1],
                        done=self.show_pos)

    def do_apply_zero(self):
        try:
            z = float(self.zero_var.get()) % 360.0
        except ValueError:
            self.log("Zero: not a number")
            return
        self.cfg["ell_zero_deg"] = z
        if self.rot is not None:
            self.rot.zero = z
        self.log(f"Analyzer zero is now mount {z:.3f} deg.")
        self.save_settings()

    def do_zero_from_rest(self):
        """Make analyzer 0 the rest (EO-zero) polarization of the scan on show:
        new zero = the scan's zero + its fitted rest azimuth."""
        if not self.result:
            self.log("Load a scan first - the zero comes from its rest azimuth.")
            return
        z0 = float(self.result["d"].manifest.get("rotator", {}).get("zero_deg", 0.0))
        psi = self.result["pol"]["psi_rest"]
        z = (z0 + psi) % 360.0
        if not messagebox.askyesno(
                "Set analyzer zero",
                f"The scan's rest polarization sits at {psi:+.3f} deg in its "
                f"analyzer frame (zero = mount {z0:.3f}).\n\nSet zero = mount "
                f"{z:.3f} deg, so analyzer 0 is aligned with it?", parent=self.root):
            return
        self.zero_var.set(f"{z:.3f}")
        self.do_apply_zero()

    def do_apply_preset(self):
        if not self.need(ell=False):
            return
        p = cfgmod.all_presets(self.cfg).get(self.preset.get())
        if not p or not p.get("scope"):
            return
        self.worker(lambda: self.link.apply(p["scope"]),
                    done=lambda _: self.log(f"Applied preset '{self.preset.get()}' to the scope."))

    # -- scope settings window ---------------------------------------------------
    def open_scope_settings(self):
        """A window laid out from the scope profile's own settings tables:
        timebase and acquisition, trigger, and every channel. Read fills it
        from the scope, Apply writes only what was changed, Save as preset
        keeps the lot (and the scan's shot settings) under a name."""
        if getattr(self, "set_win", None) is not None and self.set_win.winfo_exists():
            self.set_win.lift()
            return
        sg = self.load_sg()
        if sg is None:
            return
        prof = (self.link.prof if self.link is not None
                else sg.scope_profiles.get_profile(self.cfg["scope_model"]))
        self.set_prof = prof
        w = self.set_win = tk.Toplevel(self.root)
        w.title(f"Scope settings - {prof.name}")
        self.set_vars, self.set_read, self.set_kind = {}, {}, {}

        def field(parent, scpi, kind, choices, row, col, width=10):
            var = tk.StringVar()
            if kind in ("num", "info"):
                wdg = ttk.Entry(parent, textvariable=var, width=width,
                                state="readonly" if kind == "info" else "normal")
            elif kind == "bool":
                wdg = ttk.Combobox(parent, textvariable=var, values=("ON", "OFF"),
                                   width=5, state="readonly")
            else:
                wdg = ttk.Combobox(parent, textvariable=var, values=list(choices),
                                   width=max(6, width - 2), state="readonly")
            wdg.grid(row=row, column=col, sticky="w", padx=2, pady=1)
            var.trace_add("write", lambda *_: self.show_span())
            self.set_vars[scpi], self.set_kind[scpi] = var, kind

        top = ttk.Frame(w)
        top.pack(fill="x", padx=8, pady=6)
        tb = ttk.LabelFrame(top, text="Timebase / acquisition")
        tb.pack(side="left", fill="y")
        self.span_label = None
        rows = list(prof.timebase) + [(lbl, scpi, "info", None) for lbl, scpi in prof.info]
        for i, (lbl, scpi, kind, ch) in enumerate(rows):
            ttk.Label(tb, text=lbl + ":").grid(row=i, column=0, sticky="e", padx=4)
            field(tb, scpi, kind, ch, i, 1, 12)
        self.span_label = ttk.Label(tb, text="", foreground="#060")
        self.span_label.grid(row=len(rows), column=0, columnspan=2, sticky="w",
                             padx=4, pady=(4, 2))
        tg = ttk.LabelFrame(top, text="Trigger")
        tg.pack(side="left", fill="y", padx=(8, 0))
        for i, (lbl, scpi, kind, ch) in enumerate(prof.trigger):
            ttk.Label(tg, text=lbl + ":").grid(row=i, column=0, sticky="e", padx=4)
            field(tg, scpi, kind, ch, i, 1, 10)
        cf = ttk.LabelFrame(w, text="Channels")
        cf.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(cf, text="role").grid(row=0, column=1)
        for j, item in enumerate(prof.channel):
            ttk.Label(cf, text=item[0]).grid(row=0, column=j + 2)
        for i, chn in enumerate(prof.channels):
            ttk.Label(cf, text=f"CH{chn}").grid(row=i + 1, column=0, padx=4)
            role = self.ch_role[chn].get() if chn in self.ch_role else ""
            ttk.Label(cf, text=role, foreground="#666").grid(row=i + 1, column=1, padx=4)
            for j, (lbl, tmpl, kind, ch) in enumerate(prof.channel):
                field(cf, tmpl.format(ch=chn), kind, ch, i + 1, j + 2, 8)
        bar = ttk.Frame(w)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bar, text="Read from scope", command=self.do_read_scope).pack(side="left")
        ttk.Button(bar, text="Apply changes", command=self.do_apply_scope).pack(side="left", padx=6)
        ttk.Button(bar, text="Save as preset...", command=self.do_save_preset).pack(side="left")
        self.set_status = ttk.Label(bar, text="not read yet", foreground="#666")
        self.set_status.pack(side="left", padx=8)
        if self.link is not None:
            self.do_read_scope()

    def _set_norm(self, scpi, raw):
        """A scope reply as the field shows it: bools as ON/OFF; anything else
        as the scope said it (choices come back short: NORM, HRES, ...)."""
        raw = str(raw).strip()
        if self.set_kind.get(scpi) == "bool":
            return "ON" if raw.upper() in ("1", "ON") else "OFF"
        return raw

    def show_span(self):
        if getattr(self, "span_label", None) is None:
            return
        g = {k: v.get() for k, v in self.set_vars.items()}
        try:
            t0, t1 = cfgmod.record_span(g[":TIMebase:SCALe"], g[":TIMebase:POSition"],
                                        g.get(":TIMebase:REFerence", "LEFT"))
            self.span_label.configure(
                text=f"record: {t0 * 1e3:+.3f} to {t1 * 1e3:+.3f} ms from the trigger")
        except (KeyError, ValueError):
            self.span_label.configure(text="")

    def do_read_scope(self):
        if not self.need(ell=False):
            return
        roots = list(self.set_vars)

        def go():
            out = {}
            for r in roots:
                try:
                    out[r] = self.link.scope.get(r)
                except Exception as exc:
                    self.log(f"  {r}? failed: {exc}")
            return out

        def done(vals):
            for r, v in vals.items():
                v = self._set_norm(r, v)
                self.set_vars[r].set(v)
                self.set_read[r] = v
            self.set_status.configure(text=f"read {time.strftime('%H:%M:%S')}",
                                      foreground="#060")
        self.worker(go, done=done)

    def do_apply_scope(self):
        if not self.need(ell=False):
            return
        changes = {r: v.get().strip() for r, v in self.set_vars.items()
                   if self.set_kind[r] != "info" and v.get().strip()
                   and v.get().strip() != self.set_read.get(r)}
        if not changes:
            self.log("Scope settings: nothing changed.")
            return
        self.log("Scope settings: writing " + ", ".join(f"{k} {v}" for k, v in changes.items()))
        self.worker(lambda: self.link.apply(changes), done=lambda _: self.do_read_scope())

    def do_save_preset(self, name=None):
        if name is None:
            from tkinter import simpledialog
            name = simpledialog.askstring("Save preset", "Preset name:", parent=self.set_win)
        if not name:
            return
        scope_vals = {r: v.get().strip() for r, v in self.set_vars.items()
                      if self.set_kind[r] != "info" and v.get().strip()}
        s = self.gather()["scan"]
        self.cfg.setdefault("user_presets", {})[name] = {
            "note": f"saved {time.strftime('%Y-%m-%d %H:%M')} from the Scope settings window",
            "scope": scope_vals,
            "scan": {k: s[k] for k in ("mode", "shots", "points", "wait_s", "rep_s") if k in s},
        }
        self.save_settings()
        self.preset_box["values"] = list(cfgmod.all_presets(self.cfg))
        self.preset.set(name)
        self.log(f"Preset '{name}' saved: {len(scope_vals)} scope settings + shot settings.")

    def do_stop(self):
        self.stop_flag.set()
        self.log("Stopping after the current block...")

    # -- scanning ---------------------------------------------------------------
    def _progress(self, done, total, text):
        def ui():
            self.progress_bar["maximum"] = max(total, 1)
            self.progress_bar["value"] = done
            self.progress_text.configure(text=text)
        self.call(ui)

    def _new_run(self, name):
        c = self.cfg
        chans = cfgmod.channel_roles(c)
        clock = (lambda: self.bench.clock) if self.bench is not None else time.time
        return scanmod.ScanRun(c["outdir"], name, self.sg, self.link, self.rot,
                               chans, log=self.log, progress=self._progress,
                               cancelled=self.stop_flag.is_set, clock=clock)

    def do_start_scan(self):
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        if "PD" not in [r for r, _ in cfgmod.channel_roles(c).values()]:
            self.log("No channel has the PD role.")
            return
        run = self._new_run(c["scan_name"])
        if run.exists():
            ans = messagebox.askyesnocancel(
                "Scan exists", f"{run.name} already exists in this folder.\n\n"
                "Yes = resume it (measure the steps not done)\nNo = pick a new "
                "name\nCancel = do nothing", parent=self.root)
            if not ans:
                return
            run.load()
            self.run = run
            self.worker(run.run, done=self.scan_done)
            return
        s = c["scan"]
        angles = scanmod.ordered(scanmod.angle_list(s["start"], s["stop"], s["step"]),
                                 s["order"])
        steps = scanmod.build_steps(angles, int(s["ref_every"]), s["ref_angle"])
        if self.with_dark.get():
            steps = [{"kind": "dark", "target": 0.0}] + steps
        plan = dict(s, preset=c["preset"], software=f"rampol {__version__}")
        run.new(plan, steps, extra={"zero_deg": float(c["ell_zero_deg"])})
        self.run = run
        self.log(f"Scan {run.name}: {len(angles)} angles, {len(steps)} steps -> {run.folder}")
        if self.with_dark.get():
            self.dark_then(run, lambda: self.worker(run.run, done=self.scan_done))
        else:
            self.worker(run.run, done=self.scan_done)

    def dark_then(self, run, after, vdiv_note=""):
        """Ask for the beam to be blocked, take the dark steps, ask for it to be
        unblocked, then call after()."""
        if not messagebox.askokcancel(
                "Dark capture", f"Block the beam before the analyzer{vdiv_note}, "
                "then press OK.", parent=self.root):
            self.log("Dark skipped - the analysis will use 0 V for the dark level.")
            for s in run.manifest["steps"]:
                if s["kind"] == "dark" and s["status"] != "done":
                    s["status"] = "skipped"
            run.save()
            after()
            return
        if self.bench is not None:
            self.bench._imax_saved, self.bench.imax = self.bench.imax, 0.0

        def unblock(_):
            if self.bench is not None:
                self.bench.imax = self.bench._imax_saved
            messagebox.showinfo("Dark capture", "Dark done. Unblock the beam, then press OK.",
                                parent=self.root)
            after()
        self.worker(lambda: run.run(kinds={"dark"}), done=unblock)

    def scan_done(self, n):
        self.log(f"Scan {self.run.name}: {n} steps measured.")
        self.refresh_scan_list(select=self.run.folder)
        self.do_load_shown()

    def do_plan_refine(self):
        plans = self.refine_plans()
        if plans is None:
            return
        for p in plans:
            self.log(f"  window {p['window'] + 1} ({p['label']}): {p['t0'] * 1e3:.2f}-"
                     f"{p['t1'] * 1e3:.2f} ms, null at {p['null']:.2f} deg, "
                     f"{len(p['angles'])} angles")

    def refine_plans(self):
        if not self.result:
            self.log("Load the scan to refine first.")
            return None
        c = self.gather()
        pol = self.result["pol"]
        txt = c["refine"]["windows"]
        if str(txt).strip().lower() in ("", "auto"):
            wins = an.auto_windows(pol)
        else:
            wins = [(a, b, f"{a * 1e3:.2f}-{b * 1e3:.2f} ms") for a, b in an.parse_windows(txt)]
        try:
            offs = [float(x) for x in str(c["refine"]["offsets"]).split(",") if x.strip()]
        except ValueError:
            self.log("Offsets: comma-separated degrees")
            return None
        return an.null_targets(pol, wins, offs)

    def do_run_refine(self):
        if not self.need():
            return
        plans = self.refine_plans()
        if not plans:
            return
        c = self.cfg
        d = self.result["d"]
        pd_ch = d.roles.get("PD")
        vdiv = float(c["refine"]["pd_vdiv"])
        # 0 V one division above the bottom of the screen
        scale = {"ch": pd_ch, "vdiv": vdiv, "offset": 3.0 * vdiv}
        run = self._new_run(d.name)
        run.load()
        n_dark = sum(1 for s in run.manifest["steps"] if s["kind"] == "dark"
                     and s.get("pd_scale", {}).get("vdiv") == vdiv)
        steps = []
        if not n_dark:
            steps.append({"kind": "dark", "target": 0.0, "pd_scale": scale})
        rs = c["refine"]
        for p in plans:
            for a in p["angles"]:
                steps.append({"kind": "null", "target": float(a), "window": p["window"],
                              "t0": p["t0"], "t1": p["t1"], "label": p["label"],
                              "pd_scale": scale, "shots": int(rs.get("shots", 64)),
                              "blocks": int(rs.get("blocks", 4))})
        run.add_steps(steps)
        self.run = run
        self.log(f"Refine: {len(plans)} windows, {len(steps)} steps at PD {vdiv:g} V/div")
        go = lambda: self.worker(lambda: run.run(kinds={"null"}), done=self.scan_done)
        if not n_dark:
            self.dark_then(run, go, vdiv_note=f" (dark at {vdiv:g} V/div)")
        else:
            go()

    # -- loading and analysing ---------------------------------------------------
    def refresh_scan_list(self, select=None):
        out = self.outdir.get().strip() or self.cfg["outdir"]
        names = []
        try:
            for n in sorted(os.listdir(out)):
                if os.path.isfile(os.path.join(out, n, f"{n}_scan.json")):
                    names.append(n)
        except OSError:
            pass
        self.scan_box["values"] = names
        self.cmp_box["values"] = names
        if select:
            self.show_scan.set(os.path.basename(select))

    def _scan_path(self, text):
        text = text.strip()
        if not text:
            return None
        if os.path.isabs(text):
            return text
        return os.path.join(self.outdir.get().strip() or self.cfg["outdir"], text)

    def do_open_scan(self):
        p = filedialog.askopenfilename(
            title="Open a scan manifest", parent=self.root,
            initialdir=self.outdir.get(), filetypes=[("Scan manifest", "*_scan.json")])
        if p:
            self.show_scan.set(os.path.dirname(p))
            self.do_load_shown()

    def analyse(self, path, correct_drift=True):
        """Worker thread: everything it needs from the window is passed in."""
        sg = self.load_sg()
        if sg is None:
            raise RuntimeError("Scope Grab is not loaded, so captures cannot be read")
        a = self.cfg["analysis"]
        d = an.load_scan(path, sg.load_capture, trim=int(a["trim"]),
                         lock_tol=float(a.get("lock_tol", 0.0)))
        pol = an.polarization(d, correct_drift=correct_drift)
        dips = an.dip_er(pol, polarizer_er=a["polarizer_er"])
        refine = an.refine_result(d, pol, polarizer_er=a["polarizer_er"])
        mon = an.monitor_prediction(d, pol, a["deg_per_mon_v"])
        return {"d": d, "pol": pol, "dips": dips, "refine": refine, "mon": mon,
                "path": path}

    def do_load_shown(self):
        path = self._scan_path(self.show_scan.get())
        if not path:
            return
        if self.busy and self.run is None:
            return
        drift = bool(self.drift_on.get())

        def go():
            return self.analyse(path, drift)

        def done(res):
            self.result = res
            pol, d = res["pol"], res["d"]
            for n in d.notes:
                self.log(f"  {n}")
            self.log(f"Loaded {d.name}: {pol['n_angles']} angles, {len(d.t)} samples, "
                     f"rest azimuth {pol['psi_rest']:+.3f} deg, {len(res['dips'])} "
                     f"dip ER points, {len(res['refine'])} refined windows")
            if pol.get("mod_snr", 99) < 10:
                self.log(f"  ! the light is not modulated by the analyzer (median "
                         f"B/sigma {pol['mod_snr']:.1f}): no light, or the analyzer "
                         f"is not in the beam - angles and ER from this scan are noise")
            dr = pol.get("drift_resid")
            if dr is not None and pol.get("mod_snr", 99) >= 10:
                self.log(f"  ref returns predict each other to {dr * 1e3:.2f}e-3 (leave-one-out "
                         f"rms): ER_fit is drift-limited above ~{1 / max(dr, 1e-9):.0f}; "
                         f"the dip and refine points are not")
            off = [s for s in d.steps if any(v.any() for v in s.get("offscreen", {}).values())
                   and s["kind"] in ("scan", "ref")]
            if off:
                self.log(f"  ! {len(off)} steps have samples off screen at their V/div "
                         f"- possibly clipped; see Diagnostics")
            self.mark_dirty()
        if self.busy:
            # loading during a scan: do it here (reading only)
            try:
                done(go())
            except Exception as exc:
                self.log(f"ERROR loading: {exc}")
        else:
            self.worker(go, done=done)

    def reanalyse(self):
        if self.result:
            self.do_load_shown()

    def do_load_compare(self):
        path = self._scan_path(self.cmp_scan.get())
        if not path:
            return

        def done(res):
            self.compare = res
            self.mark_dirty()
        drift = bool(self.drift_on.get())
        self.worker(lambda: self.analyse(path, drift), done=done)

    def do_clear_compare(self):
        self.compare = None
        self.cmp_scan.set("")
        self.mark_dirty()

    # -- drawing -------------------------------------------------------------------
    def mark_dirty(self):
        self.plot_dirty = set(self.plot_tabs)
        self.draw_visible()

    def draw_visible(self):
        try:
            frame = self.root.nametowidget(self.nb.select())
        except (KeyError, tk.TclError):
            return
        if frame in self.plot_dirty:
            self.plot_dirty.discard(frame)
            fig, draw = self.plot_tabs[frame]
            if fig is None:
                draw()
            else:
                self.redraw(fig, draw)

    def redraw(self, fig, draw=None):
        draw = draw or self.plot_tabs[fig._frame][1]
        fig.clear()
        if not self.result:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No scan loaded", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_axis_off()
        else:
            try:
                draw(fig)
            except Exception as exc:
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(0.02, 0.5, f"Could not draw: {exc}", transform=ax.transAxes)
                ax.set_axis_off()
                self.log(f"draw: {exc}")
        fig._canvas.draw_idle()

    def on_click(self, ev, fig):
        if ev.inaxes is None or ev.xdata is None or fig._toolbar.mode:
            return
        if fig is self.fig_ext and self.ext_x.get() != "time":
            return
        self.cursor_t = ev.xdata * 1e-3
        self.cursor_var.set(f"{ev.xdata:.3f}")
        self.plot_dirty |= {self.fig_malus._frame, self.fig_map._frame,
                            self.fig_angle._frame, self.fig_ext._frame}
        self.draw_visible()

    def set_cursor_text(self):
        try:
            self.cursor_t = float(self.cursor_var.get()) * 1e-3
        except ValueError:
            return
        self.mark_dirty()

    def smooth_samples(self, t):
        try:
            us = max(float(self.smooth_us.get()), 0.0)
        except ValueError:
            us = 0.0
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        return us, max(1, int(round(us * 1e-6 / dt)))

    def smooth(self, y, t):
        """Running mean over the Smooth box's span (edges padded)."""
        _, n = self.smooth_samples(t)
        if n <= 1:
            return y
        yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
        return np.convolve(yp, np.ones(n) / n, mode="same")[n:-n]

    def smoothed_er(self, pol, t):
        """ER from running means of Imax and Imin (not a mean of ratios),
        with its lower-bound flag recomputed for the averaged Imin."""
        us, n = self.smooth_samples(t)
        if n <= 1:
            return pol["er"], ~pol["er_lower"], 0
        imax, imin = self.smooth(pol["imax"], t), self.smooth(pol["imin"], t)
        sig = np.sqrt(self.smooth(pol["sig_imin"] ** 2, t) / n)
        ok = imin >= 2 * sig
        er = np.where(ok, imax / np.where(ok, imin, 1), imax / (2 * sig))
        return er, ok, us

    def _cursor(self, ax):
        if self.cursor_t is not None:
            ax.axvline(self.cursor_t * 1e3, color="#d62728", lw=0.8, ls=":")

    def draw_traces(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t * 1e3
        has_mon = any(r in d.roles for r in ("MonX1", "MonX2"))
        ax = fig.add_subplot(211 if has_mon else 111)
        cmap = matplotlib.colormaps[ANGLE_CMAP]
        for th, y in zip(pol["theta"], pol["I"]):
            ax.plot(t, y, lw=0.6, color=cmap((th % 360) / 360))
        for s in d.steps:
            if s["kind"] == "dark" and "PD" in s["v"]:
                ax.plot(t, s["v"]["PD"], lw=0.6, color="k", ls="--", label="dark")
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 360))
        ax.set_ylabel("PD - dark, drift corrected (V)")
        ax.set_title(f"Analyzer photodiode at {pol['n_angles']} analyzer angles ({d.name})")
        ax.grid(alpha=0.3)
        if has_mon:
            ax.tick_params(labelbottom=False)
            ax2 = fig.add_subplot(212, sharex=ax)
            for r, col in (("MonX1", "#1f77b4"), ("MonX2", "#2ca02c")):
                if r in d.roles:
                    v = np.mean([s["v"][r] for s in pol["steps"]], axis=0)
                    ax2.plot(t, v, lw=0.8, color=col, label=f"{r} (mean of scan steps)")
            ax2.set_ylabel("monitor (V)")
            ax2.legend(loc="upper right", fontsize=7)
            ax2.grid(alpha=0.3)
            ax2.set_xlabel("time (ms)")
            fig.colorbar(sm, ax=[ax, ax2], label="analyzer angle (deg)")
        else:
            ax.set_xlabel("time (ms)")
            fig.colorbar(sm, ax=ax, label="analyzer angle (deg)")

    def draw_map(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t * 1e3
        order = np.argsort(pol["theta"] % 360)
        th = pol["theta"][order] % 360
        I = pol["I"][order] / np.maximum(pol["imax"], 1e-9)
        ax = fig.add_subplot(111)
        edges = np.concatenate([[th[0] - (th[1] - th[0]) / 2],
                                (th[1:] + th[:-1]) / 2,
                                [th[-1] + (th[-1] - th[-2]) / 2]]) if len(th) > 1 else [th[0] - 1, th[0] + 1]
        tm = np.concatenate([[t[0]], (t[1:] + t[:-1]) / 2, [t[-1]]])
        mesh = ax.pcolormesh(tm, edges, I, cmap="magma", shading="flat", rasterized=True,
                             vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax, label="I / Imax(t)")
        for k in range(-1, 4):
            ax.plot(t, pol["psi_u"] + 90 + 180 * k, color="#4fc3f7", lw=0.7)
        ax.set_ylim(edges[0], edges[-1])
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("analyzer angle (deg)")
        ax.set_title(f"Transmission vs time and analyzer angle; line: fitted null "
                     f"psi + 90 deg ({d.name})")
        self._cursor(ax)

    def draw_malus(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t
        j = int(np.argmin(np.abs(t - (self.cursor_t if self.cursor_t is not None else t[len(t) // 2]))))
        th = pol["theta"]
        y = pol["I"][:, j]
        ax = fig.add_subplot(211)
        ax.plot(th % 360, y * 1e3, "o", ms=4, color="#1f77b4", label="measured")
        g = np.linspace(0, 360, 721)
        gr = np.deg2rad(g)
        model = pol["a0"][j] + pol["c2"][j] * np.cos(2 * gr) + pol["s2"][j] * np.sin(2 * gr)
        full = model.copy()
        for k, f in (("c1", np.cos(gr)), ("s1", np.sin(gr)), ("c4", np.cos(4 * gr)), ("s4", np.sin(4 * gr))):
            if k in pol:
                full = full + pol[k][j] * f
        ax.plot(g, model * 1e3, color="k", lw=0.9, label="a0 + 2-theta terms")
        if "c1" in pol:
            ax.plot(g, full * 1e3, color="#ff7f0e", lw=0.7, ls="--", label="with 1- and 4-theta terms")
        ax.set_ylabel("PD - dark (mV)")
        ax.set_title(f"Transmission vs analyzer angle at t = {t[j] * 1e3:.3f} ms ({d.name})")
        ax.text(0.01, 0.97,
                f"psi = {pol['psi'][j]:+.3f} +- {pol['sig_psi'][j] * 1e3:.1f} mdeg\n"
                f"Imax = {pol['imax'][j] * 1e3:.1f} mV, Imin = {pol['imin'][j] * 1e3:.2f} "
                f"+- {pol['sig_imin'][j] * 1e3:.2f} mV\nER_fit = "
                f"{'>' if pol['er_lower'][j] else ''}{pol['er'][j]:.0f}, visibility "
                f"{pol['vis'][j]:.5f}", transform=ax.transAxes, va="top", fontsize=7,
                family="monospace")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        ax2 = fig.add_subplot(212, sharex=ax)
        thr = np.deg2rad(th)
        fit_at = pol["a0"][j] + pol["c2"][j] * np.cos(2 * thr) + pol["s2"][j] * np.sin(2 * thr)
        ax2.plot(th % 360, (y - fit_at) * 1e3, "o", ms=3, color="#1f77b4")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set_xlabel("analyzer angle (deg)")
        ax2.set_ylabel("residual to 2-theta fit (mV)")
        ax2.grid(alpha=0.3)

    def draw_angle(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t * 1e3
        ax = fig.add_subplot(211)
        rot, sig = pol["rotation"], pol["sig_psi"]
        ax.fill_between(t, rot - sig, rot + sig, color="#1f77b4", alpha=0.3, lw=0)
        ax.plot(t, rot, color="#1f77b4", lw=0.9, label=f"measured ({d.name})")
        if res["mon"] is not None:
            ax.plot(t, res["mon"][0], color="#2ca02c", lw=0.8, ls="--",
                    label="from Trek monitors (offset matched)")
        if self.compare:
            cp = self.compare["pol"]
            ax.plot(cp["t"] * 1e3, cp["rotation"], color=CMP_COLOUR, lw=0.8,
                    label=f"compare ({self.compare['d'].name})")
        ax.set_ylabel("rotation from rest (deg)")
        ax.set_title(f"Polarization rotation vs time (rest azimuth {pol['psi_rest']:+.3f} deg "
                     f"in the analyzer frame)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        self._cursor(ax)
        ax2 = fig.add_subplot(212, sharex=ax)
        if res["mon"] is not None:
            ax2.plot(t, res["mon"][1] * 1e3, color="#2ca02c", lw=0.7, label="measured - monitor prediction")
        if self.compare and len(self.compare["pol"]["t"]) == len(t):
            ax2.plot(t, (rot - self.compare["pol"]["rotation"]) * 1e3, color=CMP_COLOUR, lw=0.7,
                     label="this - compare")
        ax2.plot(t, sig * 1e3, color="k", lw=0.6, ls="--", label="+-1 SD of the fit")
        ax2.plot(t, -sig * 1e3, color="k", lw=0.6, ls="--")
        ax2.set_ylabel("difference (mdeg)")
        ax2.set_xlabel("time (ms)")
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(alpha=0.3)
        self._cursor(ax2)

    def draw_extinction(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        by_rot = self.ext_x.get() == "rotation"
        x = pol["rotation"] if by_rot else d.t * 1e3
        ax = fig.add_subplot(111)
        er, ok, us = self.smoothed_er(pol, d.t)
        lbl = f"ER_fit ({us:g} us mean)" if us else "ER_fit (per sample)"
        ax.plot(x, np.where(ok, er, np.nan), color="#1f77b4", lw=0.7, label=lbl)
        ax.plot(x, np.where(~ok, er, np.nan), color="#1f77b4", lw=0.5, alpha=0.35,
                label="ER_fit lower bound (Imin < 2 sigma)")
        dips = res["dips"]
        if dips:
            for sign, mk, lbl in ((1, "^", "dip, rising"), (-1, "v", "dip, falling")):
                pts = [p for p in dips if np.sign(p["rate"]) == sign]
                if not pts:
                    continue
                xx = [p["rotation"] if by_rot else p["t"] * 1e3 for p in pts]
                ax.plot(xx, [p["er"] for p in pts], mk, ms=5,
                        color="#d62728", ls="none", label=f"{lbl} ({len(pts)})")
        for r in res["refine"]:
            if "er" in r:
                m = (d.t >= r["t0"]) & (d.t <= r["t1"])
                xx = float(np.mean(pol["rotation"][m])) if by_rot else 0.5 * (r["t0"] + r["t1"]) * 1e3
                ax.plot([xx], [r["er"]], "D", ms=6, color="#9467bd",
                        label=f"null refine: {r['label']}")
        if self.compare:
            cp = self.compare
            xx = [p["rotation"] if by_rot else p["t"] * 1e3 for p in cp["dips"]]
            ax.plot(xx, [p["er"] for p in cp["dips"]], "o", ms=3, color=CMP_COLOUR,
                    ls="none", label=f"compare dips ({cp['d'].name})")
        dr = pol.get("drift_resid")
        if dr:
            ax.axhline(1 / dr, color="#1f77b4", lw=0.8, ls=":",
                       label=f"1 / ref leave-one-out scatter ({1 / dr:.0f})")
        lim = self.cfg["analysis"]["polarizer_er"]
        ax.axhline(lim, color="k", lw=0.8, ls="--",
                   label=f"analyzer ER spec floor ({lim:.0e})")
        ax.set_yscale("log")
        ax.set_xlabel("rotation from rest (deg)" if by_rot else "time (ms)")
        ax.set_ylabel("extinction ratio Imax / Imin")
        ax.set_title(f"Extinction ratio along the ramp ({d.name})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        ax.grid(alpha=0.3, which="both")
        if not by_rot:
            self._cursor(ax)

    def draw_diagnostics(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t * 1e3
        ax = fig.add_subplot(221)
        if len(pol["ref_clocks"]):
            c0 = pol["ref_clocks"].min()
            lv = pol["ref_levels"]
            ax.plot((pol["ref_clocks"] - c0) / 60, (lv / lv.mean() - 1) * 1e3, "o-", ms=3)
        ax.set_xlabel("time into scan (min)")
        ax.set_ylabel("ref level - mean (1e-3)")
        ax.set_title("Reference-angle returns")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(222)
        if "c1" in pol:
            B = np.maximum(pol["B"], 1e-9)
            sm = lambda y: self.smooth(y, d.t)
            # smooth the components, then take the amplitude: noise alone
            # averages toward zero instead of toward its rms
            ax.plot(t, np.hypot(sm(pol["c1"]), sm(pol["s1"])) / B * 1e3, lw=0.6, label="1-theta / B")
            ax.plot(t, np.hypot(sm(pol["c4"]), sm(pol["s4"])) / B * 1e3, lw=0.6, label="4-theta / B")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "needs >= 9 angles over >= 300 deg", ha="center",
                    transform=ax.transAxes, color="#888")
        ax.set_ylabel("relative amplitude (1e-3)")
        ax.set_xlabel("time (ms)")
        ax.set_title("Harmonics outside the Malus law")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(223)
        sem_med = np.nanmedian([np.nanmedian(s["sem"]["PD"]) for s in pol["steps"]])
        ax.plot(t, pol["rms"] * 1e3, lw=0.6, label="fit residual rms")
        if np.isfinite(sem_med):
            ax.axhline(sem_med * 1e3, color="k", ls="--", lw=0.8, label="median block SEM")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("mV")
        ax.set_title("Residual vs measurement noise")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(224)
        st = [s for s in d.steps if s["kind"] in ("scan", "ref", "null") and "landed" in s]
        err = [((s["landed"] - s["target"] + 180) % 360 - 180) * 1e3 for s in st]
        offs = [100 * max(s.get("offscreen_frac", {}).values() or [0]) for s in st]
        ax.plot(range(len(st)), err, "o", ms=3, label="landed - target (mdeg)")
        ax.set_xlabel("step")
        ax.set_ylabel("mdeg")
        ax2 = ax.twinx()
        ax2.plot(range(len(st)), offs, "x", ms=3, color="#d62728")
        ax2.set_ylabel("samples off screen (%)", color="#d62728")
        ax2.set_ylim(bottom=0, top=max(1.0, max(offs or [0]) * 1.1))
        ax.set_title("Analyzer landing and off-screen samples per step")
        ax.grid(alpha=0.3)

    def fill_table(self):
        self.tv.delete(*self.tv.get_children())
        if not self.result:
            return
        pol = self.result["pol"]

        def f(x, fmt):
            return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)
        for s in an.segments(pol["t"], pol["rotation"]):
            m = (pol["t"] >= s["t0"]) & (pol["t"] <= s["t1"])
            self.tv.insert("", "end", values=(
                f"segment: {s['kind']}", f"{s['t0'] * 1e3:.2f} - {s['t1'] * 1e3:.2f}",
                f(float(np.median(pol["rotation"][m])), ".3f"), "", "",
                f(float(np.median(pol["er"][m])), ".0f"), "", f(float(np.median(pol["imin"][m])) * 1e3, ".2f"),
                f(float(np.median(pol["sig_imin"][m])) * 1e3, ".2f"),
                f(float(np.median(pol["vis"][m])), ".5f"),
                f(float(np.median(pol["sig_psi"][m])) * 1e3, ".1f"), "ER_fit median"))
        for r in self.result["refine"]:
            if "er" not in r:
                continue
            self.tv.insert("", "end", values=(
                f"refine: {r['label']}", f"{r['t0'] * 1e3:.2f} - {r['t1'] * 1e3:.2f}", "",
                f(r["theta_null"], ".2f"), "", ("> " if r["er_lower"] else "") + f(r["er"], ".0f"),
                f(r["er_light"], ".0f"), f(r["imin"] * 1e3, ".3f"), f(r["sig_imin"] * 1e3, ".3f"),
                "", "", ("dark at matched V/div" if r["dark_vdiv_matched"] else "no matched dark")
                + ("; samples off screen" if r["offscreen"] else "")))
        for p in self.result["dips"]:
            self.tv.insert("", "end", values=(
                "dip", f(p["t"] * 1e3, ".3f"), f(p["rotation"], ".2f"), f(p["theta"] % 360, ".2f"),
                f(p["rate"], ".4f"), ("> " if p["er_lower"] else "") + f(p["er"], ".0f"),
                f(p["er_light"], ".0f"), f(p["imin"] * 1e3, ".3f"), f(p["sig_imin"] * 1e3, ".3f"),
                "", "", f"{p['n']} samples"))

    def save_table(self):
        if not self.result:
            return
        p = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".csv",
            initialdir=self.result["d"].folder,
            initialfile=f"{self.result['d'].name}_table.csv")
        if not p:
            return
        import csv
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(self.table_heads)
            for iid in self.tv.get_children():
                w.writerow(self.tv.item(iid, "values"))
        self.log(f"Table saved: {p}")

    # -- closing ----------------------------------------------------------------
    def on_close(self):
        if self.busy and not messagebox.askyesno(
                "Quit", "An operation is running. Stop it and quit?", parent=self.root):
            return
        self.stop_flag.set()
        self.save_settings()
        try:
            if self.bench is None:
                if self.rot is not None:
                    self.rot.close()
                if self.link is not None:
                    self.link.scope.close()
        except Exception:
            pass
        self.root.destroy()


def short_idn(idn):
    """'AGILENT TECHNOLOGIES,MSO-X 2014A,MY63080029,02.65...' -> 'MSO-X 2014A
    S/N MY63080029'. The full reply goes to the log."""
    parts = [p.strip() for p in str(idn).split(",")]
    if len(parts) >= 3:
        return f"{parts[1]}  S/N {parts[2]}"
    return str(idn).strip()[:40]


def _isnum(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()

"""The Ramp Polarimeter window.

Controls on the left (hardware, analyzer, channel roles, then the measurement
modes as tabs: ramp scan, analyzer (find angle, null refine), bias points, ILC target),
plots on the right in Scope Grab's arrangement: a plot bar above a notebook of
figure tabs, each with its matplotlib toolbar, and the log underneath.

Threading: every instrument operation runs on one worker thread at a time.
The worker never touches Tk; it hands results back through self.call(),
which the pump runs on the Tk thread.

    python polarimeter.py            (or python -m rampol, or Run in VS Code)
"""
import json
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
from . import checks, hw, lablog, provenance, scan as scanmod, sim

ANGLE_CMAP = "hsv"                 # cyclic: 0 and 360 deg share a colour, none near white
MAP_MODES = ("transmission", "fit residual (mV)", "residual / standard error")
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
        self.cmp_results = {}       # (folder, options) -> analysis, for the Compare tab
        self.cmp_sel = []
        self.cursor_t = None
        self.target = None          # last analyzer angle asked for (Go to / step / jog)
        self.scan_cache = {}        # folder -> {step key: reduced step}, see an.load_scan
        self.live_thread = None     # background analysis while a scan runs
        self.live_pending = None
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
        # the measurement modes share the column below the hardware as tabs
        # (selected by frame); Stop and progress sit under them, for all
        self.modes = ttk.Notebook(left)
        self.modes.pack(fill="x", padx=8, pady=3)
        self.build_scan(self._mode_tab("Ramp scan"))
        self.build_analyzer_mode(self._mode_tab("Analyzer"))
        self.build_bias(self._mode_tab("Bias points"))
        self.build_ilc(self._mode_tab("ILC target"))
        self.build_runbar(left)
        self.bias_result = None
        self.ilc_summary = None
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
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Step").pack(side="left")
        self.step_var = tk.StringVar(value="0.1")
        ttk.Entry(r, textvariable=self.step_var, width=8).pack(side="left", padx=4)
        ttk.Label(r, text="deg").pack(side="left")
        self._btn(r, "- step", lambda: self.do_step(-1), padx=(6, 0))
        self._btn(r, "+ step", lambda: self.do_step(+1), padx=(2, 0))
        ttk.Label(r, text="smallest 0.0025 (1 pulse)", foreground="#666").pack(
            side="left", padx=(8, 0))
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
                  text="One PD required. Mon = Trek monitor, Cmd = Trek command, "
                       "Ref = pick-off before the analyzer.").pack(anchor="w", padx=6, pady=(0, 4))

    def build_scan(self, left):
        f = ttk.Frame(left)
        f.pack(fill="x", padx=2, pady=3)
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
        self.check_first = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="check first", variable=self.check_first).pack(side="left")
        ttk.Label(rr, text="(a used name counts up)", foreground="#666").pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        self.dark_mode = tk.StringVar(value="none")
        self.bg_mode = tk.StringVar(value="measure")
        for label, var in (("Dark (PD covered)", self.dark_mode),
                           ("Background (beam blocked)", self.bg_mode)):
            ttk.Label(rr, text=label).pack(side="left")
            ttk.Combobox(rr, textvariable=var, width=11, state="readonly",
                         values=("measure", "reuse latest", "none")).pack(side="left",
                                                                         padx=(2, 8))
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(4, 2))
        self._btn(rr, "Check scope", self.do_check_scope)
        self._btn(rr, "Start scan", self.do_start_scan, padx=(4, 0))
        self.est_label = ttk.Label(rr, text="", foreground="#666")
        self.est_label.pack(side="left", padx=6)
        for v in list(self.sv.values()) + [self.order, self.mode]:
            v.trace_add("write", lambda *_: self.update_estimate())

    def build_analyzer_mode(self, f):
        """The two ways the analyzer goes near crossed: find an angle in the
        light as it is and leave the analyzer there, or refine the shown
        scan's static nulls (steps added to that scan)."""
        a = ttk.LabelFrame(f, text="Find the min / max transmission angle (and stay there)")
        a.pack(fill="x", padx=4, pady=(4, 2))
        self.build_find(a)
        b = ttk.LabelFrame(f, text="Refine the shown scan's static nulls (adds steps to it)")
        b.pack(fill="x", padx=4, pady=(2, 4))
        self.build_refine(b)

    def build_refine(self, left):
        f = ttk.Frame(left)
        f.pack(fill="x", padx=2, pady=3)
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
                  text="auto = rest/hold/after. A window just after a bright "
                       "excursion can carry overdrive recovery.").pack(anchor="w", padx=6, pady=(0, 4))

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
        ttk.Button(r, text="Export brief", command=self.do_export_brief).pack(
            side="left", padx=(12, 0))
        ttk.Button(r, text="Lab log", command=self.open_lab_log).pack(side="left", padx=4)
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
        r = ttk.Frame(bar)
        r.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(r, text="Apply:").pack(side="left")
        self.sub_dark = tk.BooleanVar(value=True)
        self.gains_on = tk.BooleanVar(value=True)
        self.lock_on = tk.BooleanVar(value=True)
        for text, var in (("dark / background", self.sub_dark),
                          ("per-angle transmission", self.gains_on),
                          ("drop missed-lock shots", self.lock_on)):
            ttk.Checkbutton(r, text=text, variable=var, command=self.reanalyse).pack(
                side="left", padx=(6, 0))
        self.corr_label = ttk.Label(bar, text="", foreground="#8a4b00", wraplength=900,
                                    justify="left")
        self.corr_label.pack(fill="x", padx=6, pady=(0, 4))

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
        ttk.Label(self.fig_map._ctl, text="show:").pack(side="left")
        self.map_show = tk.StringVar(value=MAP_MODES[0])
        cb = ttk.Combobox(self.fig_map._ctl, textvariable=self.map_show, values=MAP_MODES,
                          width=24, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw(self.fig_map))
        ttk.Label(self.fig_map._ctl, foreground="#666",
                  text="residual: what a0 + B cos 2(theta - psi) leaves, per angle").pack(
            side="left", padx=6)
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
        self.ext_show = {}
        for key, text in (("fit", "ER_fit"), ("dips", "dips (fitted Imax)"),
                          ("direct", "direct (both measured)"), ("refine", "null refine")):
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(ctl, text=text, variable=v,
                            command=lambda: self.redraw(self.fig_ext)).pack(side="left",
                                                                            padx=(6, 0))
            self.ext_show[key] = v
        self.fig_poin = self._fig_tab("Poincaré", self.draw_poincare, click=True)
        ttk.Label(self.fig_poin._ctl, foreground="#666", text=(
            "A linear analyzer measures S1 and S2 only: |S3| is drawn as sqrt(1 - p^2), "
            "which assumes full polarization; the handedness is not measured. Click "
            "the time plot to move the cursor.")).pack(side="left")
        self.fig_diag = self._fig_tab("Diagnostics", self.draw_diagnostics)
        self.build_table_tab()
        self.build_shots_tab()
        self.build_build_tab()
        self.fig_corr = self._fig_tab("Corrections", self.draw_corrections)
        ttk.Button(self.fig_corr._ctl, text="Borrow dark / background from another scan...",
                   command=self.do_borrow_dialog).pack(side="left")
        self.build_compare_tab()
        # these draw without a ramp scan loaded
        self.fig_bias = self._fig_tab("Bias points", self.draw_bias)
        self.fig_ilc = self._fig_tab("ILC target", self.draw_ilc)
        self.fig_find = self._fig_tab("Find angle", self.draw_find)
        self.free_tabs = {self.fig_bias._frame, self.fig_ilc._frame, self.fig_find._frame,
                          self.fig_cmp._frame}
        ttk.Label(self.fig_ilc._ctl, text="figure:").pack(side="left")
        self.ilc_fig = tk.StringVar(value="fig2_rotation_vs_target.png")
        cb = ttk.Combobox(self.fig_ilc._ctl, textvariable=self.ilc_fig, width=28,
                          state="readonly",
                          values=("fig1_time_map.png", "fig2_rotation_vs_target.png",
                                  "fig3_error_structure.png", "fig4_correction.png",
                                  "fig5_line_ripple.png"))
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw(self.fig_ilc))
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

    # -- measurement-mode tabs ---------------------------------------------------
    def _mode_tab(self, name):
        frame = ttk.Frame(self.modes)
        self.modes.add(frame, text=name)
        return frame

    def build_runbar(self, left):
        """Stop and progress, under the mode tabs: every mode uses them."""
        f = ttk.Frame(left)
        f.pack(fill="x", padx=8, pady=(0, 4))
        r = ttk.Frame(f)
        r.pack(fill="x")
        self.stop_btn = ttk.Button(r, text="Stop", command=self.do_stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.progress_bar = ttk.Progressbar(r, mode="determinate", maximum=1)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.progress_text = ttk.Label(f, text="", foreground="#060", width=48,
                                       wraplength=330)
        self.progress_text.pack(anchor="w", pady=(0, 2))

    def build_bias(self, f):
        """Bias points: the AWG holds the EOMs at fixed rotations, the analyzer
        steps around each null at a sensitive V/div (rampol.bias)."""
        self.bv = {}
        for items in ((("Biases (deg)", "biases", 13), ("split on X1", "split", 5)),
                      (("Shots", "shots", 4), ("null +-deg", "null_half_deg", 4),
                       ("null points", "null_points", 3)),
                      (("Hold ms", "hold_ms", 5), ("settle ms", "settle_ms", 4),
                       ("Name", "name", 11))):
            rr = ttk.Frame(f)
            rr.pack(fill="x", padx=6, pady=1)
            for label, key, w in items:
                ttk.Label(rr, text=label).pack(side="left", padx=(0, 2))
                self.bv[key] = tk.StringVar()
                ttk.Entry(rr, textvariable=self.bv[key], width=w).pack(side="left", padx=(0, 6))
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        ttk.Label(rr, text="Order").pack(side="left")
        self.bias_order = tk.StringVar()
        ttk.Combobox(rr, textvariable=self.bias_order, values=("up", "updown"),
                     width=8, state="readonly").pack(side="left", padx=4)
        self._btn(rr, "Start bias points", self.do_start_bias, padx=(8, 0))
        self._btn(rr, "Load...", self.do_load_bias, padx=(4, 0))
        ttk.Label(f, foreground="#666", justify="left", wraplength=330, text=(
            "Biases: start:stop:step or a list, in target rotation degrees. The "
            "4063B (close its GUI) plays plateaus on the bench trigger (EXT), CH1 "
            "-> X1, CH2 -> X2, checked against the Trek limits first. Per bias: 4 "
            "angles find the azimuth, then the analyzer steps +-null deg around the "
            "crossed position at the most sensitive V/div that holds it (Imin), and "
            "once to the bright angle (Imax). The window opens 'settle' ms into the "
            "hold (scope overdrive recovery). The beam is blocked once, for the "
            "darks at every V/div used.")).pack(anchor="w", padx=6, pady=(2, 4))

    def build_ilc(self, f):
        """The ILC target comparison and correction files (rampol.ilc_target)."""
        self.iv = {}
        for label, key, w, browse in (("X1 ILC state", "x1", 30, "state"),
                                      ("X2 ILC state", "x2", 30, "state"),
                                      ("Line ref scan", "line_ref", 30, "scan")):
            rr = ttk.Frame(f)
            rr.pack(fill="x", padx=6, pady=1)
            ttk.Label(rr, text=label, width=12).pack(side="left")
            self.iv[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.iv[key], width=w).pack(side="left", padx=2)
            ttk.Button(rr, text="...", width=3,
                       command=lambda k=key, b=browse: self.pick_ilc_file(k, b)).pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        for label, key, w in (("band Hz", "f_cut", 6), ("PD delay us", "pd_delay_us", 5),
                              ("split X1", "split", 4)):
            ttk.Label(rr, text=label).pack(side="left", padx=(0, 2))
            self.iv[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.iv[key], width=w).pack(side="left", padx=(0, 6))
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(3, 1))
        self._btn(rr, "Compare shown scan", self.do_ilc_compare)
        ttk.Button(rr, text="Open output folder", command=self.open_ilc_folder).pack(
            side="left", padx=4)
        ttk.Label(f, foreground="#666", justify="left", wraplength=330, text=(
            "Light against the target the ILC was given (P92PX1H/P92PX2A), time-"
            "mapped onto the experiment's waveform. Writes target_<name>_played.csv "
            "(run the ILC on it) and corr_<name>_optical.csv per crystal for the "
            "ILC panel's Corrections tab: minus (light - monitors), so the loop does "
            "not count the monitors' own error twice. Line ref: the same sequence "
            "with the ramps off, so the 60 Hz ripple is not baked into the "
            "correction. PD delay: the photodiode chain's own (an RC filter: 3.1 "
            "us).")).pack(anchor="w", padx=6, pady=(2, 4))

    def pick_ilc_file(self, key, kind):
        if kind == "state":
            p = filedialog.askopenfilename(
                title="ILC state", parent=self.root, filetypes=[("ILC state", "*.state.npz")],
                initialdir=os.path.dirname(self.iv[key].get()) or None)
        else:
            p = filedialog.askdirectory(title="Drive-off scan of the same sequence",
                                        parent=self.root, initialdir=self.outdir.get())
        if p:
            self.iv[key].set(p)

    # -- bias points ---------------------------------------------------------------
    def ask_main(self, title, text):
        """A yes/no from the worker thread, asked on the Tk thread. In simulate
        mode it also blocks/unblocks the simulated beam."""
        if self.stop_flag.is_set():
            return False
        ans, ev = [], threading.Event()

        def ui():
            try:
                ans.append(messagebox.askokcancel(title, text, parent=self.root))
            finally:
                ev.set()
        self.call(ui)
        while not ev.wait(0.2):
            if self.stop_flag.is_set():
                return False
        ok = bool(ans and ans[0])
        if ok and self.bench is not None:
            if "Block the beam" in text:
                self.bench._imax_saved, self.bench.imax = self.bench.imax, 0.0
            elif "Unblock" in text and hasattr(self.bench, "_imax_saved"):
                self.bench.imax = self.bench._imax_saved
        return ok

    def do_start_bias(self):
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(c).items()}
        if "PD" not in roles:
            self.log("No channel has the PD role.")
            return
        plan = dict(c["bias"])
        name = scanmod.safe_name(plan.pop("name", "bias") or "bias")
        new = scanmod.next_free_name(c["outdir"], name)
        if new != name:
            self.log(f"{name} exists - this bias run is {new}")
            self.bv["name"].set(new)
            c["bias"]["name"] = new
            name = new
        sim_mode = self.bench is not None
        prov = self._provenance(c)

        def go():
            from . import bias as biasmod
            eom = ib = None
            if sim_mode:
                awg = sim.FakeAWG(self.bench)
            else:
                eom = hw.load_eomilc(c["eomilc_path"])
                mod = hw.load_module(c["awg_path"], "bk4063b")
                import ilc_bench as ib
                ib._AWGMOD = mod
                awg = mod.BK4063B(connect=False,
                                  resource_manager=getattr(self.link.scope, "rm", None))
                self.log(f"AWG: {awg.connect()}")
            self.bias_live = {"points": [], "name": name, "plan": plan}
            run = biasmod.BiasRun(c["outdir"], name, self.link, self.rot, awg, roles,
                                  plan=plan, log=self.log, cancelled=self.stop_flag.is_set,
                                  ask=self.ask_main, progress=self._progress,
                                  on_point=lambda p: self.call(self._bias_point, p),
                                  eomilc=eom, ilc_bench=ib, provenance=prov)
            try:
                run.run()
            finally:
                if not sim_mode:
                    awg.close()
            return run.folder

        def done(folder):
            self.load_bias(folder)
        self.worker(go, done=done)

    def _bias_point(self, p):
        self.bias_live.setdefault("points", []).append(p)
        from . import bias as biasmod
        self.bias_result = dict(self.bias_live, transfer=biasmod.transfer(
            self.bias_live["points"]))
        self.plot_dirty.add(self.fig_bias._frame)
        self.nb.select(self.fig_bias._frame)
        self.draw_visible()

    def do_load_bias(self):
        d = filedialog.askdirectory(title="A bias run folder (holds bias.json)",
                                    parent=self.root, initialdir=self.outdir.get())
        if d:
            self.load_bias(d)

    def load_bias(self, folder):
        from . import bias as biasmod
        try:
            self.bias_result = biasmod.load(folder)
        except (OSError, ValueError) as exc:
            self.log(f"Bias run not loaded: {exc}")
            return
        tf = self.bias_result.get("transfer")
        n = len(self.bias_result.get("points", []))
        self._lab_upsert(os.path.dirname(os.path.abspath(folder)),
                         lablog.bias_row(self.bias_result))
        self.log(f"Bias run {self.bias_result['name']}: {n} points"
                 + (f"; light / monitors gain {tf['gain']:.4f}, residual "
                    f"{tf['rms_resid']*1e3:.0f} mdeg rms" if tf else ""))
        self.plot_dirty.add(self.fig_bias._frame)
        self.nb.select(self.fig_bias._frame)
        self.draw_visible()

    def _ramp_lm(self):
        """The shown scan's light - monitors against target rotation, from its
        ILC comparison (analysis/target_compare), if there is one."""
        if not self.result:
            return None
        p = os.path.join(self.result["d"].folder, "analysis", "target_compare",
                         "target_compare.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        return z["phi_target"], z["leg1_d_lm"]

    def draw_bias(self, fig):
        r = getattr(self, "bias_result", None)
        if not r or not r.get("points"):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No bias run yet - Bias points tab: Start, or Load",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
            ax.set_axis_off()
            return
        pts = r["points"]
        cols = {"up": "#1f77b4", "down": "#d62728"}
        ax = fig.add_subplot(221)
        for d_ in ("up", "down"):
            ps = [p for p in pts if p.get("dir", "up") == d_]
            if not ps:
                continue
            x = [p["phi_mon"] if p.get("phi_mon") is not None else p["bias"] for p in ps]
            ok = [i for i, p in enumerate(ps) if p.get("er")]
            ax.errorbar([x[i] for i in ok], [ps[i]["er"] for i in ok],
                        [ps[i].get("sig_er") or 0 for i in ok], fmt="o", ms=4,
                        color=cols[d_], label=f"Imax / Imin ({d_})")
            lb = [i for i, p in enumerate(ps) if not p.get("er") and p.get("er_lower")]
            if lb:
                ax.plot([x[i] for i in lb], [ps[i]["er_lower"] for i in lb], "^",
                        color=cols[d_], label="lower bound (Imin unresolved)")
            cv = [i for i in ok if ps[i].get("malus_ratio")]
            ax.plot([x[i] for i in cv],
                    [(ps[i]["imin"] + ps[i]["fit"]["k"]) / ps[i]["imin"] for i in cv],
                    "x", color=cols[d_], alpha=0.6, label="from the null's curvature")
        ax.set_yscale("log")
        ax.set_xlabel("rotation, monitors (deg)")
        ax.set_ylabel("extinction ratio")
        ax.set_title(f"Static extinction ratio ({r['name']})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

        ax = fig.add_subplot(222)
        tf = r.get("transfer")
        if tf:
            phi = np.array(tf["phi_mon"])
            dev = (np.array(tf["rot_light"]) - phi) * 1e3
            dirs = tf.get("dir", ["up"] * len(phi))
            for d_ in ("up", "down"):
                m = np.array([x == d_ for x in dirs])
                if m.any():
                    ax.plot(phi[m], dev[m], "o-", ms=4, color=cols[d_],
                            label=f"static, {d_}")
            ramp = self._ramp_lm()
            if ramp is not None:
                act = ramp[0] > 0.5
                ax.plot(ramp[0][act], ramp[1][act] * 1e3, ",", color="0.5", alpha=0.5,
                        label=f"ramp {self.result['d'].name}, leg 1")
            ax.set_xlabel("rotation, monitors (deg)")
            ax.set_ylabel("light - monitors (mdeg)")
            ax.set_title(f"Light - monitors, static: gain {tf['gain']:.4f}, "
                         f"{tf['rms_resid']*1e3:.0f} mdeg rms left", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        ax = fig.add_subplot(223)
        x = [p["phi_mon"] if p.get("phi_mon") is not None else p["bias"] for p in pts]
        ax.errorbar(x, [p["imin"] * 1e3 for p in pts], [p["sig_imin"] * 1e3 for p in pts],
                    fmt="o", ms=4, label="Imin (dark-subtracted)")
        for xi, p in zip(x, pts):
            if p.get("scan"):
                ax.annotate(f"{p['scan']['vdiv']*1e3:g}", (xi, p["imin"] * 1e3), fontsize=6,
                            xytext=(3, 3), textcoords="offset points", color="#666")
        ax.set_yscale("log")
        ax.set_xlabel("rotation, monitors (deg)")
        ax.set_ylabel("Imin (mV); labels: mV/div")
        ax.grid(alpha=0.3, which="both")

        ax = fig.add_subplot(224)
        p = pts[-1]
        sc = p.get("scan")
        if sc:
            th = np.array(sc["theta"])
            ax.errorbar(th - p["theta_n"], np.array(sc["I"]) * 1e3,
                        np.array(sc["sem"]) * 1e3, fmt="o", ms=4)
            xx = np.linspace(th.min(), th.max(), 200)
            f_ = p["fit"]
            ax.plot(xx - p["theta_n"], (f_["imin"] + f_["k"] * np.sin(np.deg2rad(
                xx - f_["theta_n"])) ** 2) * 1e3, color="k", lw=0.8)
            ax.set_xlabel("analyzer - null (deg)")
            ax.set_ylabel("PD (mV)")
            ax.set_title(f"Null scan at {p['bias']:g} deg: null {p['theta_n']:.3f} "
                         f"+- {p['sig_theta_n']*1e3:.0f} mdeg", fontsize=8)
            ax.grid(alpha=0.3)

    # -- dark and background ---------------------------------------------------------
    OFFSET_PROMPTS = {
        "dark": ("Dark (PD covered)",
                 "Cover the photodiode so no light at all reaches it (cap or card "
                 "over the PD itself), then OK.",
                 "Dark done. Uncover the photodiode, then OK."),
        "background": ("Background (beam blocked)",
                       "Block the laser beam before the EOMs. Leave the room, the PD "
                       "and its cover as they are during the scan, so stray light is "
                       "included. Then OK.",
                       "Background done. Unblock the beam, then OK."),
    }

    def offsets_then(self, run, kinds, after):
        """Ask for each offset measurement in turn (dark: PD covered;
        background: beam blocked), take it, ask to undo it, then after()."""
        kinds = list(kinds)
        if not kinds:
            after()
            return
        kind = kinds.pop(0)
        title, ask, undo = self.OFFSET_PROMPTS[kind]
        if not messagebox.askokcancel(title, ask, parent=self.root):
            self.log(f"{title} skipped.")
            for s in run.manifest["steps"]:
                if s["kind"] == kind and s["status"] != "done":
                    s["status"] = "skipped"
            run.save()
            self.offsets_then(run, kinds, after)
            return
        b = self.bench
        if b is not None:
            if kind == "dark":
                b.covered = True
            else:
                b._imax_saved, b.imax = b.imax, 0.0

        def undone(_):
            if b is not None:
                if kind == "dark":
                    b.covered = False
                else:
                    b.imax = b._imax_saved
            messagebox.showinfo(title, undo, parent=self.root)
            self.offsets_then(run, kinds, after)
        self.worker(lambda: run.run(kinds={kind}), done=undone)

    def borrow_latest(self, run, kinds, then):
        """Worker: the newest dark/background of other scans at this PD
        V/div and offset, written into the run's manifest (borrowed)."""
        roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(self.cfg).items()}
        outdir = self.cfg["outdir"]
        sg = self.load_sg()

        def go():
            pd = roles["PD"]
            vdiv, off = self.link.channel_state([pd])[pd]
            found = an.find_offsets(outdir, vdiv, off, sg.load_capture, exclude=run.folder)
            got = {}
            for kind in kinds:
                f = next((x for x in found if x["kind"] == kind), None)
                if f is None:
                    self.log(f"  no earlier {kind} at {vdiv:g} V/div, offset {off:+.4g} V in "
                             f"{outdir} - nothing borrowed (measure one, or Borrow... later)")
                    continue
                got[kind] = {k: f[k] for k in ("level", "sem", "n", "vdiv", "offset",
                                                "source", "measured")}
                self.log(f"  {kind}: reusing {f['level']*1e3:+.2f} mV from {f['source']} "
                         f"({f['n']} shots, {f['measured']})")
            if got:
                run.manifest.setdefault("borrowed", {}).update(got)
                run.save()
            return got
        self.worker(go, done=lambda _g: then())

    def do_borrow_dialog(self):
        """Pick a dark/background from another scan at the shown scan's PD
        V/div and offset, for the shown scan (written to its manifest)."""
        if not self.result:
            self.log("Load a scan first.")
            return
        d = self.result["d"]
        st = [s for s in d.steps if s["kind"] == "scan"]
        if not st:
            return
        v, off = an._scale_of(d, st[0])[:2]
        found = an.find_offsets(os.path.dirname(d.folder), v, off, self.load_sg().load_capture,
                                exclude=d.folder)
        top = tk.Toplevel(self.root)
        top.title("Borrow a dark or background")
        top.transient(self.root)
        ttk.Label(top, wraplength=520, justify="left", text=(
            f"Measurements in other scans at the PD's {v:g} V/div and {off:+.4g} V offset "
            f"(the scope's offset error depends on both), newest first. The one picked "
            f"is written into {d.name}'s manifest and subtracted (a background is used "
            f"in preference to a dark).")).pack(padx=8, pady=6)
        lb = tk.Listbox(top, width=90, height=min(12, max(3, len(found))))
        for f in found:
            lb.insert("end", f"{f['kind']:10s} {f['level']*1e3:+8.2f} mV  {f['n']:3d} shots  "
                             f"{f['measured']}  {f['source']}")
        lb.pack(padx=8)
        if not found:
            lb.insert("end", "(none at these settings)")

        def use():
            sel = lb.curselection()
            if not sel or not found:
                return
            f = found[sel[0]]
            self._write_borrowed(d.folder, {f["kind"]: {k: f[k] for k in (
                "level", "sem", "n", "vdiv", "offset", "source", "measured")}})
            top.destroy()

        def clear():
            self._write_borrowed(d.folder, None)
            top.destroy()
        r = ttk.Frame(top)
        r.pack(pady=6)
        ttk.Button(r, text="Use selected", command=use).pack(side="left", padx=4)
        ttk.Button(r, text="Remove borrowed", command=clear).pack(side="left", padx=4)
        ttk.Button(r, text="Cancel", command=top.destroy).pack(side="left", padx=4)

    def _write_borrowed(self, folder, entries):
        name = os.path.basename(folder)
        mp = os.path.join(folder, f"{name}_scan.json")
        with open(mp, encoding="utf-8") as fh:
            man = json.load(fh)
        if entries is None:
            man.pop("borrowed", None)
            self.log(f"{name}: borrowed dark/background removed")
        else:
            man.setdefault("borrowed", {}).update(entries)
            for k, e in entries.items():
                self.log(f"{name}: {k} {e['level']*1e3:+.2f} mV borrowed from {e['source']}")
        tmp = mp + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        cfgmod.replace_retrying(tmp, mp)
        self.scan_cache.pop(os.path.normcase(os.path.abspath(folder)), None)
        self.reanalyse()

    def _opts(self):
        """The analysis switches on the plot bar (read on the Tk thread)."""
        return {"sub_dark": bool(self.sub_dark.get()), "gains": bool(self.gains_on.get()),
                "lock": bool(self.lock_on.get())}

    def show_corrections(self):
        res = self.result
        txt = res.get("corr", {}).get("text", "") if res else ""
        self.corr_label.configure(text=("Corrections: " + txt) if txt else "")

    # -- shots: single traces and averages ------------------------------------------
    def build_shots_tab(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="Shots")
        side = ttk.Frame(frame)
        side.pack(side="left", fill="y", padx=(4, 2), pady=4)
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(side, text="Steps (ctrl/shift-click for several)").pack(anchor="w")
        lf = ttk.Frame(side)
        lf.pack(fill="y", expand=True)
        self.shots_lb = tk.Listbox(lf, selectmode="extended", width=30, height=16,
                                   exportselection=False, font=("Consolas", 8))
        sb = ttk.Scrollbar(lf, command=self.shots_lb.yview)
        self.shots_lb.configure(yscrollcommand=sb.set)
        self.shots_lb.pack(side="left", fill="y", expand=True)
        sb.pack(side="left", fill="y")
        self.shots_lb.bind("<<ListboxSelect>>", lambda _e: self.redraw(self.fig_shots))
        self._shots_keys = []
        r = ttk.Frame(side)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Channel").pack(side="left")
        self.shots_role = tk.StringVar(value="PD")
        self.shots_role_cb = ttk.Combobox(r, textvariable=self.shots_role, width=8,
                                          state="readonly", values=("PD",))
        self.shots_role_cb.pack(side="left", padx=4)
        self.shots_role_cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw(self.fig_shots))
        self.sv_shots = {}
        for key, text, val in (("single", "single shots (from the files)", True),
                               ("avg", "average the fit uses", True),
                               ("sem", "+-1 standard error band", False),
                               ("env", "min-max over the shots shown", False),
                               ("dropped", "dropped shots (dashed)", True),
                               ("dark", "subtract dark / background", True),
                               ("partner", "+ the analyzer 90 deg away", False)):
            v = tk.BooleanVar(value=val)
            ttk.Checkbutton(side, text=text, variable=v,
                            command=lambda: self.redraw(self.fig_shots)).pack(anchor="w")
            self.sv_shots[key] = v
        r = ttk.Frame(side)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Shots").pack(side="left")
        self.shots_pick = tk.StringVar(value="all")
        e = ttk.Entry(r, textvariable=self.shots_pick, width=10)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _e: self.redraw(self.fig_shots))
        ttk.Label(r, text="(all, 1, 2-4)", foreground="#666").pack(side="left")
        r = ttk.Frame(side)
        r.pack(fill="x", pady=(2, 0))
        ttk.Label(r, text="t from").pack(side="left")
        self.shots_t0 = tk.StringVar()
        self.shots_t1 = tk.StringVar()
        for var in (self.shots_t0, self.shots_t1):
            e = ttk.Entry(r, textvariable=var, width=7)
            e.pack(side="left", padx=2)
            e.bind("<Return>", lambda _e: self.redraw(self.fig_shots))
            if var is self.shots_t0:
                ttk.Label(r, text="to").pack(side="left")
        ttk.Label(r, text="ms").pack(side="left")
        r = ttk.Frame(side)
        r.pack(fill="x", pady=(4, 0))
        ttk.Button(r, text="Draw", command=lambda: self.redraw(self.fig_shots)).pack(side="left")
        ttk.Button(r, text="Crossed at cursor", command=self.shots_crossed).pack(
            side="left", padx=4)
        ttk.Button(r, text="Whole record", command=self.shots_whole).pack(side="left")
        self.shots_info = ttk.Label(side, text="", foreground="#666", wraplength=230,
                                    justify="left")
        self.shots_info.pack(anchor="w", pady=(4, 0))
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=right)
        toolbar = NavigationToolbar2Tk(canvas, right)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._canvas, fig._toolbar, fig._ctl, fig._frame = canvas, toolbar, side, frame
        canvas.mpl_connect("button_press_event", lambda ev: self._shots_click(ev))
        self.plot_tabs[frame] = (fig, self.draw_shots)
        self.plot_dirty.add(frame)
        self.fig_shots = fig
        self._shot_cache = {}
        self._shots_drawing = False

    def _step_label(self, s):
        k = s["kind"]
        n = s.get("nb", 0)
        rej = s.get("rejected", 0)
        tail = f"{n:2d} shots" + (f", {rej} dropped" if rej else "")
        if k in ("dark", "background"):
            return f"{k:10s}        {tail}"
        a = s.get("landed", s.get("target", 0.0))
        if k == "ref":
            return f"ref #{s.get('ref', 0):<4d} {a:7.2f}  {tail}"
        if k == "null":
            return f"null w{s.get('window', 0) + 1:<4d} {a:7.2f}  {tail}"
        return f"scan       {a:7.2f}  {tail}"

    @staticmethod
    def _step_short(s):
        k = s["kind"]
        if k in ("dark", "background"):
            return k
        a = s.get("landed", s.get("target", 0.0))
        if k == "ref":
            return f"ref #{s.get('ref', 0)} ({a:.1f} deg)"
        if k == "null":
            return f"null w{s.get('window', 0) + 1} {a:.2f} deg"
        return f"analyzer {a:.1f} deg"

    def _shots_fill(self, d):
        keys = [(s["kind"], s.get("ref"), s.get("window"), round(s.get("target", 0.0), 4))
                for s in d.steps]
        if keys == self._shots_keys:
            return
        sel = {self._shots_keys[i] for i in self.shots_lb.curselection()
               if i < len(self._shots_keys)}
        self.shots_lb.delete(0, "end")
        for s in d.steps:
            self.shots_lb.insert("end", self._step_label(s))
        self._shots_keys = keys
        idx = [i for i, k in enumerate(keys) if k in sel]
        if not idx:
            # a sensible start: the first scan angle
            idx = [next((i for i, s in enumerate(d.steps) if s["kind"] == "scan"), 0)]
        for i in idx:
            self.shots_lb.selection_set(i)
        roles = sorted(d.roles, key=lambda r: (r != "PD", r))
        self.shots_role_cb["values"] = roles
        if self.shots_role.get() not in roles:
            self.shots_role.set("PD")

    def _shots_of(self, d, s):
        key = (d.folder, tuple(s.get("files", [])))
        hit = self._shot_cache.get(key)
        if hit is None:
            hit = an.step_shots(d, s, self.load_sg().load_capture,
                                trim=int(self.cfg["analysis"]["trim"]))
            self._shot_cache[key] = hit
            while len(self._shot_cache) > 8:
                self._shot_cache.pop(next(iter(self._shot_cache)))
        return hit

    @staticmethod
    def _pick_shots(text, n):
        text = (text or "").strip().lower()
        if not text or text == "all":
            return list(range(n))
        out = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                out += list(range(int(a) - 1, int(b)))
            elif part:
                out.append(int(part) - 1)
        return [i for i in out if 0 <= i < n]

    @staticmethod
    def _decimate(t, y, n=3000):
        """Min and max of each bin, so a dip survives the thinning."""
        if len(t) <= 2 * n:
            return t, y
        k = len(t) // n
        m = (len(t) // k) * k
        tb = t[:m].reshape(-1, k)
        yb = y[:m].reshape(-1, k)
        lo, hi = yb.min(axis=1), yb.max(axis=1)
        tt = np.repeat(tb.mean(axis=1), 2)
        yy = np.empty(2 * len(lo))
        yy[0::2], yy[1::2] = lo, hi
        return tt, yy

    def _shots_range(self, d):
        try:
            t0 = float(self.shots_t0.get()) * 1e-3 if self.shots_t0.get().strip() else d.t[0]
            t1 = float(self.shots_t1.get()) * 1e-3 if self.shots_t1.get().strip() else d.t[-1]
        except ValueError:
            t0, t1 = d.t[0], d.t[-1]
        return min(t0, t1), max(t0, t1)

    def draw_shots(self, fig):
        res = self.result
        d = res["d"]
        self._shots_fill(d)
        sel = list(self.shots_lb.curselection())
        role = self.shots_role.get() or "PD"
        o = {k: v.get() for k, v in self.sv_shots.items()}
        steps = [d.steps[i] for i in sel if i < len(d.steps)]
        if o["partner"]:
            extra = []
            for s in steps:
                if s["kind"] in ("scan", "ref") and "landed" in s:
                    want = (s["landed"] + 90) % 180
                    p = min((x for x in d.steps if x["kind"] == "scan" and "landed" in x),
                            key=lambda x: abs((x["landed"] - want + 90) % 180 - 90), default=None)
                    if p is not None and p not in steps and p not in extra:
                        extra.append(p)
            steps += extra
        ax = fig.add_subplot(111)
        if not steps:
            ax.text(0.5, 0.5, "Pick steps in the list", ha="center", transform=ax.transAxes,
                    color="#888")
            ax.set_axis_off()
            return
        t0, t1 = self._shots_range(d)
        cmap = matplotlib.colormaps[ANGLE_CMAP]
        scale = 1e3
        n_drawn, info = 0, []
        self._shots_drawing = True
        for s in steps:
            if s["kind"] in ("dark", "background"):
                col = "k"
            else:
                col = cmap((s.get("landed", s.get("target", 0)) % 360) / 360)
            lab = self._step_short(s)
            sub = 0.0
            if o["dark"] and role == "PD":
                sub = an.dark_level(d, an._scale_of(d, s)[0])[0]
            if o["single"] or o["env"] or o["dropped"]:
                t, sh, files = self._shots_of(d, s)
                y_all = sh.get(role)
                if y_all is not None and len(y_all):
                    m = (t >= t0) & (t <= t1)
                    pick = self._pick_shots(self.shots_pick.get(), len(y_all))
                    kept = s.get("kept") if role == "PD" else None
                    shown = []
                    for i in pick:
                        dropped = kept is not None and i < len(kept) and not kept[i]
                        if dropped and not o["dropped"]:
                            continue
                        y = y_all[i][m] - sub
                        shown.append(y)
                        if o["single"] or dropped:
                            tt, yy = self._decimate(t[m], y)
                            ax.plot(tt * 1e3, yy * scale, lw=0.5,
                                    color="#d62728" if dropped else col,
                                    ls="--" if dropped else "-",
                                    alpha=0.9 if dropped else 0.55,
                                    label=(f"{lab}: shot {i + 1} (dropped)" if dropped
                                           else (f"{lab}: shots" if not n_drawn or i == pick[0]
                                                 else None)))
                            n_drawn += 1
                    if o["env"] and shown:
                        A = np.array(shown)
                        tt, lo = self._decimate(t[m], A.min(axis=0))
                        _, hi = self._decimate(t[m], A.max(axis=0))
                        ax.fill_between(tt * 1e3, lo * scale, hi * scale, color=col, alpha=0.15,
                                        lw=0)
                    info.append(f"{lab}: {len(y_all)} shots ({len(files)} files)")
            if o["avg"] and role in s.get("v", {}):
                m = (d.t >= t0) & (d.t <= t1)
                y = s["v"][role][m] - sub
                tt, yy = self._decimate(d.t[m], y)
                ax.plot(tt * 1e3, yy * scale, lw=1.4, color=col,
                        label=f"{lab}: average of {s.get('nb', 0)} kept")
                if o["sem"]:
                    e = s["sem"][role][m]
                    tt, lo = self._decimate(d.t[m], y - e)
                    _, hi = self._decimate(d.t[m], y + e)
                    ax.fill_between(tt * 1e3, lo * scale, hi * scale, color=col, alpha=0.25, lw=0)
        if self.cursor_t is not None and t0 <= self.cursor_t <= t1:
            ax.axvline(self.cursor_t * 1e3, color="0.4", lw=0.6, ls=":")
        ax.set_xlim(t0 * 1e3, t1 * 1e3)
        ax.set_xlabel("time from trigger (ms)")
        ax.set_ylabel(f"{role} (mV)" + (" - dark/background" if o["dark"] and role == "PD" else ""))
        ax.set_title(f"{d.name}: single shots from the files; averages as the fit "
                     f"uses them", fontsize=8)
        ax.grid(alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        if l:
            ax.legend(fontsize=7, loc="best", ncol=1 + len(l) // 8)
        self.shots_info.configure(text="\n".join(info[:6]) + (
            "\nZoom with the toolbar; the view re-reads full resolution." if info else ""))
        ax.callbacks.connect("xlim_changed", self._shots_zoomed)
        self._shots_drawing = False

    def _shots_zoomed(self, ax):
        if self._shots_drawing:
            return
        a, b = ax.get_xlim()
        self.shots_t0.set(f"{a:.4f}")
        self.shots_t1.set(f"{b:.4f}")
        if getattr(self, "_shots_after", None):
            self.root.after_cancel(self._shots_after)
        self._shots_after = self.root.after(150, lambda: self.redraw(self.fig_shots))

    def _shots_click(self, ev):
        if ev.inaxes is None or ev.xdata is None or self.fig_shots._toolbar.mode:
            return
        self.cursor_t = ev.xdata * 1e-3
        self.cursor_var.set(f"{ev.xdata:.3f}")

    def shots_whole(self):
        self.shots_t0.set("")
        self.shots_t1.set("")
        self.redraw(self.fig_shots)

    def shots_crossed(self):
        """Select the scan angle closest to crossed at the cursor time, and
        the one 90 deg from it, around the cursor: the view a direct ER
        reading is made from."""
        res = self.result
        if not res or res.get("pol") is None or self.cursor_t is None:
            self.log("Set a cursor time (click a plot) on a scan with a fit first.")
            return
        pol, d = res["pol"], res["d"]
        j = int(np.argmin(np.abs(d.t - self.cursor_t)))
        want = (pol["psi"][j] + 90) % 180
        idx = [i for i, s in enumerate(d.steps) if s["kind"] == "scan" and "landed" in s]
        best = min(idx, key=lambda i: abs((d.steps[i]["landed"] - want + 90) % 180 - 90))
        self.shots_lb.selection_clear(0, "end")
        self.shots_lb.selection_set(best)
        self.shots_lb.see(best)
        self.sv_shots["partner"].set(True)
        self.shots_t0.set(f"{(self.cursor_t - 0.4e-3) * 1e3:.4f}")
        self.shots_t1.set(f"{(self.cursor_t + 0.4e-3) * 1e3:.4f}")
        self.log(f"Crossed at {self.cursor_t*1e3:.3f} ms: azimuth {pol['psi'][j]:.2f} deg, "
                 f"nearest crossed analyzer {d.steps[best]['landed']:.2f} deg")
        self.nb.select(self.fig_shots._frame)
        self.redraw(self.fig_shots)

    # -- build: how the angles add up -------------------------------------------------
    def build_build_tab(self):
        self.fig_build = self._fig_tab("Build", self.draw_build)
        ctl = self.fig_build._ctl
        ttk.Label(ctl, text="time").pack(side="left")
        self.build_t = tk.DoubleVar(value=0.0)
        self.build_scale = ttk.Scale(ctl, from_=0, to=1, orient="horizontal", length=360,
                                     variable=self.build_t, command=lambda _v: self._build_move())
        self.build_scale.pack(side="left", padx=4)
        self.build_lbl = ttk.Label(ctl, text="", width=12)
        self.build_lbl.pack(side="left")
        self.build_raw = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="show before corrections", variable=self.build_raw,
                        command=lambda: self._build_move()).pack(side="left", padx=8)
        ttk.Label(ctl, foreground="#666", text="drag: every angle's average (top) gives one "
                  "point per time; the points fit to a0 + B cos 2(theta - psi)").pack(side="left")
        self._build = None

    def draw_build(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t
        gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
        ax_t = fig.add_subplot(gs[0, :])
        ax_m = fig.add_subplot(gs[1, 0])
        ax_r = fig.add_subplot(gs[1, 1], sharex=ax_t)
        cmap = matplotlib.colormaps[ANGLE_CMAP]
        for th, y in zip(pol["theta"], pol["I"]):
            tt, yy = self._decimate(t, y, 2000)
            ax_t.plot(tt * 1e3, yy, lw=0.5, color=cmap((th % 360) / 360))
        ax_t.set_ylabel("PD, corrected (V)")
        ax_t.set_title(f"{d.name}: the {pol['n_angles']} analyzer angles' averaged traces, "
                       f"as fitted", fontsize=9)
        ax_t.grid(alpha=0.3)
        vl_t = ax_t.axvline(0, color="k", lw=0.8)
        tt, rr = self._decimate(t, pol["rotation"], 2000)
        ax_r.plot(tt * 1e3, rr, lw=0.8, color="#1f77b4")
        mk, = ax_r.plot([], [], "o", color="#d62728", ms=6)
        vl_r = ax_r.axvline(0, color="k", lw=0.6)
        ax_r.set_xlabel("time from trigger (ms)")
        ax_r.set_ylabel("rotation from rest (deg)")
        ax_r.grid(alpha=0.3)
        g = np.linspace(-5, 185, 381)
        corr, = ax_m.plot([], [], "o", ms=5, color="#1f77b4", label="as fitted (corrected)")
        raw, = ax_m.plot([], [], "o", ms=5, mfc="none", color="0.5", label="raw step average")
        fitl, = ax_m.plot([], [], color="k", lw=0.9, label="a0 + B cos 2(theta - psi)")
        vmax = ax_m.axvline(0, color="#2ca02c", lw=0.8, ls="--", label="psi (max)")
        vmin = ax_m.axvline(0, color="#9467bd", lw=0.8, ls=":", label="psi + 90 (null)")
        txt = ax_m.set_title("", fontsize=7, family="monospace", loc="left")
        ax_m.set_xlim(-5, 185)
        ax_m.set_xlabel("analyzer angle (deg, mod 180)")
        ax_m.set_ylabel("PD (V)")
        ax_m.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
                    frameon=False)
        ax_m.grid(alpha=0.3)
        self._build = dict(fig=fig, ax_m=ax_m, ax_r=ax_r, vl_t=vl_t, vl_r=vl_r, mk=mk,
                           corr=corr, raw=raw, fitl=fitl, vmax=vmax, vmin=vmin, txt=txt,
                           g=g, res=res)
        self.build_scale.configure(from_=t[0] * 1e3, to=t[-1] * 1e3)
        if self.cursor_t is not None:
            self.build_t.set(self.cursor_t * 1e3)
        elif not (t[0] * 1e3 <= self.build_t.get() <= t[-1] * 1e3):
            self.build_t.set(t[len(t) // 3] * 1e3)
        self._build_move(draw=False)

    def _build_move(self, draw=True):
        b = self._build
        if not b or b["res"] is not self.result or self.result.get("pol") is None:
            return
        pol, d = self.result["pol"], self.result["d"]
        t = d.t
        j = int(np.argmin(np.abs(t - self.build_t.get() * 1e-3)))
        tj = t[j] * 1e3
        self.build_lbl.configure(text=f"{tj:8.3f} ms")
        th = np.asarray(pol["theta"]) % 180
        y = pol["I"][:, j]
        b["corr"].set_data(th, y)
        if self.build_raw.get():
            b["raw"].set_data(th, [s["v"]["PD"][j] for s in pol["steps"]])
        else:
            b["raw"].set_data([], [])
        gr = np.deg2rad(b["g"])
        model = pol["a0"][j] + pol["c2"][j] * np.cos(2 * gr) + pol["s2"][j] * np.sin(2 * gr)
        b["fitl"].set_data(b["g"], model)
        psi = pol["psi"][j] % 180
        b["vmax"].set_xdata([psi, psi])
        b["vmin"].set_xdata([(psi + 90) % 180] * 2)
        ys = np.r_[y, model]
        pad = 0.05 * (ys.max() - ys.min() + 1e-6)
        b["ax_m"].set_ylim(ys.min() - pad, ys.max() + pad)
        b["txt"].set_text(f"t = {tj:.3f} ms  psi {pol['psi'][j]:+.3f} deg  rotation "
                          f"{pol['rotation'][j]:+.2f} deg\nImax {pol['imax'][j]:.4f} V  "
                          f"Imin {pol['imin'][j]*1e3:.2f} mV  ER_fit {pol['er'][j]:.0f}")
        for vl in (b["vl_t"], b["vl_r"]):
            vl.set_xdata([tj, tj])
        b["mk"].set_data([tj], [pol["rotation"][j]])
        if draw:
            b["fig"]._canvas.draw_idle()

    # -- corrections up front -------------------------------------------------------
    def draw_corrections(self, fig):
        res = self.result
        d, pol = res["d"], res["pol"]
        cs = res.get("corr") or an.corrections_summary(d, pol)
        ax = fig.add_subplot(221)
        t = d.t * 1e3
        lv = an.offset_levels(d, an._pd_vdiv(d, "scan"))
        for kind, col in (("dark", "k"), ("background", "#ff7f0e")):
            e = lv.get(kind)
            if not e:
                continue
            if e.get("step") is not None:
                tt, yy = self._decimate(d.t, e["step"]["v"]["PD"], 2000)
                ax.plot(tt * 1e3, yy * 1e3, lw=0.6, color=col, alpha=0.7)
            ax.axhline(e["level"] * 1e3, color=col, lw=1.0, ls="--",
                       label=f"{kind}: {e['level']*1e3:+.2f} mV ({e['source']}, {e['what']})")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("PD (mV)")
        k = cs.get("subtracted_kind")
        ax.set_title(f"Subtracted from every PD trace: "
                     + (f"{k} {cs['subtracted']*1e3:+.2f} mV" if k else "nothing"), fontsize=9)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(222)
        if pol is not None and len(pol.get("ref_levels", [])) >= 2:
            c = np.asarray(pol["ref_clocks"])
            L = np.asarray(pol["ref_levels"])
            c0 = c.min()
            ax.plot((c - c0) / 60, (L / L.mean() - 1) * 100, "o-", ms=4)
            ax.set_xlabel("minutes into the scan")
            ax.set_ylabel("reference level - mean (%)")
            ax.set_title("Intensity drift from the reference returns"
                         + ("" if self.drift_on.get() else " (NOT applied)"), fontsize=9)
        else:
            ax.text(0.5, 0.5, "fewer than 2 reference returns: no drift correction",
                    ha="center", transform=ax.transAxes, color="#888")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(223)
        if pol is not None and pol.get("angle_gain") is not None:
            g = pol["angle_gain"]
            o = np.argsort(np.asarray(pol["theta"]) % 360)
            ax.plot(np.asarray(pol["theta"])[o] % 360, (g[o] - 1) * 100, "o-", ms=4)
            ax.set_title("Per-angle transmission, divided out", fontsize=9)
        else:
            ax.text(0.5, 0.5, "per-angle transmission not fitted (off, or < 8 angles)",
                    ha="center", transform=ax.transAxes, color="#888")
        ax.set_xlabel("analyzer angle (deg)")
        ax.set_ylabel("transmission - mean (%)")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(224)
        st = [s for s in d.steps if s["kind"] not in an.OFFSET_KINDS]
        x = np.arange(len(st))
        kept = [s.get("nb", 0) for s in st]
        drop = [s.get("rejected", 0) for s in st]
        ax.bar(x, kept, color="#1f77b4", label="kept")
        ax.bar(x, drop, bottom=kept, color="#d62728", label="dropped (lock missed)")
        off = [i for i, s in enumerate(st)
               if any(np.any(v) for v in s.get("offscreen", {}).values())]
        if off:
            ax.plot(off, [kept[i] + drop[i] + 0.5 for i in off], "v", color="k",
                    label="samples off screen")
        ax.set_xlabel("step, in measuring order")
        ax.set_ylabel("shots")
        ax.set_title(f"{cs['dropped'][0]} of {cs['dropped'][1]} shots dropped", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3, axis="y")

    # -- find the min / max transmission angle --------------------------------------
    def build_find(self, f):
        self.fv = {}
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        ttk.Label(rr, text="Find").pack(side="left")
        self.find_kind = tk.StringVar(value="min")
        ttk.Combobox(rr, textvariable=self.find_kind, values=("min", "max"), width=5,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(rr, text="transmission in window (ms)").pack(side="left")
        self.fv["window"] = tk.StringVar()
        ttk.Entry(rr, textvariable=self.fv["window"], width=12).pack(side="left", padx=4)
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        self.find_bias_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(rr, text="hold with the AWG at", variable=self.find_bias_on).pack(side="left")
        self.fv["bias"] = tk.StringVar()
        ttk.Entry(rr, textvariable=self.fv["bias"], width=6).pack(side="left", padx=2)
        ttk.Label(rr, text="deg").pack(side="left")
        for label, key, w in (("+-deg", "half", 4), ("points", "points", 3), ("shots", "shots", 3)):
            ttk.Label(rr, text=label).pack(side="left", padx=(8, 2))
            self.fv[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.fv[key], width=w).pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(3, 1))
        self._btn(rr, "Find and go there", self.do_find_angle)
        self.find_zero_btn = ttk.Button(rr, text="Make it analyzer 0", state="disabled",
                                        command=self.do_find_zero)
        self.find_zero_btn.pack(side="left", padx=6)
        self.find_lbl = ttk.Label(f, text="", foreground="#060", wraplength=330)
        self.find_lbl.pack(anchor="w", padx=6)
        ttk.Label(f, foreground="#666", justify="left", wraplength=330, text=(
            "Window: the rest before the ramp (-10:-0.5) or a hold; ticked, the AWG "
            "holds a bias instead. 4 angles give the azimuth, then +-deg steps around "
            "crossed (min, at the most sensitive V/div) or aligned (max) are fitted. "
            "'Make it analyzer 0': crossed reads 0 deg.")).pack(anchor="w", padx=6, pady=(2, 4))

    def do_find_angle(self):
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        fcfg = c["find"]
        roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(c).items()}
        kind = self.find_kind.get()
        try:
            win = None
            txt = str(fcfg.get("window", "")).strip()
            if txt:
                a, b = (float(x) * 1e-3 for x in txt.split(":"))
                win = (min(a, b), max(a, b))
            bias_deg = float(fcfg["bias"]) if self.find_bias_on.get() else None
        except ValueError:
            self.log("Find: window as from:to in ms (e.g. -10:-0.5), bias in deg")
            return
        plan = {"shots": int(fcfg.get("shots") or 8)}
        sim_mode = self.bench is not None

        def go():
            from . import bias as biasmod
            awg = eom = ib = None
            if bias_deg is not None:
                if sim_mode:
                    awg = sim.FakeAWG(self.bench)
                else:
                    eom = hw.load_eomilc(c["eomilc_path"])
                    mod = hw.load_module(c["awg_path"], "bk4063b")
                    import ilc_bench as ib
                    ib._AWGMOD = mod
                    awg = mod.BK4063B(connect=False,
                                      resource_manager=getattr(self.link.scope, "rm", None))
                    self.log(f"AWG: {awg.connect()}")
            try:
                return biasmod.find_extremum(
                    self.link, self.rot, roles, kind, window_s=win, bias_deg=bias_deg,
                    awg=awg, plan=plan,
                    half_deg=float(fcfg["half"]) if str(fcfg.get("half", "")).strip() else None,
                    points=int(fcfg["points"]) if str(fcfg.get("points", "")).strip() else None,
                    log=self.log, cancelled=self.stop_flag.is_set, ask=self.ask_main,
                    eomilc=eom, ilc_bench=ib)
            finally:
                if awg is not None and not sim_mode:
                    awg.close()

        def done(out):
            self.find_result = out
            self._lab_upsert(c["outdir"], lablog.find_row(out))
            self.find_lbl.configure(
                text=f"{out['kind']} at analyzer {out['angle']:.3f} +- "
                     f"{out['sig']*1e3:.0f} mdeg ({out['level']*1e3:.2f} mV raw); "
                     f"the analyzer is there now")
            self.find_zero_btn.configure(state="normal" if out["kind"] == "min" else "disabled")
            self.show_pos(out["angle"])
            self.plot_dirty.add(self.fig_find._frame)
            self.nb.select(self.fig_find._frame)
            self.draw_visible()
        self.worker(go, done=done)

    def do_find_zero(self):
        out = getattr(self, "find_result", None)
        if not out or out["kind"] != "min" or self.rot is None:
            return
        z = (self.rot.zero + out["angle"]) % 360.0
        if not messagebox.askyesno(
                "Analyzer zero", f"Crossed was found at analyzer {out['angle']:.3f} deg. "
                f"Set zero = mount {z:.3f} deg so that it reads 0?", parent=self.root):
            return
        self.zero_var.set(f"{z:.3f}")
        self.do_apply_zero()
        self.show_pos(self.rot.position())

    def draw_find(self, fig):
        out = getattr(self, "find_result", None)
        ax = fig.add_subplot(111)
        if not out:
            ax.text(0.5, 0.5, "Nothing found yet - Find angle tab", ha="center",
                    transform=ax.transAxes, color="#888")
            ax.set_axis_off()
            return
        th = np.array(out["theta"])
        I = np.array(out["I"])
        ax.errorbar(th, I * 1e3, np.array(out["sem"]) * 1e3, fmt="o", ms=4)
        f_ = out["fit"]
        xx = np.linspace(th.min(), th.max(), 300)
        model = f_["imin"] + f_["k"] * np.sin(np.deg2rad(xx - f_["theta_n"])) ** 2
        ax.plot(xx, (model if out["kind"] == "min" else -model) * 1e3, color="k", lw=0.9)
        ax.axvline(out["angle"] if abs(out["angle"] - th.mean()) < 90 else out["angle"] + 180,
                   color="#d62728", lw=0.8, ls="--")
        ax.set_xlabel("analyzer angle (deg)")
        ax.set_ylabel(f"PD (mV, raw, at {out['vdiv']*1e3:g} mV/div)")
        w = out.get("window")
        where = (f"held at {out['bias']:g} deg by the AWG" if out.get("bias") is not None
                 else ("whole record" if not w else f"window {w[0]*1e3:.2f}..{w[1]*1e3:.2f} ms"))
        ax.set_title(f"{out['kind']} transmission at {out['angle']:.3f} +- "
                     f"{out['sig']*1e3:.0f} mdeg ({where})", fontsize=9)
        ax.grid(alpha=0.3)

    # -- ILC target ----------------------------------------------------------------
    def do_ilc_compare(self):
        if not self.result:
            self.log("Load the ramp scan to compare first (Plot data: Scan).")
            return
        c = self.gather()
        self.save_settings()
        i = c["ilc"]
        folder = self.result["d"].folder

        def go():
            from . import ilc_target
            return ilc_target.compare(
                folder, i["x1"], i["x2"], f_cut=float(i["f_cut"]),
                lock_tol=float(c["analysis"].get("lock_tol", 0.006)),
                pd_delay_us=float(i["pd_delay_us"]), split=float(i["split"]),
                line_ref=i.get("line_ref") or None,
                scope_grab_path=c["scope_grab_path"], eomilc_path=c["eomilc_path"],
                log=self.log)

        def done(summary):
            self.ilc_summary = summary
            for f_ in summary["correction"]["files"]:
                self.log(f"  wrote {f_}")
            self.log("  ILC panel: Init on target_<name>_played.csv, converge, then "
                     "Corrections -> Add corr_<name>_optical.csv -> Preview -> Apply")
            self.plot_dirty.add(self.fig_ilc._frame)
            self.nb.select(self.fig_ilc._frame)
            self.draw_visible()
        self.worker(go, done=done)

    def open_ilc_folder(self):
        s = getattr(self, "ilc_summary", None)
        out = s["out"] if s else (os.path.join(self.result["d"].folder, "analysis",
                                               "target_compare") if self.result else None)
        if out and os.path.isdir(out):
            os.startfile(out) if hasattr(os, "startfile") else self.log(out)
        else:
            self.log("Nothing written yet for the shown scan.")

    def draw_ilc(self, fig):
        s = getattr(self, "ilc_summary", None)
        out = s["out"] if s else (os.path.join(self.result["d"].folder, "analysis",
                                               "target_compare") if self.result else None)
        name = self.ilc_fig.get()
        p = os.path.join(out, name) if out else None
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        if not p or not os.path.exists(p):
            ax.text(0.5, 0.5, "No ILC comparison for the shown scan yet - ILC target "
                              "tab: Compare shown scan", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return
        import matplotlib.image as mpimg
        ax.imshow(mpimg.imread(p))

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
        self.dark_mode.set(s.get("dark_mode", "none"))
        self.bg_mode.set(s.get("bg_mode", "measure"))
        for k, v in self.fv.items():
            v.set(str(c["find"].get(k, "")))
        self.find_kind.set(c["find"].get("kind", "min"))
        self.find_bias_on.set(bool(c["find"].get("bias_on", False)))
        self.order.set(s["order"])
        self.mode.set(s["mode"])
        self.preset.set(c["preset"])
        self.outdir.set(c["outdir"])
        self.scan_name.set(c["scan_name"])
        for k, v in self.rv.items():
            v.set(str(c["refine"].get(k, "")))
        for k, v in self.bv.items():
            v.set(str(c["bias"].get(k, "")))
        self.bias_order.set(c["bias"].get("order", "up"))
        for k, v in self.iv.items():
            v.set(str(c["ilc"].get(k, "")))

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
        s["dark_mode"], s["bg_mode"] = self.dark_mode.get(), self.bg_mode.get()
        fd = c["find"]
        for k, v in self.fv.items():
            fd[k] = v.get().strip()
        fd["kind"], fd["bias_on"] = self.find_kind.get(), bool(self.find_bias_on.get())
        c["preset"] = self.preset.get()
        c["outdir"] = self.outdir.get().strip()
        c["scan_name"] = self.scan_name.get().strip()
        for k, v in self.rv.items():
            c["refine"][k] = float(v.get()) if k == "pd_vdiv" and _isnum(v.get()) else v.get().strip()
        b = c["bias"]
        for k, v in self.bv.items():
            txt = v.get().strip()
            if k in ("biases", "name"):
                b[k] = txt
            elif _isnum(txt):
                b[k] = int(float(txt)) if k in ("shots", "null_points") else float(txt)
        b["order"] = self.bias_order.get() or "up"
        i = c["ilc"]
        for k, v in self.iv.items():
            txt = v.get().strip()
            if k in ("x1", "x2", "line_ref"):
                i[k] = txt
            elif _isnum(txt):
                i[k] = float(txt)
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
            # noise that scales with V/div, so a null read at mV/div looks like
            # the bench (see SimScope.noise_per_div)
            self.bench.pd_noise, scope.noise_per_div = 0.2e-3, 0.012
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
        # four decimals: one encoder pulse is 2.5 mdeg
        self.pos_label.configure(text=f"position: {pos:9.4f} deg")

    def do_read_pos(self):
        if self.need(scope=False):
            self.worker(self.rot.position, done=self.show_pos)

    def do_home(self):
        if self.need(scope=False):
            self.target = None
            self.worker(lambda: (self.rot.home(), self.rot.position())[1], done=self.show_pos)

    def do_goto(self):
        if not self.need(scope=False):
            return
        try:
            a = float(self.goto_var.get())
        except ValueError:
            self.log("Go to: not a number")
            return
        self.move_to_target(a)

    def move_to_target(self, a):
        """Go to analyzer angle `a`, approached from below like a scan step,
        and remember it as the target the step and jog buttons count from -
        counting from the read-back position instead would add each landing
        error (~8 mdeg rms on this mount, 5 Oct 2026) to the next."""
        back = float(self.cfg["scan"].get("backoff_deg", 3.0))
        self.target = a
        self.goto_var.set(f"{a:.4f}".rstrip("0").rstrip("."))
        self.worker(lambda: self.rot.approach(a, backoff=back), done=self.show_pos)

    def do_jog(self, d):
        if not self.need(scope=False):
            return
        if getattr(self, "target", None) is None:
            # first relative move after connect or home: start from where it is
            def go():
                return self.rot.position()

            def done(pos):
                self.show_pos(pos)
                self.target = pos
                self.move_to_target(pos + d)
            self.worker(go, done=done)
            return
        self.move_to_target(self.target + d)

    def do_step(self, sign):
        try:
            step = abs(float(self.step_var.get()))
        except ValueError:
            self.log("Step: not a number")
            return
        pulse = 360.0 / getattr(self.rot.dev, "pulses_per_rev", 143360) if self.rot else 0.0025
        if step < pulse:
            self.log(f"Step: {step:g} deg is below one encoder pulse ({pulse:.4f} deg)")
            return
        if step < 0.02:
            self.log(f"Step {step:g} deg: below the ~8 mdeg rms landing scatter measured "
                     f"on this mount - read the position back to see where it went")
        self.do_jog(sign * step)

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
        name = self.preset.get()
        p = cfgmod.all_presets(self.cfg).get(name)
        if not p:
            return
        roles = self.roles()
        writes = self.link.preset_writes(p, roles)
        plan = dict(self.gather()["scan"])

        def go():
            bad, errs = self.link.apply_checked(writes)
            st = self.link.scope.read_settings()
            return bad, errs, checks.settings_checks(st, self.link.prof, roles, plan)

        def done(out):
            bad, errs, found = out
            self.log(f"Preset '{name}': wrote {len(writes)} settings"
                     + ("" if bad or errs else ", every one read back as written."))
            for root, (want, got) in bad.items():
                self.log(f"  ! {root}: wrote {want}, scope reads {got}")
            for e in errs:
                self.log(f"  ! scope error: {e}")
            self.report_checks(found, "Settings check", popup=bool(bad or errs))
            if getattr(self, "set_win", None) is not None and self.set_win.winfo_exists():
                self.do_read_scope()
        self.worker(go, done=done)

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

    def report_checks(self, found, title, popup=True):
        """Log every finding; pop up the WARN/FAIL ones. Returns the worst level."""
        worst = checks.summary(found)
        self.log(f"{title}: {worst}")
        for lv, msg in found:
            self.log(f"  {lv:4s} {msg}")
        bad = [f"{lv}: {msg}" for lv, msg in found if lv in ("WARN", "FAIL")]
        if popup and bad:
            messagebox.showwarning(title, "\n\n".join(bad), parent=self.root)
        return worst

    def do_check_scope(self, then=None):
        """Judge the scope settings, then take one shot the way the scan will
        and judge that: clipping, screen use, light, ramps inside the record,
        trigger. `then(worst, findings)` runs afterwards on the Tk thread."""
        if not self.need(ell=False):
            return
        c = self.gather()
        roles = self.roles()
        if not roles:
            self.log("No channel is recorded.")
            return
        plan = dict(c["scan"])
        self.log("Checking the scope: settings, then one shot "
                 f"(waits up to {plan.get('wait_s', 10):g} s for a trigger)...")

        def go():
            st = self.link.scope.read_settings()
            found = checks.settings_checks(st, self.link.prof, roles, plan)
            try:
                t, tr, st2 = self.link.test_shot(
                    list(roles), plan.get("mode", "single"), plan.get("points"),
                    wait_s=float(plan.get("wait_s", 10)), cancelled=self.stop_flag.is_set)
                found += checks.shot_checks(t, tr, st2, self.link.prof, roles, plan)
            except hw.Cancelled:
                raise
            except Exception as exc:
                found.append(("FAIL", f"test shot: {exc}"))
            return found

        def done(found):
            self.last_check = [list(x) for x in found]
            worst = self.report_checks(found, "Scope check", popup=then is None)
            if then is not None:
                then(worst, found)
        self.worker(go, done=done)

    def do_start_scan(self):
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        if "PD" not in [r for r, _ in cfgmod.channel_roles(c).values()]:
            self.log("No channel has the PD role.")
            return
        self.last_check = None
        if not self.check_first.get():
            self._start_scan_now()
            return

        def after(worst, found):
            bad = [f"{lv}: {msg}" for lv, msg in found if lv in ("WARN", "FAIL")]
            if worst == "FAIL":
                if not messagebox.askyesno(
                        "Scope check failed", "\n\n".join(bad)
                        + "\n\nStart the scan anyway?", parent=self.root):
                    self.log("Scan not started.")
                    return
            elif worst == "WARN":
                if not messagebox.askokcancel(
                        "Scope check", "\n\n".join(bad) + "\n\nOK to start the scan.",
                        parent=self.root):
                    self.log("Scan not started.")
                    return
            self._start_scan_now()
        self.do_check_scope(then=after)

    def _start_scan_now(self):
        c = self.cfg
        run = self._new_run(c["scan_name"])
        if run.exists():
            man = run.load()
            new = scanmod.next_free_name(c["outdir"], run.name)
            unfinished = [x for x in man["steps"] if x.get("status") != "done"]
            ans = False
            if unfinished:
                ans = messagebox.askyesnocancel(
                    "Scan exists", f"{run.name} already exists with {len(unfinished)} "
                    f"step(s) not measured.\n\nYes = resume it\nNo = new scan "
                    f"{new}\nCancel = do nothing", parent=self.root)
                if ans is None:
                    return
            if ans:
                self.run = run
                self.worker(lambda: self._run_scan(run), done=self.scan_done)
                return
            self.log(f"{run.name} exists - this scan is {new}")
            self.scan_name.set(new)
            c["scan_name"] = new
            run = self._new_run(new)
        s = c["scan"]
        angles = scanmod.ordered(scanmod.angle_list(s["start"], s["stop"], s["step"]),
                                 s["order"])
        steps = scanmod.build_steps(angles, int(s["ref_every"]), s["ref_angle"])
        dm, bm = self.dark_mode.get(), self.bg_mode.get()
        pre = [k for k, m in (("dark", dm), ("background", bm)) if m == "measure"]
        steps = [{"kind": k, "target": 0.0} for k in pre] + steps
        plan = dict(s, preset=c["preset"], software=f"rampol {__version__}",
                    dark_mode=dm, bg_mode=bm)
        run.new(plan, steps, extra={"zero_deg": float(c["ell_zero_deg"]),
                                    "precheck": getattr(self, "last_check", None),
                                    "provenance": self._provenance(c)})
        self.run = run
        self.log(f"Scan {run.name}: {len(angles)} angles, {len(steps)} steps -> {run.folder}")
        reuse = [k for k, m in (("dark", dm), ("background", bm)) if m == "reuse latest"]

        def start():
            self.worker(lambda: self._run_scan(run), done=self.scan_done)

        def after_offsets():
            if reuse:
                self.borrow_latest(run, reuse, start)
            else:
                start()
        self.offsets_then(run, pre, after_offsets)

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
        n_dark = sum(1 for s in run.manifest["steps"] if s["kind"] in an.OFFSET_KINDS
                     and s.get("pd_scale", {}).get("vdiv") == vdiv)
        steps = []
        if not n_dark:
            steps.append({"kind": "background", "target": 0.0, "pd_scale": scale})
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
        go = lambda: self.worker(lambda: self._run_scan(run, {"null"}), done=self.scan_done)
        if not n_dark:
            self.offsets_then(run, ["background"], go)
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
        self._fill_compare_list(names)
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

    def analyse(self, path, correct_drift=True, opts=None):
        """Worker thread: everything it needs from the window is passed in
        (opts: the plot bar's Apply switches, read on the Tk thread)."""
        sg = self.load_sg()
        if sg is None:
            raise RuntimeError("Scope Grab is not loaded, so captures cannot be read")
        a = self.cfg["analysis"]
        opts = opts or {"sub_dark": True, "gains": True, "lock": True}
        cache = self.scan_cache.setdefault(os.path.normcase(os.path.abspath(path)), {})
        d = an.load_scan(path, sg.load_capture, trim=int(a["trim"]),
                         lock_tol=float(a.get("lock_tol", 0.0)) if opts["lock"] else 0.0,
                         cache=cache)
        d.subtract_dark = opts["sub_dark"]
        steps = d.manifest.get("steps", [])
        res = {"d": d, "path": path, "pol": None, "dips": [], "refine": [], "mon": None,
               "direct": [],
               "raw": an.scan_matrix(d, "scan", correct_drift),
               "n_done": sum(1 for x in steps if x.get("status") == "done"),
               "n_partial": sum(1 for x in steps if x.get("status") == "partial"
                                and x.get("files")),
               "n_total": sum(1 for x in steps if x.get("status") != "skipped")}
        try:
            pol = an.polarization(d, correct_drift=correct_drift,
                                  angle_gain=None if opts["gains"] else False)
        except ValueError:
            res["corr"] = an.corrections_summary(d, None)
            return res                    # fewer than 3 angles so far: traces only
        res["corr"] = an.corrections_summary(d, pol)
        res.update(pol=pol, dips=an.dip_er(pol, polarizer_er=a["polarizer_er"]),
                   refine=an.refine_result(d, pol, polarizer_er=a["polarizer_er"]),
                   mon=an.monitor_prediction(d, pol, a["deg_per_mon_v"]))
        try:
            # the same corrections as the fit (Apply switches): drift, and the
            # per-angle transmission, which otherwise puts its +-2 % into Imax
            res["direct"] = an.direct_er(d, pol, correct_drift=correct_drift,
                                         gains=pol.get("angle_gain"))
        except Exception as exc:              # no crossings, an odd record
            d.notes.append(f"direct ER not computed: {exc}")
        return res

    def live_refresh(self, folder):
        """Re-analyse a scan that is still being measured, in a thread of its
        own (the worker is busy with the instruments), and draw what there is.
        Only the steps finished since the last refresh are read. A refresh
        asked for while one runs is queued, not stacked."""
        if self.live_thread is not None and self.live_thread.is_alive():
            self.live_pending = folder
            return
        self.live_pending = None
        drift = bool(self.drift_on.get())
        opts = self._opts()
        name = os.path.basename(folder)
        if name not in self.scan_box["values"]:
            self.refresh_scan_list()
        out = self.outdir.get().strip() or self.cfg["outdir"]
        same = os.path.normcase(os.path.dirname(os.path.abspath(folder))) == \
            os.path.normcase(os.path.abspath(out))
        self.show_scan.set(name if same else folder)

        def bg():
            try:
                res = self.analyse(folder, drift, opts)
            except Exception as exc:
                res = None
                self.log(f"  live view: {exc}")
            self.call(self._live_done, res)
        self.live_thread = threading.Thread(target=bg, daemon=True)
        self.live_thread.start()

    def _live_done(self, res):
        if res is not None:
            self.result = res
            self.show_corrections()
            n = res["pol"]["n_angles"] if res["pol"] else len(res["raw"][0])
            part = " + 1 in progress" if res.get("n_partial") else ""
            self.plot_status.configure(
                text=f"live: {res['n_done']}{part} of {res['n_total']} steps, {n} analyzer "
                     f"angle(s)" + ("" if res["pol"] else " - the fit needs 3"),
                foreground="#060")
            self.mark_dirty()
        if self.live_pending:
            folder, self.live_pending = self.live_pending, None
            self.live_refresh(folder)

    def _run_scan(self, run, kinds=None):
        """Worker: measure, redrawing after every step. A stop or an error
        still returns, so the steps measured are loaded and shown."""
        start = getattr(run, "done_count", 0)
        try:
            return run.run(kinds=kinds,
                           on_step=lambda s: self.call(self.live_refresh, run.folder))
        except hw.Cancelled:
            n = getattr(run, "done_count", 0) - start
            self.log(f"Stopped after {n} step(s); showing what was measured. Start the "
                     f"scan again under the same name to resume it.")
            return n
        except Exception as exc:
            n = getattr(run, "done_count", 0) - start
            self.log(f"ERROR during the scan: {exc}")
            self.log(f"  {n} step(s) were measured and are shown; start the scan again "
                     f"under the same name to resume it.")
            return n

    def do_load_shown(self):
        path = self._scan_path(self.show_scan.get())
        if not path:
            return
        if self.busy:
            # a scan is using the worker: analyse in the background instead
            self.live_refresh(path)
            return
        drift = bool(self.drift_on.get())
        opts = self._opts()

        def go():
            return self.analyse(path, drift, opts)

        def done(res):
            self.result = res
            self.show_corrections()
            pol, d = res["pol"], res["d"]
            self.plot_status.configure(text=PLOT_HINT, foreground="#666")
            for n in d.notes:
                self.log(f"  {n}")
            if pol is None:
                self.log(f"Loaded {d.name}: {res['n_done']} of {res['n_total']} steps, "
                         f"{len(res['raw'][0])} analyzer angle(s) - traces only, the fit "
                         f"needs 3")
                self.mark_dirty()
                return
            self.log(f"Loaded {d.name}: {res['n_done']} of {res['n_total']} steps, "
                     f"{pol['n_angles']} angles, {len(d.t)} samples, "
                     f"rest azimuth {pol['psi_rest']:+.3f} deg, {len(res['dips'])} "
                     f"dip ER points, {len(res['refine'])} refined windows")
            if pol.get("dof", 9) < 2:
                self.log(f"  ! {pol['n_angles']} angles for a 3-term fit: no residual is "
                         f"left, so the error bars come from the shot scatter only "
                         f"({pol.get('err_source')}) and nothing checks the Malus model. "
                         f"Measure 6+ angles over 0-170 deg.")
            if pol.get("theta_span", 180) < 90:
                self.log(f"  ! the angles cover {pol['theta_span']:.0f} deg of the 180 deg "
                         f"Malus period: Imin and the ER from the fit are poorly "
                         f"determined. Spread them over 0-170 deg.")
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
            prov = d.manifest.get("provenance")
            if prov:
                self.log(f"  recorded with: {provenance.short(prov) or 'versions unknown'}")
            dr = [p for p in res.get("direct", []) if not p["lower"]
                  and not p.get("offset_limited")]
            if dr:
                lo = min(dr, key=lambda p: p["er"])
                self.log(f"  direct ER (both intensities measured): {len(res['direct'])} "
                         f"points, lowest {lo['er']:.1f} ({lo['kind']} {lo['seg']}, "
                         f"{lo['t_ms']:.3f} ms, rotation {lo['rotation']:+.1f} deg)")
            self.log_lab(res)
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

    # -- G and P: provenance, lab log, brief ------------------------------------------
    def _provenance(self, c):
        try:
            return provenance.collect(c)
        except Exception as exc:              # never stop a measurement over this
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _lab_upsert(self, outdir, row):
        try:
            lablog.upsert(outdir, row)
        except OSError as exc:
            self.log(f"  lab log not written ({exc}) - is it open in Excel? The row is "
                     f"written the next time")

    def log_lab(self, res):
        """The scan's row in the lab log, from the standard analysis only
        (every Apply switch and the drift correction on): a look with
        something switched off does not overwrite it."""
        if not all(self._opts().values()) or not self.drift_on.get():
            return
        self._lab_upsert(os.path.dirname(os.path.abspath(res["d"].folder)),
                         lablog.scan_row(an.scan_summary(res, res.get("direct"))))

    def open_lab_log(self):
        out = self.outdir.get().strip() or self.cfg["outdir"]
        p = lablog.path(out)
        if not os.path.exists(p):
            self.log(f"No lab log yet in {out}: a row is added when a scan is loaded "
                     f"with every Apply switch on, a bias run finishes or an angle is found.")
            return
        self.log(f"Lab log: {p}")
        if hasattr(os, "startfile"):
            os.startfile(p)

    BRIEF = (("traces", "Traces", "draw_traces", {},
              "Analyzer photodiode at every analyzer angle (dark / background "
              "subtracted, drift corrected), with the Trek monitors."),
             ("map", "Map", "draw_map", {"mode": MAP_MODES[0]},
              "Transmission against time and analyzer angle, normalised to Imax(t); "
              "the line is the fitted null."),
             ("map_residual", "Map residual", "draw_map", {"mode": MAP_MODES[2]},
              "Malus-fit residual in units of each step's standard error; right: "
              "rms per analyzer angle."),
             ("rotation", "Angle", "draw_angle", {},
              "Polarization rotation from rest, with the Trek monitors' prediction "
              "and the difference."),
             ("extinction", "Extinction", "draw_extinction", {},
              "Extinction ratio: ER_fit per sample, dips (fitted Imax), direct "
              "points (Imin and Imax both measured), null refine."),
             ("poincare", "Poincaré", "draw_poincare", {},
              "Linear Stokes parameters in the rest frame; |S3| and the ellipticity "
              "assume full polarization, handedness not measured."),
             ("corrections", "Corrections", "draw_corrections", {},
              "Subtracted offsets, reference drift, per-angle transmission, shots "
              "kept and dropped."),
             ("diagnostics", "Diagnostics", "draw_diagnostics", {},
              "Reference returns, out-of-model harmonics, residual against noise, "
              "analyzer landing."))

    def do_export_brief(self):
        """The standard figure set and the key numbers of the shown scan into
        <scan>/analysis/brief/: PNGs at print size, summary.json, summary.md.
        Drawn with the window's current settings (Apply switches, smoothing,
        cursor)."""
        res = self.result
        if not res or res.get("pol") is None:
            self.log("Export brief: load a scan with a fit first.")
            return
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        d = res["d"]
        out = os.path.join(d.folder, "analysis", "brief")
        os.makedirs(out, exist_ok=True)
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        made = []
        try:
            for stem, title, meth, kw, caption in self.BRIEF:
                fig = Figure(figsize=(10, 6.5), dpi=150, constrained_layout=True)
                FigureCanvasAgg(fig)
                try:
                    getattr(self, meth)(fig, **kw)
                except Exception as exc:
                    self.log(f"  brief: {title} not drawn ({exc})")
                    continue
                fn = f"{d.name}_{stem}.png"
                fig.savefig(os.path.join(out, fn))
                made.append((fn, title, caption))
            summ = an.scan_summary(res, res.get("direct"))
            summ["options"] = dict(self._opts(), drift=bool(self.drift_on.get()),
                                   smooth_us=self.smooth_samples(d.t)[0])
            summ["direct_er"] = res.get("direct", [])
            summ["figures"] = [m[0] for m in made]
            with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summ, fh, indent=1, default=_jsonable)
            with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
                fh.write(_brief_md(summ, made))
        finally:
            self.root.configure(cursor="")
        self.log(f"Brief: {len(made)} figures, summary.json and summary.md -> {out}")
        if hasattr(os, "startfile"):
            try:
                os.startfile(out)
            except OSError:
                pass

    # -- H: compare scans ----------------------------------------------------------------
    def build_compare_tab(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="Compare")
        side = ttk.Frame(frame)
        side.pack(side="left", fill="y", padx=(4, 2), pady=4)
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(side, text="Scans to set against the shown one (ctrl/shift-click, up "
                             "to 6)", wraplength=230, justify="left").pack(anchor="w")
        lf = ttk.Frame(side)
        lf.pack(fill="y", expand=True)
        self.cmp_lb = tk.Listbox(lf, selectmode="extended", width=30, height=16,
                                 exportselection=False, font=("Consolas", 8))
        sb = ttk.Scrollbar(lf, command=self.cmp_lb.yview)
        self.cmp_lb.configure(yscrollcommand=sb.set)
        self.cmp_lb.pack(side="left", fill="y", expand=True)
        sb.pack(side="left", fill="y")
        r = ttk.Frame(side)
        r.pack(fill="x", pady=(4, 0))
        self._btn(r, "Compare selected", self.do_compare_load)
        ttk.Button(r, text="Clear", command=self.do_compare_clear).pack(side="left", padx=4)
        r = ttk.Frame(side)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="ER against").pack(side="left")
        self.cmp_x = tk.StringVar(value="time")
        cb = ttk.Combobox(r, textvariable=self.cmp_x, values=("time", "rotation"), width=9,
                          state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw(self.fig_cmp))
        self.cmp_diff = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="difference from the shown scan", variable=self.cmp_diff,
                        command=lambda: self.redraw(self.fig_cmp)).pack(anchor="w")
        ttk.Label(side, foreground="#666", justify="left", wraplength=230, text=(
            "Each scan with the window's Apply switches. Rotation is from each "
            "scan's own rest azimuth; the difference is smoothed by the Smooth "
            "box.")).pack(anchor="w", pady=(4, 0))
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=right)
        toolbar = NavigationToolbar2Tk(canvas, right)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._canvas, fig._toolbar, fig._ctl, fig._frame = canvas, toolbar, side, frame
        self.plot_tabs[frame] = (fig, self.draw_compare)
        self.plot_dirty.add(frame)
        self.fig_cmp = fig

    def _fill_compare_list(self, names):
        keep = {self.cmp_lb.get(i) for i in self.cmp_lb.curselection()}
        self.cmp_lb.delete(0, "end")
        for i, n in enumerate(names):
            self.cmp_lb.insert("end", n)
            if n in keep:
                self.cmp_lb.selection_set(i)

    def _cmp_key(self):
        return (bool(self.drift_on.get()), tuple(sorted(self._opts().items())))

    def do_compare_load(self):
        names = [self.cmp_lb.get(i) for i in self.cmp_lb.curselection()]
        if not names:
            self.log("Compare: pick one or more scans in the list first.")
            return
        if len(names) > 6:
            self.log(f"Compare: {len(names)} picked - the first 6 are used.")
            names = names[:6]
        paths = [os.path.normcase(os.path.abspath(self._scan_path(n))) for n in names]
        key = self._cmp_key()
        drift, opts = key[0], dict(key[1])
        todo = [p for p in paths if (p, key) not in self.cmp_results]

        def go():
            out = {}
            for p in todo:
                if self.stop_flag.is_set():
                    raise hw.Cancelled()
                self.log(f"Compare: analysing {os.path.basename(p)}...")
                out[p] = self.analyse(p, drift, opts)
            return out

        def done(out):
            for p, r in out.items():
                self.cmp_results[(p, key)] = r
                if r.get("pol") is None:
                    self.log(f"Compare: {os.path.basename(p)} has {len(r['raw'][0])} "
                             f"analyzer angle(s) - no fit, not drawn")
            self.cmp_sel = [(p, key) for p in paths]
            # hold only what is shown: each analysis keeps every step's traces
            self.cmp_results = {k: v for k, v in self.cmp_results.items() if k in self.cmp_sel}
            self.nb.select(self.fig_cmp._frame)
            self.redraw(self.fig_cmp)
        self.worker(go, done=done)

    def do_compare_clear(self):
        self.cmp_sel, self.cmp_results = [], {}
        self.cmp_lb.selection_clear(0, "end")
        self.redraw(self.fig_cmp)

    def draw_compare(self, fig):
        items = []
        shown = self.result if self.result and self.result.get("pol") is not None else None
        here = os.path.normcase(os.path.abspath(shown["d"].folder)) if shown else None
        if shown:
            items.append((shown, True))
        for k in self.cmp_sel:
            r = self.cmp_results.get(k)
            if r is None or r.get("pol") is None:
                continue
            if here and os.path.normcase(os.path.abspath(r["d"].folder)) == here:
                continue
            items.append((r, False))
        if not items:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Load a scan (Plot data: Scan), pick scans in the list and "
                              "press Compare selected", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_axis_off()
            return
        by_rot = self.cmp_x.get() == "rotation"
        diff_on = bool(self.cmp_diff.get()) and shown is not None and len(items) > 1
        gs = fig.add_gridspec(3 if diff_on else 2, 1,
                              height_ratios=[1.2, 1, 1.3] if diff_on else [1.2, 1.3])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1) if diff_on else None
        ax3 = fig.add_subplot(gs[-1], sharex=None if by_rot else ax1)
        cols = matplotlib.colormaps["tab10"]
        us = 0
        for i, (r, is_shown) in enumerate(items):
            pol, d = r["pol"], r["d"]
            c = "k" if is_shown else cols(i % 10)
            lab = d.name + (" (shown)" if is_shown else "")
            tt, rr = self._decimate(d.t, pol["rotation"], 3000)
            ax1.plot(tt * 1e3, rr, color=c, lw=1.0 if is_shown else 0.8, label=lab)
            if diff_on and not is_shown:
                tr = shown["d"].t
                m = (tr >= d.t[0]) & (tr <= d.t[-1])
                if m.sum() > 10:
                    dd = (np.interp(tr[m], d.t, pol["rotation"])
                          - shown["pol"]["rotation"][m]) * 1e3
                    tt, dd = self._decimate(tr[m], self.smooth(dd, tr[m]), 3000)
                    ax2.plot(tt * 1e3, dd, color=c, lw=0.8, label=f"{d.name} - shown")
            er, ok, us = self.smoothed_er(pol, d.t)
            x = pol["rotation"] if by_rot else d.t * 1e3
            k = max(1, len(x) // 4000)
            ax3.plot(x[::k], np.where(ok, er, np.nan)[::k], color=c, lw=0.6, alpha=0.6)
            cr = [p for p in r.get("direct", []) if p["kind"] == "crossing"]
            for lower, mk in ((False, "o"), (True, "^")):
                sel = [p for p in cr if p["lower"] == lower]
                if sel:
                    ax3.plot([p["rotation"] if by_rot else p["t_ms"] for p in sel],
                             [p["er"] for p in sel], mk, ms=3.5, color=c,
                             mfc=c if not lower else "none", ls="none")
        ax1.set_ylabel("rotation from rest (deg)")
        ax1.set_title("Polarization rotation, each scan from its own rest azimuth", fontsize=9)
        ax1.legend(fontsize=7, loc="best")
        ax1.grid(alpha=0.3)
        if diff_on:
            sig = shown["pol"]["sig_psi"] * 1e3
            t3 = shown["d"].t * 1e3
            kk = max(1, len(t3) // 4000)
            ax2.fill_between(t3[::kk], -sig[::kk], sig[::kk], color="0.85", lw=0,
                             label="+-1 SD per sample, shown scan's fit")
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_ylabel("difference (mdeg)")
            ax2.legend(fontsize=7, loc="best")
            ax2.grid(alpha=0.3)
            ax1.tick_params(labelbottom=False)
        ax3.set_yscale("log")
        ax3.set_xlabel("rotation from rest (deg)" if by_rot else "time (ms)")
        if not by_rot:
            ax3.set_xlabel("time (ms)")
            if diff_on:
                ax2.tick_params(labelbottom=False)
        ax3.set_ylabel("extinction ratio")
        ax3.set_title(f"Lines: ER_fit ({us:g} us mean); dots: measured directly at the "
                      f"crossings (triangles: lower bounds)" if us else
                      "Lines: ER_fit per sample; dots: measured directly at the crossings "
                      "(triangles: lower bounds)", fontsize=8)
        ax3.grid(alpha=0.3, which="both")

    # -- K: the polarization state -------------------------------------------------------
    def draw_poincare(self, fig):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t
        st = an.stokes(pol, self.smooth(pol["a0"], t), self.smooth(pol["c2"], t),
                       self.smooth(pol["s2"], t))
        er, ok, us = self.smoothed_er(pol, t)
        chi = np.rad2deg(np.arctan(1.0 / np.sqrt(np.maximum(er, 1.0))))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1.1])
        ax3 = fig.add_subplot(gs[:, 0], projection="3d")
        u, v = np.mgrid[0:2 * np.pi:37j, 0:np.pi / 2:7j]
        ax3.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                           color="0.85", lw=0.4)
        idx = np.linspace(0, len(t) - 1, min(len(t), 4000)).astype(int)
        sc = ax3.scatter(st["s1"][idx], st["s2"][idx], st["s3"][idx], c=t[idx] * 1e3,
                         cmap="viridis", s=2, depthshade=False)
        rest = np.flatnonzero(an.rest_index(t))
        j0 = int(rest[len(rest) // 2]) if len(rest) else 0
        ax3.scatter([st["s1"][j0]], [st["s2"][j0]], [st["s3"][j0]], color="k", s=30,
                    marker="s", label="rest")
        if ok.any():
            jw = int(np.flatnonzero(ok)[np.argmax(chi[ok])])
        else:
            jw = int(np.argmax(chi))
        j = (int(np.argmin(np.abs(t - self.cursor_t))) if self.cursor_t is not None else jw)
        ax3.scatter([st["s1"][j]], [st["s2"][j]], [st["s3"][j]], color="#d62728", s=40,
                    label=f"{'cursor' if self.cursor_t is not None else 'largest chi'}, "
                          f"{t[j]*1e3:.3f} ms")
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(0, 1)
        try:
            ax3.set_box_aspect((1, 1, 0.5))
        except AttributeError:
            pass
        ax3.view_init(elev=30, azim=-60)
        ax3.set_xlabel("S1 / S0", fontsize=7)
        ax3.set_ylabel("S2 / S0", fontsize=7)
        ax3.set_zlabel("|S3| / S0", fontsize=7)
        ax3.legend(fontsize=6, loc="upper left")
        ax3.set_title(f"Poincaré sphere, rest frame ({d.name})", fontsize=9)
        fig.colorbar(sc, ax=ax3, shrink=0.5, pad=0.02, orientation="horizontal",
                     label="time (ms)")
        ax = fig.add_subplot(gs[0, 1])
        tt = t * 1e3
        ax.plot(tt, np.where(ok, chi, np.nan), lw=0.7, color="#1f77b4",
                label=f"chi ({us:g} us mean)" if us else "chi (per sample)")
        ax.plot(tt, np.where(~ok, chi, np.nan), lw=0.5, alpha=0.35, color="#1f77b4",
                label="upper bound (Imin < 2 sigma)")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("ellipticity angle chi (deg)")
        ax.set_title("If fully polarized: tan chi = sqrt(Imin / Imax)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
        self._cursor(ax)
        self._poin_tax = ax
        axe = fig.add_subplot(gs[1, 1])
        ph = np.linspace(0, 2 * np.pi, 200)
        for jj, col, ls, lab in ((j0, "0.5", "--", "rest"),
                                 (j, "#d62728", "-", f"{t[j]*1e3:.3f} ms")):
            c_ = np.deg2rad(chi[jj])
            az = np.deg2rad(st["azimuth_deg"][jj] - st["azimuth_deg"][j0])
            x = np.cos(c_) * np.cos(ph)
            y = np.sin(c_) * np.sin(ph)
            axe.plot(x * np.cos(az) - y * np.sin(az), x * np.sin(az) + y * np.cos(az),
                     color=col, ls=ls, lw=1.0, label=lab)
            axe.plot([-np.cos(az), np.cos(az)], [-np.sin(az), np.sin(az)], color=col,
                     lw=0.5, ls=":")
        axe.set_aspect("equal")
        axe.set_xlim(-1.1, 1.1)
        axe.set_ylim(-1.1, 1.1)
        axe.axhline(0, color="k", lw=0.4)
        axe.axvline(0, color="k", lw=0.4)
        az_j = st["azimuth_deg"][j] - st["azimuth_deg"][j0]
        axe.set_title(f"{t[j]*1e3:.3f} ms: azimuth {az_j:+.2f} deg\n"
                      f"chi {'<' if not ok[j] else ''}{chi[j]:.2f} deg, ER_fit "
                      f"{'>' if not ok[j] else ''}{er[j]:.0f}", fontsize=8)
        axe.set_xlabel("rest polarization direction")
        axe.legend(fontsize=6, loc="lower right")

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
        if fig._frame in getattr(self, "free_tabs", ()):
            try:
                draw(fig)
            except Exception as exc:
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(0.02, 0.5, f"Could not draw: {exc}", transform=ax.transAxes)
                ax.set_axis_off()
                self.log(f"draw: {exc}")
        elif not self.result:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No scan loaded", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_axis_off()
        elif self.result["pol"] is None and draw != self.draw_traces:  # == on bound methods; "is" is always False
            ax = fig.add_subplot(111)
            n = len(self.result["raw"][0])
            ax.text(0.5, 0.5, f"{n} analyzer angle(s) measured so far - this tab needs "
                              f"the fit, which needs 3.\nThe Traces tab shows them now.",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
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
        if fig is self.fig_poin and ev.inaxes is not getattr(self, "_poin_tax", None):
            return
        self.cursor_t = ev.xdata * 1e-3
        self.cursor_var.set(f"{ev.xdata:.3f}")
        self.plot_dirty |= {self.fig_malus._frame, self.fig_map._frame,
                            self.fig_angle._frame, self.fig_ext._frame,
                            self.fig_poin._frame}
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
        d = res["d"]
        theta, I, _sem, steps = res["raw"]
        pol = res["pol"] or {"theta": theta, "I": I, "steps": steps,
                             "n_angles": len(theta)}
        t = d.t * 1e3
        has_mon = any(r in d.roles for r in ("MonX1", "MonX2", "CmdX1", "CmdX2"))
        ax = fig.add_subplot(211 if has_mon else 111)
        cmap = matplotlib.colormaps[ANGLE_CMAP]
        for th, y in zip(pol["theta"], pol["I"]):
            ax.plot(t, y, lw=0.6, color=cmap((th % 360) / 360))
        for s in d.steps:
            if s["kind"] in an.OFFSET_KINDS and "PD" in s["v"]:
                ax.plot(t, s["v"]["PD"], lw=0.6, color="k", ls="--", label=s["kind"])
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 360))
        ax.set_ylabel("PD - dark, drift corrected (V)")
        ax.set_title(f"Analyzer photodiode at {pol['n_angles']} analyzer angles ({d.name})")
        ax.grid(alpha=0.3)
        if has_mon:
            ax.tick_params(labelbottom=False)
            ax2 = fig.add_subplot(212, sharex=ax)
            names = cfgmod.ROLE_NAMES
            for r, col, ls in (("MonX1", "#1f77b4", "-"), ("MonX2", "#2ca02c", "-"),
                               ("CmdX1", "#1f77b4", "--"), ("CmdX2", "#2ca02c", "--")):
                if r in d.roles:
                    src = [s for s in (pol["steps"] or d.steps) if r in s["v"]]
                    if not src:
                        continue
                    v = np.mean([s["v"][r] for s in src], axis=0)
                    ax2.plot(t, v, lw=0.8, color=col, ls=ls,
                             label=f"CH{d.roles[r]} {names[r]}")
            ax2.set_ylabel("monitor (V, 1 V/kV) and command (V)")
            ax2.legend(loc="upper right", fontsize=7)
            ax2.grid(alpha=0.3)
            ax2.set_xlabel("time (ms)")
            fig.colorbar(sm, ax=[ax, ax2], label="analyzer angle (deg)")
        else:
            ax.set_xlabel("time (ms)")
            fig.colorbar(sm, ax=ax, label="analyzer angle (deg)")

    def draw_map(self, fig, mode=None):
        res = self.result
        pol, d = res["pol"], res["d"]
        mode = mode or self.map_show.get()
        t = d.t * 1e3
        order = np.argsort(wrap_angle(pol["theta"]))
        th = wrap_angle(pol["theta"][order])
        edges = np.concatenate([[th[0] - (th[1] - th[0]) / 2],
                                (th[1:] + th[:-1]) / 2,
                                [th[-1] + (th[-1] - th[-2]) / 2]]) if len(th) > 1 else [th[0] - 1, th[0] + 1]
        if mode == MAP_MODES[0]:
            ax = fig.add_subplot(111)
            I = pol["I"][order] / np.maximum(pol["imax"], 1e-9)
            tm = np.concatenate([[t[0]], (t[1:] + t[:-1]) / 2, [t[-1]]])
            mesh = ax.pcolormesh(tm, edges, I, cmap="magma", shading="flat", rasterized=True,
                                 vmin=0, vmax=1)
            fig.colorbar(mesh, ax=ax, label="I / Imax(t)")
            line = "#4fc3f7"
            ax.set_title(f"Transmission vs time and analyzer angle; line: fitted null "
                         f"psi + 90 deg ({d.name})")
        else:
            gs = fig.add_gridspec(1, 2, width_ratios=[7, 1])
            ax = fig.add_subplot(gs[0])
            axr = fig.add_subplot(gs[1], sharey=ax)
            _, resid, z = an.malus_residual(pol)
            mv = mode == MAP_MODES[1]
            M = (resid * 1e3 if mv else z)[order]
            # block means over ~3000 columns: drawable, and a systematic
            # pattern stands out of the noise at its per-sample size
            n = max(1, len(t) // 3000)
            m = (len(t) // n) * n
            Mb = np.nanmean(M[:, :m].reshape(len(th), -1, n), axis=2)
            tb = t[:m].reshape(-1, n)
            tm = np.r_[tb[:, 0], tb[-1, -1]]
            # a robust scale: on test-4 the record past the lock switching off
            # had 40 mV residuals and a percentile scale washed out the rest
            lim = 6.0 * float(np.nanmedian(np.abs(Mb)))
            lim = lim if np.isfinite(lim) and lim > 0 else 1.0
            if not mv:
                lim = max(lim, 3.0)
            mesh = ax.pcolormesh(tm, edges, Mb, cmap="RdBu_r", shading="flat",
                                 rasterized=True, vmin=-lim, vmax=lim)
            fig.colorbar(mesh, ax=axr, extend="both",
                         label=("I - fit (mV)" if mv else "(I - fit) / standard error"))
            rms = np.sqrt(np.nanmean(np.square(M), axis=1))
            axr.barh(th, rms, height=0.8 * np.min(np.diff(edges)), color="#1f77b4")
            axr.set_xlabel("rms (mV)" if mv else "rms")
            if not mv:
                axr.axvline(1, color="k", lw=0.6, ls="--")
            axr.tick_params(labelleft=False)
            axr.grid(alpha=0.3, axis="x")
            line = "k"
            ax.set_title(f"Residual to a0 + B cos 2(theta - psi), {d.name}"
                         + (f" ({n}-sample block means)" if n > 1 else ""), fontsize=9)
        for k in range(-1, 4):
            ax.plot(t, pol["psi_u"] + 90 + 180 * k, color=line, lw=0.7,
                    alpha=1.0 if mode == MAP_MODES[0] else 0.4)
        ax.set_ylim(edges[0], edges[-1])
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("analyzer angle (deg)")
        self._cursor(ax)

    def draw_malus(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t
        j = int(np.argmin(np.abs(t - (self.cursor_t if self.cursor_t is not None else t[len(t) // 2]))))
        th = pol["theta"]
        y = pol["I"][:, j]
        ax = fig.add_subplot(211)
        ax.plot(wrap_angle(th), y * 1e3, "o", ms=4, color="#1f77b4", label="measured")
        g = np.linspace(-5, 355, 721)
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
        ax2.plot(wrap_angle(th), (y - fit_at) * 1e3, "o", ms=3, color="#1f77b4")
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
        ax.set_ylabel("rotation from rest (deg)")
        ax.set_title(f"Polarization rotation vs time (rest azimuth {pol['psi_rest']:+.3f} deg "
                     f"in the analyzer frame)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        self._cursor(ax)
        ax2 = fig.add_subplot(212, sharex=ax)
        if res["mon"] is not None:
            ax2.plot(t, res["mon"][1] * 1e3, color="#2ca02c", lw=0.7, label="measured - monitor prediction")
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
        show = {k: v.get() for k, v in self.ext_show.items()}
        er, ok, us = self.smoothed_er(pol, d.t)
        if show["fit"]:
            lbl = f"ER_fit ({us:g} us mean)" if us else "ER_fit (per sample)"
            ax.plot(x, np.where(ok, er, np.nan), color="#1f77b4", lw=0.7, label=lbl)
            ax.plot(x, np.where(~ok, er, np.nan), color="#1f77b4", lw=0.5, alpha=0.35,
                    label="ER_fit lower bound (Imin < 2 sigma)")
        dips = res["dips"] if show["dips"] else []
        if dips:
            for sign, mk, lbl in ((1, "^", "dip, rising"), (-1, "v", "dip, falling")):
                pts = [p for p in dips if np.sign(p["rate"]) == sign]
                if not pts:
                    continue
                xx = [p["rotation"] if by_rot else p["t"] * 1e3 for p in pts]
                ax.plot(xx, [p["er"] for p in pts], mk, ms=5,
                        color="#d62728", ls="none", label=f"{lbl} ({len(pts)})")
        direct = res.get("direct", []) if show["direct"] else []
        for kind, lower, mk, col, lbl in (
                ("crossing", False, "o", "k", "direct: crossing (Imin, and Imax at +90 deg, measured)"),
                ("crossing", True, "^", "k", "direct: crossing, lower bound"),
                ("static", False, "D", "#2ca02c", "direct: static, angle nearest crossed"),
                ("static", True, "^", "#2ca02c", "direct: static, lower bound"),
                ("offset", False, "D", "#2ca02c",
                 "direct: static, angle too far from crossed (offset-limited)")):
            if kind == "offset":
                pts = [p for p in direct if p.get("offset_limited")]
                lower = True                      # drawn hollow
            else:
                pts = [p for p in direct if p["kind"] == kind and p["lower"] == lower
                       and not p.get("offset_limited")]
            if pts:
                ax.plot([p["rotation"] if by_rot else p["t_ms"] for p in pts],
                        [p["er"] for p in pts], mk, ms=4, color=col,
                        mfc=col if not lower else "none", ls="none",
                        label=f"{lbl} ({len(pts)})")
        for r in (res["refine"] if show["refine"] else []):
            if "er" in r:
                m = (d.t >= r["t0"]) & (d.t <= r["t1"])
                xx = float(np.mean(pol["rotation"][m])) if by_rot else 0.5 * (r["t0"] + r["t1"]) * 1e3
                ax.plot([xx], [r["er"]], "D", ms=6, color="#9467bd",
                        label=f"null refine: {r['label']}")
        dr = pol.get("drift_resid")
        if dr and show["fit"]:
            ax.axhline(1 / dr, color="#1f77b4", lw=0.8, ls=":",
                       label=f"1 / ref leave-one-out scatter ({1 / dr:.0f})")
        ax.set_yscale("log")
        lim = self.cfg["analysis"]["polarizer_er"]
        top = ax.get_ylim()[1]
        if lim <= 10 * top:
            ax.axhline(lim, color="k", lw=0.8, ls="--", label=f"analyzer's own ER ({lim:.1e})")
        else:
            ax.text(0.99, 0.01, f"analyzer's own ER at 843 nm: {lim:.1e}, off scale",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#666")
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
        if pol.get("angle_gain") is not None:
            g = pol["angle_gain"]
            order = np.argsort(wrap_angle(pol["theta"]))
            ax.plot(wrap_angle(pol["theta"])[order], (g[order] - 1) * 100, "o-", ms=3)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel("analyzer angle (deg)")
            ax.set_ylabel("transmission - mean (%)")
            ax.set_title("Per-angle transmission (fitted, divided out)")
            ax.grid(alpha=0.3)
        elif "c1" in pol:
            B = np.maximum(pol["B"], 1e-9)
            sm = lambda y: self.smooth(y, d.t)
            # smooth the components, then take the amplitude: noise alone
            # averages toward zero instead of toward its rms
            ax.plot(t, np.hypot(sm(pol["c1"]), sm(pol["s1"])) / B * 1e3, lw=0.6, label="1-theta / B")
            ax.plot(t, np.hypot(sm(pol["c4"]), sm(pol["s4"])) / B * 1e3, lw=0.6, label="4-theta / B")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "needs >= 8 angles over >= 150 deg", ha="center",
                    transform=ax.transAxes, color="#888")
        if pol.get("angle_gain") is None:
            ax.set_ylabel("relative amplitude (1e-3)")
            ax.set_xlabel("time (ms)")
            ax.set_title("Harmonics outside the Malus law")
            ax.grid(alpha=0.3)
        ax = fig.add_subplot(223)
        sem_med = np.nanmedian([np.nanmedian(s["sem"]["PD"]) for s in pol["steps"]])
        src = pol.get("err_source", "residual")
        ax.plot(t, pol["rms"] * 1e3, lw=0.6,
                label="fit residual rms" if src == "residual"
                else f"error scale from {src}")
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
        if pol is None:
            return

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
                "dip", f(p["t"] * 1e3, ".3f"), f(p["rotation"], ".2f"), f(float(wrap_angle(p["theta"])), ".2f"),
                f(p["rate"], ".4f"), ("> " if p["er_lower"] else "") + f(p["er"], ".0f"),
                f(p["er_light"], ".0f"), f(p["imin"] * 1e3, ".3f"), f(p["sig_imin"] * 1e3, ".3f"),
                "", "", f"{p['n']} samples"))
        for p in self.result.get("direct", []):
            self.tv.insert("", "end", values=(
                f"direct {p['kind']}", f(p["t_ms"], ".3f"), f(p["rotation"], ".2f"),
                f(float(wrap_angle(p["theta"])), ".2f"), f(p["rate"] * 1e-3, ".4f"),
                ("> " if p["lower"] else "") + f(p["er"], ".1f"), "",
                f(p["imin_mV"], ".3f"), f(p["sig_mV"], ".3f"), "", "",
                f"{p['seg']}; Imax {p['imax_V']:.4f} V measured at +90 deg"
                + (f"; {p['off_deg']:+.2f} deg from crossed" if "off_deg" in p else "")
                + (" (offset-limited)" if p.get("offset_limited") else "")))

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


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _brief_md(summ, figures):
    """summary.md: the numbers and the figures, factual captions only."""
    L = [f"# {summ['name']}", ""]
    L.append(f"- Measured: {summ.get('created', '')} (manifest updated {summ.get('updated', '')})")
    L.append(f"- Steps: {summ.get('steps_done')} of {summ.get('steps_total')}; "
             f"{summ.get('n_angles', '?')} analyzer angles; {summ.get('shots_per_angle', '?')} "
             f"shots per angle; PD at {summ.get('pd_vdiv')} V/div")
    L.append(f"- Corrections: {summ.get('corrections', '')}")
    o = summ.get("options", {})
    L.append(f"- Analysis switches: subtract dark/background {o.get('sub_dark')}, per-angle "
             f"transmission {o.get('gains')}, drop missed-lock shots {o.get('lock')}, drift "
             f"{o.get('drift')}, smoothing {o.get('smooth_us')} us")
    prov = summ.get("provenance")
    if prov:
        L.append(f"- Recorded with: {provenance.short(prov) or 'versions unknown'}")
    L += ["", "## Numbers", ""]
    if "psi_rest_deg" in summ:
        L.append(f"- Rest azimuth {summ['psi_rest_deg']:+.3f} deg (analyzer frame); rotation "
                 f"{summ['rotation_min_deg']:+.2f} to {summ['rotation_max_deg']:+.2f} deg")
    lo = summ.get("direct_er_min")
    if lo:
        L.append(f"- Lowest directly measured ER {lo['er']:.1f}: {lo['kind']} {lo['seg']}, "
                 f"t {lo['t_ms']:.3f} ms, rotation {lo['rotation']:+.1f} deg, analyzer "
                 f"{lo['theta']:.2f} deg, Imin {lo['imin_mV']:.1f} mV, Imax {lo['imax_V']:.3f} V")
        L.append(f"- Direct points: {summ.get('direct_er_n')} "
                 f"({summ.get('direct_er_lower_bounds')} lower bounds)")
    if summ.get("drift_resid"):
        L.append(f"- Reference returns predict each other to {summ['drift_resid']*1e3:.2f}e-3 "
                 f"(ER_fit drift-limited above ~{1/summ['drift_resid']:.0f})")
    if summ.get("segments"):
        L += ["", "| segment | t (ms) | rotation (deg) | ER_fit median |", "|---|---|---|---|"]
        for sg in summ["segments"]:
            L.append(f"| {sg['kind']} | {sg['t0_ms']:.2f} - {sg['t1_ms']:.2f} | "
                     f"{sg['rotation_deg']:+.2f} | {sg['er_fit_median']:.0f} |")
    L += ["", "## Figures", ""]
    for fn, title, cap in figures:
        L.append(f"**{title}** - {cap}")
        L.append("")
        L.append(f"![{title}]({fn})")
        L.append("")
    return "\n".join(L) + "\n"


def wrap_angle(a):
    """Analyzer angles onto [-5, 355) deg for plotting. Plain % 360 sent a
    0 deg step that landed at -0.01 to 359.99 - the far end of the axis - and
    the Map stretched one band across it (5 Oct 2026, test-2)."""
    return (np.asarray(a, float) + 5.0) % 360.0 - 5.0


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

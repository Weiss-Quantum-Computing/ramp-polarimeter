"""The Ramp Polarimeter window.

Controls on the left (hardware, analyzer, channel roles, then the measurement
modes as tabs: ramp scan, analyzer (find angle, null refine), fixed rotations
(the code's 'bias points'), ILC target),
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
from .widgets import CopyLabel
from . import awg as awgmod
from . import calib

ANGLE_CMAP = "hsv"                 # cyclic: 0 and 360 deg share a colour, none near white
AWG_HELP = (
    "AWG tab. The 4063B plays on the bench trigger (EXT burst), CH1 -> Trek X1 -> EO1, "
    "CH2 -> Trek X2 -> EO2. 'ramp': idle -> the rotation (split between the crystals, "
    "volts from the EOM calibration) -> idle, both ends exactly at idle; idle blank = the "
    "ILC state files' first sample (the learned trim; file zero parks the EOMs at -9 / "
    "-41 V). The record is lead + rise + hold + fall + after (the defaults: the ILC's 11 "
    "ms at 2 us); a different length is set up through idle with the outputs live, and "
    "Park goes back to 11 ms. The scope's span is set apart: from 'before' ms before the "
    "trigger to 'after' ms past the record. 'ILC drives': two "
    "drive_<stem>_iNN.csv, checked against their own state's target. "
    "ORDER: Preview (draws it, checks the Trek limits, length, trigger period, duty, "
    "idle cap) -> Dry run on scope (AWG outputs teed to two scope channels, Treks not "
    "driving the EOMs: each output alone, then both, compared with what was meant - "
    "cabling, gain, delay, time scale, shape, idle, triggering) -> reconnect the Treks "
    "-> Outputs ON (or Load to AWG for another dry-run waveform) -> Find min/max in the "
    "hold, or a ramp scan in the Ramp scan tab. With 'require a dry run' nothing that has "
    "not passed one this session reaches live outputs. NEVER FLOAT (both outputs): the "
    "program never switches an output off - changes are made live and the end of anything "
    "(close, Disconnect, a fixed-rotation run) is Park: an idle waveform with the outputs ON, because "
    "the FPGA/buffer stage drives high (-4 to -5.7 kV) on a floating input. Outputs OFF "
    "then asks first.")
ER_HELP = """Every point is Imax / Imin of the light at one moment: how far from perfectly linear the light is. They differ only in HOW Imin and Imax are obtained.

MEASURED AT A CROSSING (circles)
During a ramp the polarization sweeps past the crossed position of each analyzer angle the scan used. At that instant:
  Imin = that analyzer angle's own trace at its lowest point (4 us average),
  Imax = the trace of the analyzer angle 90 deg away, at the same instant.
Both numbers are read straight off the photodiode; the Malus fit only says WHEN the crossing happens. The most direct number there is.

DIP FIT (triangles)
The same crossings, analysed differently: the dip I(t) = Imin + Imax sin^2(psi(t) - theta) is fitted over +-8 deg around the crossing, with psi(t) and Imax taken from the per-sample Malus fit. It uses many samples, so it is less noisy, but it leans on the fit. The older plot's "dip, rising" / "dip, falling" were these points split by which way the rotation was moving. Circles and triangles at the same time should agree; where they do not, trust the circle.

MEASURED, STATIC (squares)
At rest, in the holds and after the ramp nothing sweeps through crossed, so the scan angle nearest crossed is used: Imin = its trace averaged over the stretch, Imax = its 90-deg partner. If that angle sat a few degrees off crossed, the offset alone puts Imax sin^2(offset) into Imin: hollow grey squares are those, where the offset explains over half of Imin - they say how far the angle was, not what the light is.

NULL REFINE (diamonds)
The static stretches again, measured properly: extra analyzer angles stepped within a few degrees of crossed, read at a sensitive V/div (Analyzer tab, Refine), Imin fitted. The best static number.

PER-SAMPLE MALUS FIT (grey line, "ER_fit")
At every time sample, all analyzer angles fitted to a0 + B cos 2(theta - psi): Imax = a0 + B, Imin = a0 - B. It needs the light level to be identical for every angle, so intensity drift between angles limits it: the dotted line is 1 / (how well the reference returns predict each other). Above that line ER_fit means nothing - which is why the other methods exist.

COLOURS, FILL, ARROWS, BARS
  Colour = which transport: leg 1 (the first motion) blue, leg 2 (the second) orange. The time axis is shaded per leg.
  Filled = on a ramp out from rest; hollow = on the ramp back; half-filled = the rotation standing still (rest, holds, after - the static and null-refine points, and dip fits that land in a still stretch). Which one is taken from the segment the point is in (the Table's segments).
  A marker with a dotted line going up from it = a lower bound: Imin is under 2 sigma of its own noise, so the ER is at least Imax / (2 sigma) - the marker sits at that bound and the true value is somewhere above. A method with smaller noise gives a higher bound for the same light: where the dip fit's triangles sit far above the measured circles with dotted lines on both, neither has resolved Imin - they agree that it is near zero. 'lower bounds' hides them.
  Error bars = +-1 sigma from the statistical errors of Imin (and of Imax for the measured points); asymmetric, because ER goes as 1 / Imin. They do not include systematic errors (the dark / background subtraction, the analyzer's per-angle transmission, the scope's gain).

x axis "rotation" puts both legs and both directions on one axis, so a property of the optics (the same at the same rotation) lines up, and one of the dynamics does not.

COMPARE SCANS
  With 'compare scans' ticked, the scans picked in the Compare tab (Compare selected) are drawn too, one colour each (green, red, purple, ...), same markers; the shown scan keeps the leg colours.

EXPORT CSV / COPY CSV
  Every value on this plot as a table: method, leg, direction, time, rotation, analyzer angle, ER with its +-1 sigma, lower-bound flag, Imin, Imax, plus the per-sample fit's ER binned by rotation. A '#' header says what the scan was, what was subtracted, and what limits the numbers (read it with pandas.read_csv(path, comment='#')).
"""
ANALYZER_OFFSETS = ("measure after", "reuse latest", "none")
SEQ_ORDERS = ("interleaved (per angle)", "one setting at a time")
FIND_LIGHT = ("record window", "static light (line trigger)")
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
        self.cursor_dirty = set()     # tabs whose only change is the cursor time
        self.awg_sess = None          # awg.Session once the AWG is connected
        self.awg_eom = None           # EOM-ILC's eomilc, loaded with the AWG
        self.awg_wave = None          # the AWG tab's previewed waveform
        self.awg_set = None           # or a set of them (Fixed rotations, Sequence)
        self._worker_thread = None
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
        # Stop and progress are packed first, at the bottom: a tall mode tab
        # once squeezed them out of the window (AWG tab, 6 Oct 2026)
        self.build_runbar(left)
        # the measurement modes share the column below the hardware as tabs
        # (selected by frame)
        self.modes = ttk.Notebook(left)
        self.modes.pack(fill="x", padx=8, pady=3)
        self.build_scan(self._mode_tab("Ramp scan"))
        self.build_analyzer_mode(self._mode_tab("Analyzer"))
        self.build_awg(self._mode_tab("AWG"))
        self.build_bias(self._mode_tab("Fixed rotations"))
        self.build_ilc(self._mode_tab("ILC target"))
        self.bias_result = None
        self.ilc_summary = None
        self.build_right(right)
        self.load_settings()
        try:
            calib.apply(calib.get(self.cfg), self.cfg)
        except ValueError as exc:
            self.log(f"EOM calibration in the config refused ({exc}): using 1 Sep 2026 values")
            calib.apply(calib.get({}), self.cfg)
        self.refresh_scan_list()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.pump()
        self.log(f"Ramp Polarimeter {__version__}. Config: {cfgmod.CONFIG_PATH}")
        self.log("EOM calibration: " + calib.summary(calib.get(self.cfg)))
        self.load_sg(quiet=True)
        if self.autoconnect.get() and not self.simulate.get():
            root.after(300, self.auto_connect)

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

        self._worker_thread = threading.Thread(target=body, daemon=True)
        self._worker_thread.start()
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
        """One row per instrument: what it is, its address (blank = find it),
        Connect, and what is connected - the status sits in the row rather than
        on a line of its own. Fixed widths with wrap: an identity or a VISA
        address sized its label and once widened the whole left column."""
        f = ttk.LabelFrame(left, text="Hardware")
        f.pack(fill="x", padx=8, pady=(6, 3))
        g = ttk.Frame(f)
        g.pack(fill="x", padx=6, pady=(2, 0))
        g.columnconfigure(3, weight=1)

        def row(i, label, addr, connect, status):
            ttk.Label(g, text=label).grid(row=i, column=0, sticky="w", pady=1)
            addr.grid(row=i, column=1, sticky="w", padx=(4, 4), pady=1)
            b = ttk.Button(g, text="Connect", command=connect, width=8)
            b.grid(row=i, column=2, sticky="w", pady=1)
            self.busy_widgets.append(b)
            lab = CopyLabel(g, text=status, foreground="#666", width=30)
            lab.grid(row=i, column=3, sticky="ew", padx=(6, 0), pady=1)
            return lab
        self.scope_addr = tk.StringVar()
        self.scope_status = row(0, "Scope", ttk.Entry(g, textvariable=self.scope_addr, width=19),
                                self.do_connect_scope, "not connected")
        a = ttk.Frame(g)
        self.ell_port = tk.StringVar()
        ttk.Entry(a, textvariable=self.ell_port, width=7).pack(side="left")
        ttk.Label(a, text=" addr").pack(side="left")
        self.ell_addr = tk.StringVar()
        ttk.Entry(a, textvariable=self.ell_addr, width=3).pack(side="left", padx=(2, 0))
        self.ell_status = row(1, "ELL14", a, self.do_connect_ell, "not connected")
        # the BK Precision 4063B: never connected on open (another program may
        # hold it); the status shows the VISA session it is open on
        self.awg_addr = tk.StringVar()
        self.awg_hw = row(2, "AWG", ttk.Entry(g, textvariable=self.awg_addr, width=19),
                          self.do_connect_awg, "not connected")
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=(1, 3))
        self.simulate = tk.BooleanVar()
        ttk.Checkbutton(r, text="Simulate", variable=self.simulate).pack(side="left")
        self.autoconnect = tk.BooleanVar(value=bool(self.cfg.get("autoconnect", True)))
        ttk.Checkbutton(r, text="Connect on open (scope, ELL14)",
                        variable=self.autoconnect).pack(side="left", padx=(6, 0))
        self._btn(r, "Disconnect all", self.do_disconnect, padx=(8, 0))
        ttk.Button(r, text="Scope settings...", command=self.open_scope_settings).pack(
            side="left", padx=(4, 0))

    def build_analyzer(self, left):
        f = ttk.LabelFrame(left, text="Analyzer (ELL14)")
        f.pack(fill="x", padx=8, pady=3)
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=2)
        self.pos_label = CopyLabel(r, text="position: -", width=24)
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
        r = ttk.Frame(f)
        r.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(r, foreground="#666", justify="left", wraplength=230,
                  text="One PD required. Mon = Trek monitor, Cmd = Trek command, "
                       "Ref = pick-off before the analyzer.").pack(side="left")
        ttk.Button(r, text="EOM calibration...", command=self.open_calibration).pack(
            side="right")

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
        # the spin-echo sequence: shown when the preset is built from it
        self.seq_row = ttk.Frame(f)
        self.seq = {}
        for label, key, w in (("legs", "spacing_ms", 7), ("ms apart, motion", "motion_ms", 5),
                              ("ms; keep", "before_ms", 4), ("before /", "after_ms", 4)):
            ttk.Label(self.seq_row, text=label).pack(side="left", padx=(0, 2))
            v = tk.StringVar()
            e = ttk.Entry(self.seq_row, textvariable=v, width=w)
            e.pack(side="left", padx=(0, 4))
            e.bind("<Return>", lambda _e: self.sequence_changed())
            e.bind("<FocusOut>", lambda _e: self.sequence_changed())
            self.seq[key] = v
        ttk.Label(self.seq_row, text="ms after").pack(side="left")
        self.seq_lbl = CopyLabel(f, text="", foreground="#666", width=48)
        self._seq_anchor = r

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
        rr.pack(fill="x", padx=6, pady=1)
        self.stray_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="stray light at", variable=self.stray_on).pack(side="left")
        self.stray_vdiv = tk.StringVar(value="5")
        ttk.Entry(rr, textvariable=self.stray_vdiv, width=4).pack(side="left", padx=2)
        ttk.Label(rr, text="mV/div (dark and background both read there too)",
                  foreground="#666").pack(side="left")
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(4, 2))
        self._btn(rr, "Check scope", self.do_check_scope)
        self._btn(rr, "Start scan", self.do_start_scan, padx=(4, 0))
        self.est_label = CopyLabel(rr, text="", foreground="#666", width=44)
        self.est_label.pack(side="left", padx=6)
        for v in list(self.sv.values()) + [self.order, self.mode]:
            v.trace_add("write", lambda *_: self.update_estimate())

    def build_analyzer_mode(self, f):
        """The two ways the analyzer goes near crossed: find an angle in the
        light as it is and leave the analyzer there, or measure the shown
        scan's extinction precisely where its rotation is flat (null refine:
        extra angles near crossed, added to that scan)."""
        a = ttk.LabelFrame(f, text="Find the min / max transmission angle (and stay there)")
        a.pack(fill="x", padx=4, pady=(4, 2))
        self.build_find(a)
        b = ttk.LabelFrame(f, text="Null refine: precise Imin of the shown scan where it is flat")
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
        self._btn(rr, "Plan (log only)", self.do_plan_refine)
        self._btn(rr, "Measure, add to shown scan", self.do_run_refine, padx=6)
        ttk.Label(f, foreground="#666", justify="left", wraplength=440,
                  text="Where the rotation stands still (rest, holds, after: 'auto', or "
                       "ms ranges), steps the analyzer to these offsets from crossed at the "
                       "sensitive V/div and adds the captures to the shown scan: the "
                       "Extinction tab's null-refine points.").pack(anchor="w", padx=6,
                                                                    pady=(0, 4))

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
        ttk.Button(r, text="Rename / edit...", command=self.open_edit_scan).pack(side="left")
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
        self.plot_status = CopyLabel(r, text=PLOT_HINT, foreground="#666", width=100)
        self.plot_status.pack(side="left", padx=(12, 0), fill="x", expand=True)
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
        self.corr_label = CopyLabel(bar, text="", foreground="#8a4b00", width=130)
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
        self._ylog_box(self.fig_traces, False, "log y (linear within +-1 mV)")
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
        self._ylog_box(self.fig_malus, False, "log y (linear within +-1 mV)")
        self._cmp_box(self.fig_malus, "compare scans (same time)")
        self.fig_angle = self._fig_tab("Angle", self.draw_angle, click=True)
        self._cmp_box(self.fig_angle)
        self.fig_ext = self._fig_tab("Extinction", self.draw_extinction, click=True)
        top = self.fig_ext._ctl
        ctl = ttk.Frame(top)
        ctl.pack(fill="x")
        ctl2 = ttk.Frame(top)
        ctl2.pack(fill="x", pady=(2, 0))
        ttk.Label(ctl, text="x axis:").pack(side="left")
        self.ext_x = tk.StringVar(value="time")
        cb = ttk.Combobox(ctl, textvariable=self.ext_x, values=("time", "rotation"),
                          width=9, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw(self.fig_ext))
        ttk.Button(ctl, text="What are these?", command=self.show_er_help).pack(
            side="left", padx=(8, 0))
        ttk.Button(ctl, text="Export CSV...", command=self.do_export_er).pack(
            side="left", padx=(8, 0))
        ttk.Button(ctl, text="Copy CSV", command=self.do_copy_er).pack(side="left", padx=(4, 0))
        self._ylog_box(self.fig_ext, True, parent=ctl)
        self._cmp_box(self.fig_ext, parent=ctl)
        ttk.Label(ctl2, text="show:").pack(side="left")
        self.ext_show = {}
        for key, text in (("direct", "measured"), ("dips", "dip fit"),
                          ("refine", "null refine"), ("fit", "ER_fit line"),
                          ("lower", "lower bounds")):
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(ctl2, text=text, variable=v,
                            command=lambda: self.redraw(self.fig_ext)).pack(side="left",
                                                                            padx=(6, 0))
            self.ext_show[key] = v
        self.fig_poin = self._fig_tab("Poincaré", self.draw_poincare, click=True)
        self._cmp_box(self.fig_poin, "compare scans (chi)")
        ttk.Label(self.fig_poin._ctl, foreground="#666", text=(
            "A linear analyzer measures S1 and S2 only: |S3| is drawn as sqrt(1 - p^2), "
            "which assumes full polarization; the handedness is not measured. Click "
            "the time plot to move the cursor.")).pack(side="left")
        self.fig_diag = self._fig_tab("Diagnostics", self.draw_diagnostics)
        self._cmp_box(self.fig_diag)
        self.build_table_tab()
        self.build_shots_tab()
        self._ylog_box(self.fig_shots, False, "log y (linear within +-1 mV)", parent=None)
        self.build_build_tab()
        self.fig_corr = self._fig_tab("Corrections", self.draw_corrections)
        ttk.Button(self.fig_corr._ctl, text="Borrow dark / background from another scan...",
                   command=self.do_borrow_dialog).pack(side="left")
        self.build_compare_tab()
        # these draw without a ramp scan loaded
        self.fig_bias = self._fig_tab("Fixed rotations", self.draw_bias)
        self._ylog_box(self.fig_bias, True, "log y (ER, Imin)")
        self.fig_ilc = self._fig_tab("ILC target", self.draw_ilc)
        self.fig_find = self._fig_tab("Find angle", self.draw_find)
        self.fig_awg = self._fig_tab("AWG", self.draw_awg)
        self.free_tabs = {self.fig_bias._frame, self.fig_ilc._frame, self.fig_find._frame,
                          self.fig_cmp._frame, self.fig_awg._frame}
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
        self._hook_save(fig, toolbar)
        self.plot_tabs[frame] = (fig, draw)
        self.plot_dirty.add(frame)
        canvas.mpl_connect("button_press_event", self.copy_coords)
        if click:
            canvas.mpl_connect("button_press_event",
                               lambda ev, fig=fig: self.on_click(ev, fig))
        return fig

    # one colour per compared scan; the shown scan keeps its own (the leg colours)
    CMP_COLS = ("#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")

    def _ylog_box(self, fig, default=False, text="log y", parent="ctl"):
        """A 'log y' box for a tab. parent 'ctl': the tab's control row (packed
        right); None: the tab's side panel (stacked); or a frame."""
        v = tk.BooleanVar(value=default)
        fig._ylog = v
        par = fig._ctl if parent in ("ctl", None) else parent
        cb = ttk.Checkbutton(par, text=text, variable=v, command=lambda: self.redraw(fig))
        if parent is None:
            cb.pack(anchor="w", pady=(4, 0))
        else:
            cb.pack(side="right" if parent == "ctl" else "left", padx=(8, 0))
        return v

    def _cmp_box(self, fig, text="compare scans", parent="ctl"):
        """A 'compare scans' box: the scans analysed in the Compare tab drawn
        on this tab too."""
        v = tk.BooleanVar(value=False)
        fig._cmp_on = v

        def toggled():
            if v.get() and not self._compared():
                self.log("Compare scans: pick scans in the Compare tab's list and press "
                         "'Compare selected' - they are drawn on this tab.")
            self.redraw(fig)
        par = fig._ctl if parent == "ctl" else parent
        ttk.Checkbutton(par, text=text, variable=v, command=toggled).pack(
            side="right" if parent == "ctl" else "left", padx=(8, 0))
        return v

    def _compared(self, fig=None):
        """[(result, colour)]: the scans analysed in the Compare tab, without
        the shown one. With `fig`, only when that tab's 'compare scans' box is
        ticked."""
        if fig is not None and not (getattr(fig, "_cmp_on", None) and fig._cmp_on.get()) \
                and not getattr(fig, "_force_cmp", False):
            return []
        here = (os.path.normcase(os.path.abspath(self.result["d"].folder))
                if self.result else None)
        out = []
        for k in self.cmp_sel:
            r = self.cmp_results.get(k)
            if r is None or r.get("pol") is None:
                continue
            if here and os.path.normcase(os.path.abspath(r["d"].folder)) == here:
                continue
            out.append((r, self.CMP_COLS[len(out) % len(self.CMP_COLS)]))
        return out

    @staticmethod
    def _logy(fig, ax, kind="ratio", linthresh=1.0):
        """The tab's 'log y' box applied to ax: a ratio (always > 0) goes
        log; a level that can go below zero once the dark is subtracted goes
        symlog, linear within +-linthresh. Returns whether it is log."""
        v = getattr(fig, "_ylog", None)
        if v is None or not v.get():
            return False
        if kind == "ratio":
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=linthresh)
        return True

    def _cmp_dirty(self):
        """The compared scans changed: every tab that can draw them redraws."""
        for f in (self.fig_ext, self.fig_malus, self.fig_angle, self.fig_poin, self.fig_diag):
            self.plot_dirty.add(f._frame)

    # -- the extinction ratio as a table ---------------------------------------------
    def _er_csv(self):
        res = self.result
        if not res or res.get("pol") is None:
            self.log("Export: load a scan with a fit first (Plot data: Scan).")
            return None
        o = self._opts()
        extra = [f"exported {time.strftime('%Y-%m-%d %H:%M')} by rampol {__version__}; "
                 f"Apply switches: subtract dark/background {'on' if o['sub_dark'] else 'OFF'}, "
                 f"per-angle transmission {'on' if o['gains'] else 'OFF'}, drift correction "
                 f"{'on' if self.drift_on.get() else 'OFF'}, lock filter "
                 f"{'on' if o['lock'] else 'OFF'}"]
        return an.er_csv(res, extra=extra)

    def do_export_er(self):
        txt = self._er_csv()
        if txt is None:
            return
        d = self.result["d"]
        start = os.path.join(d.folder, "analysis")
        os.makedirs(start, exist_ok=True)
        p = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv",
                                         initialdir=start, initialfile=f"{d.name}_extinction.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(txt)
        n = sum(1 for ln in txt.splitlines() if ln and not ln.startswith("#")) - 1
        self.log(f"Extinction ratio: {n} rows -> {p}")

    def do_copy_er(self):
        txt = self._er_csv()
        if txt is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        n = sum(1 for ln in txt.splitlines() if ln and not ln.startswith("#")) - 1
        self.log(f"Extinction ratio of {self.result['d'].name}: {n} rows on the clipboard "
                 f"(CSV with a '#' header) - paste it anywhere")

    def _hook_save(self, fig, toolbar):
        """The toolbar's Save button: our dialog (a name and a folder filled
        in), not matplotlib's ('image' in the working directory). The button
        took its command when the toolbar was built, so it is re-pointed."""
        toolbar.save_figure = lambda *_a: self.save_figure(fig)
        try:
            toolbar._buttons["Save"].configure(command=toolbar.save_figure)
        except (AttributeError, KeyError, tk.TclError):
            pass

    def _figure_name(self, fig):
        """(folder, file stem) for saving a tab's figure: under the shown
        scan's folder (the output folder for the tabs that are not about a
        scan), in saved_figures/; named scan_tab[_view]."""
        tab = self.nb.tab(fig._frame, "text")
        out = self.outdir.get().strip() or self.cfg["outdir"]
        free = fig._frame in getattr(self, "free_tabs", ())
        res = self.result
        parts = []
        if fig is getattr(self, "fig_cmp", None):
            names = [r["d"].name for r, _c in self._compared()]
            if res:
                names.insert(0, res["d"].name)
            folder = res["d"].folder if res else out
            parts = ["compare"] + names[:3] + ([f"and_{len(names) - 3}_more"]
                                                if len(names) > 3 else [])
        elif free or not res:
            folder = out
            parts = [tab]
            if fig is getattr(self, "fig_bias", None) and getattr(self, "bias_result", None):
                parts = [self.bias_result.get("name", ""), tab]
            if fig is getattr(self, "fig_ilc", None):
                parts.append(os.path.splitext(self.ilc_fig.get())[0])
        else:
            folder = res["d"].folder
            parts = [res["d"].name, tab]
            if fig is self.fig_ext:
                parts.append(f"vs_{self.ext_x.get()}")
            if fig is self.fig_map:
                parts.append(self.map_show.get().split(" (")[0])
            if self.cursor_t is not None and fig in (self.fig_malus, self.fig_poin):
                parts.append(f"t{self.cursor_t * 1e3:.3f}ms")
        import unicodedata
        stem = "_".join(p for p in parts if p)
        stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
        stem = scanmod.safe_name(stem).replace("/", "_")
        return os.path.join(folder, "saved_figures"), stem

    def save_figure(self, fig):
        folder, stem = self._figure_name(fig)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            folder = os.path.dirname(folder)
        name, k = stem, 2
        while os.path.exists(os.path.join(folder, name + ".png")):
            name, k = f"{stem}_{k}", k + 1
        p = filedialog.asksaveasfilename(
            parent=self.root, title="Save the figure", initialdir=folder,
            initialfile=name + ".png", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")])
        if not p:
            return None
        fig.savefig(p, dpi=150)
        self.log(f"Figure saved: {p}")
        return p

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
        f.pack(side="bottom", fill="x", padx=8, pady=(0, 4))
        r = ttk.Frame(f)
        r.pack(fill="x")
        self.stop_btn = ttk.Button(r, text="Stop", command=self.do_stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.progress_bar = ttk.Progressbar(r, mode="determinate", maximum=1)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.progress_text = CopyLabel(f, text="", foreground="#060", width=48)
        self.progress_text.pack(anchor="w", pady=(0, 2))

    def build_bias(self, f):
        """Fixed rotations (in the code and the files: bias points): the AWG
        holds the EOMs at fixed rotations, the analyzer steps around each null
        at a sensitive V/div (rampol.bias)."""
        ttk.Label(f, justify="left", wraplength=330, text=(
            "Extinction ratio and rotation with the EOMs held still: the AWG holds "
            "each rotation in the list (a plateau, no ramp) and the analyzer measures "
            "Imin, Imax and the light's angle there.")).pack(anchor="w", padx=6, pady=(4, 2))
        self.bv = {}
        for items in ((("Rotations (deg)", "biases", 13), ("split on X1", "split", 5)),
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
        ttk.Button(rr, text="Preview", command=self.do_bias_preview).pack(
            side="left", padx=(8, 0))
        self._btn(rr, "Dry run on scope", self.do_bias_dry, padx=(4, 0))
        self._btn(rr, "Start", self.do_start_bias, padx=(4, 0))
        self._btn(rr, "Load...", self.do_load_bias, padx=(4, 0))
        ttk.Label(f, foreground="#666", justify="left", wraplength=330, text=(
            "Rotations: start:stop:step or a list, in degrees of EOM rotation. The "
            "4063B (close its GUI) plays plateaus on the bench trigger (EXT), CH1 "
            "-> X1, CH2 -> X2, checked against the Trek limits first. Per rotation: 4 "
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

    # -- AWG mode -----------------------------------------------------------------------------
    def build_awg(self, f):
        """The 4063B playing a waveform into the Treks (CH1 -> X1, CH2 -> X2):
        a ramp to a rotation and back, or two ILC drive files. Preview and
        check, dry run on the scope, load, outputs on / park / off, then
        measure in the hold."""
        self.av, self.a_choice = {}, {}

        def row(pady=1):
            rr = ttk.Frame(f)
            rr.pack(fill="x", padx=6, pady=pady)
            return rr

        def entry(rr, key, w, label=None, after=None):
            if label:
                ttk.Label(rr, text=label).pack(side="left", padx=(0, 1))
            self.av[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.av[key], width=w).pack(side="left", padx=(0, 4))
            if after:
                ttk.Label(rr, text=after).pack(side="left", padx=(0, 2))

        def combo(rr, key, values, w, default):
            self.a_choice[key] = tk.StringVar(value=default)
            ttk.Combobox(rr, textvariable=self.a_choice[key], values=values, width=w,
                         state="readonly").pack(side="left", padx=(0, 4))

        rr = row((4, 1))
        ttk.Label(rr, text="Waveform").pack(side="left", padx=(0, 2))
        combo(rr, "source", ("ramp", "ILC drives"), 10, "ramp")
        entry(rr, "rotation", 6, "rotation", "deg, split X1")
        entry(rr, "split", 4)
        combo(rr, "edge", ("cosine", "linear"), 7, "cosine")
        rr = row()
        for label, key in (("lead", "lead_ms"), ("rise", "rise_ms"), ("hold", "hold_ms"),
                           ("fall", "fall_ms"), ("after", "tail_ms")):
            entry(rr, key, 4, label)
        entry(rr, "dt_us", 4, "ms, dt", "us")
        self.a_record = CopyLabel(f, text="", foreground="#666", width=46)
        self.a_record.pack(anchor="w", padx=6)
        for k in ("lead_ms", "rise_ms", "hold_ms", "fall_ms", "tail_ms", "dt_us"):
            self.av[k].trace_add("write", lambda *_: self._awg_record_text())
        rr = row()
        entry(rr, "idle1", 7, "Idle X1")
        entry(rr, "idle2", 7, "X2", "V")
        ttk.Button(rr, text="EOM calibration...", command=self.open_calibration).pack(
            side="right")
        rr = row()
        ttk.Label(rr, text="ILC drives").pack(side="left")
        for label, key in (("X1", "file1"), ("X2", "file2")):
            ttk.Label(rr, text=label).pack(side="left", padx=(4, 1))
            self.av[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.av[key], width=13).pack(side="left")
            ttk.Button(rr, text="...", width=3,
                       command=lambda k=key: self.pick_awg_file(k)).pack(side="left")
        rr = row()
        self.a_fit_tb = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="scope from", variable=self.a_fit_tb).pack(side="left")
        entry(rr, "scope_before_ms", 4, None, "ms before the trigger to")
        entry(rr, "scope_after_ms", 4, None, "ms past the record")
        rr = row()
        entry(rr, "trig_hz", 4, "Trigger", "Hz")
        self.a_never = tk.BooleanVar(value=True)
        self.a_require = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="never let an output float", variable=self.a_never,
                        command=self._awg_flags).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(rr, text="require a dry run", variable=self.a_require,
                        command=self._awg_flags).pack(side="left", padx=(6, 0))
        rr = row()
        ttk.Label(rr, text="Dry run: AWG CH1 -> scope CH").pack(side="left")
        combo(rr, "dry_ch1", ("1", "2", "3", "4"), 2, "3")
        ttk.Label(rr, text="CH2 -> CH").pack(side="left")
        combo(rr, "dry_ch2", ("1", "2", "3", "4"), 2, "4")
        entry(rr, "dry_shots", 3, "shots")
        rr = row((3, 1))
        ttk.Button(rr, text="Preview", command=self.do_awg_preview).pack(side="left")
        self._btn(rr, "Dry run on scope", self.do_awg_dry, padx=(4, 0))
        self._btn(rr, "Load to AWG", self.do_awg_load, padx=(4, 0))
        ttk.Button(rr, text="?", width=2, command=lambda: self.log(AWG_HELP)).pack(
            side="left", padx=(4, 0))
        rr = row()
        self._btn(rr, "Outputs ON", self.do_awg_on)
        # never greyed out: they must work while anything else runs
        ttk.Button(rr, text="Park (idle, ON)", command=self.do_awg_park).pack(
            side="left", padx=(4, 0))
        ttk.Button(rr, text="Outputs OFF", command=self.do_awg_off).pack(side="left", padx=(4, 0))
        rr = row()
        entry(rr, "settle_ms", 4, "In the hold, after", "ms settle,")
        entry(rr, "shots", 3, "shots")
        self._btn(rr, "Find min", lambda: self.do_awg_find("min"), padx=(4, 0))
        self._btn(rr, "Find max", lambda: self.do_awg_find("max"), padx=(4, 0))
        # a sequence of end points: one ramp scan each, with the analyzer
        rr = row((5, 1))
        ttk.Label(rr, text="Sequence  X1 ends").pack(side="left", padx=(0, 1))
        self.av["seq_x1"] = tk.StringVar()
        ttk.Entry(rr, textvariable=self.av["seq_x1"], width=10).pack(side="left", padx=(0, 4))
        ttk.Label(rr, text="X2 ends").pack(side="left", padx=(0, 1))
        self.av["seq_x2"] = tk.StringVar()
        ttk.Entry(rr, textvariable=self.av["seq_x2"], width=10).pack(side="left", padx=(0, 2))
        ttk.Label(rr, text="deg").pack(side="left", padx=(0, 4))
        combo(rr, "seq_how", ("pairs", "grid"), 5, "pairs")
        rr = row()
        combo(rr, "seq_order", SEQ_ORDERS, 19, SEQ_ORDERS[0])
        ttk.Button(rr, text="Preview", command=self.do_seq_preview).pack(
            side="left", padx=(2, 0))
        self._btn(rr, "Dry run all", self.do_seq_dry, padx=(4, 0))
        self._btn(rr, "Start sequence", self.do_seq_start, padx=(4, 0))
        self.awg_seq_lbl = CopyLabel(f, text="", foreground="#666", width=46)
        self.awg_seq_lbl.pack(anchor="w", padx=6)
        for k in ("seq_x1", "seq_x2"):
            self.av[k].trace_add("write", lambda *_: self._seq_text())
        self.a_choice["seq_how"].trace_add("write", lambda *_: self._seq_text())
        self.awg_lbl = CopyLabel(f, text="AWG: not connected (connects on first use)",
                                 foreground="#666", width=47)
        self.awg_lbl.pack(anchor="w", padx=6, pady=(2, 4))

    def pick_awg_file(self, key):
        start = os.path.dirname(self.av[key].get()) or os.path.join(
            self.cfg.get("eomilc_path", ""), "run")
        p = filedialog.askopenfilename(title="EOM-ILC drive (AWG volts)", parent=self.root,
                                       initialdir=start if os.path.isdir(start) else None,
                                       filetypes=[("drive CSV", "drive_*.csv"), ("CSV", "*.csv")])
        if p:
            self.av[key].set(p)

    def _awg_flags(self):
        """The two safety ticks, onto a live session at once. Unticking
        never-float asks first."""
        if not self.a_never.get() and not messagebox.askyesno(
                "Never float", "Untick only when no AWG output reaches a stage that drives "
                "high on a floating input (the FPGA/buffer stage: -4 to -5.7 kV) - i.e. with "
                "it bypassed on BOTH channels. With the rule off, outputs go OFF for changes "
                "and at the end.\n\nUntick it?", parent=self.root):
            self.a_never.set(True)
        if not self.a_require.get() and not messagebox.askyesno(
                "Dry run", "Untick 'require a dry run': waveforms that have not been "
                "checked on the scope may then drive the Treks. Untick it?", parent=self.root):
            self.a_require.set(True)
        s = self.awg_sess
        if s is not None:
            s.never_float = bool(self.a_never.get())
            s.require_dry_run = bool(self.a_require.get())
        self._awg_status()

    def _awg_idle(self, c):
        """{EO1, EO2: idle V}: typed, or the ILC state files' first sample."""
        a = c["awg"]
        auto = awgmod.idle_from_states([c["ilc"].get("x1"), c["ilc"].get("x2")])
        out = {}
        for name, key in (("EO1", "idle1"), ("EO2", "idle2")):
            txt = str(a.get(key, "")).strip()
            v = float(txt) if txt else auto[name]
            if abs(v) > awgmod.IDLE_CAP:
                raise ValueError(f"{name} idle {v*1e3:+.0f} mV is past the "
                                 f"{awgmod.IDLE_CAP*1e3:.0f} mV cap")
            out[name] = v
        return out

    def _eom(self, c):
        try:
            return hw.load_eomilc(c["eomilc_path"])
        except Exception as exc:
            self.log(f"  EOM-ILC not loaded ({exc}): no Trek limit check")
            return None

    def _awg_record_text(self):
        """The ramp's record length under its fields: their sum."""
        try:
            p = {k: float(self.av[k].get()) for k in
                 ("lead_ms", "rise_ms", "hold_ms", "fall_ms", "tail_ms", "dt_us")}
        except (ValueError, KeyError):
            self.a_record.configure(text="")
            return
        rec = awgmod.record_ms(p)
        n = int(round(rec / (p["dt_us"] * 1e-3))) + 1 if p["dt_us"] > 0 else 0
        note = (" - the ILC's own record" if abs(rec - awgmod.ILC_RECORD_MS) < 1e-9
                and abs(p["dt_us"] - awgmod.ILC_DT_US) < 1e-9 else
                " - not the ILC's 11 ms: with the outputs live the change goes through "
                "idle; Park returns to 11 ms")
        self.a_record.configure(text=f"record {rec:g} ms = {n} points at {p['dt_us']:g} us"
                                     + note)

    def _awg_scope_tb(self, c, wave):
        a = c["awg"]
        return awgmod.timebase_for(wave, float(a.get("scope_before_ms", 0.2) or 0),
                                   float(a.get("scope_after_ms", 0.0) or 0))

    def _awg_build(self, c):
        """The waveform and its checks from the AWG tab (Tk thread)."""
        a = c["awg"]
        if a.get("source") == "ILC drives":
            wave = awgmod.from_files(a.get("file1"), a.get("file2"))
        else:
            wave = awgmod.ramp_hold(float(a["rotation"]), a, idle=self._awg_idle(c))
        return wave, self._wave_checks(c, wave)

    def _wave_checks(self, c, wave):
        a = c["awg"]
        found = awgmod.check(wave, self._eom(c), trig_hz=float(a.get("trig_hz") or 0) or None)
        # a spin-echo sequence triggers every leg: a record longer than the
        # legs' spacing is still playing when leg 2's trigger comes
        if (cfgmod.PRESETS.get(c.get("preset")) or {}).get("sequence"):
            gap = float((c.get("sequence") or {}).get("spacing_ms", 0) or 0)
            if gap and wave.period * 1e3 >= gap:
                found.append(("WARN", f"record {wave.period*1e3:.2f} ms is longer than the "
                                      f"{gap:g} ms between the spin-echo legs (Spin echo "
                                      f"preset): leg 2's trigger comes while the burst still "
                                      f"plays, and the 4063B ignores it - leg 2 is not driven"))
        return found

    def do_awg_preview(self):
        c = self.gather()
        try:
            wave, found = self._awg_build(c)
        except (ValueError, OSError) as exc:
            self.log(f"AWG preview: {exc}")
            return
        self.awg_wave, self.awg_found = wave, found
        self.awg_set = None
        self.report_checks(found, f"AWG waveform: {wave.label}", popup=False)
        idle = wave.idle()
        _, rot = awgmod.predict(wave)
        self.log(f"  {wave.n} points at {wave.dt*1e6:g} us = {wave.period*1e3:.3f} ms "
                 f"(FRQ {1/wave.period:.4f} Hz); idle X1 {idle['EO1']*1e3:+.1f} mV, "
                 f"X2 {idle['EO2']*1e3:+.1f} mV; peak rotation {np.max(np.abs(rot)):.2f} deg")
        s = self.awg_sess
        if s is not None:
            self.log("  dry run: " + ("passed this session" if s.is_verified(wave)
                                      else "not yet"))
        self.plot_dirty.add(self.fig_awg._frame)
        self.nb.select(self.fig_awg._frame)
        self.draw_visible()

    def _awg_session(self, c):
        """Worker thread: the AWG session, connecting on first use. Never on
        open: CH1 is often live from another program. The safety ticks are
        applied on every use."""
        a = c["awg"]
        if self.bench is not None:
            if self.awg_sess is None or getattr(self.awg_sess.awg, "bench", None) is not self.bench:
                self.awg_sess = awgmod.Session(sim.FakeAWG(self.bench), None, log=self.log)
        elif self.awg_sess is None:
            self.awg_eom = hw.load_eomilc(c["eomilc_path"])
            mod = hw.load_module(c["awg_path"], "bk4063b")
            import ilc_bench as ib
            ib._AWGMOD = mod
            awg = mod.BK4063B(connect=False, resource_manager=hw.shared_rm(mod.pyvisa))
            self.log(f"AWG: {awg.connect(c.get('awg_addr') or None).strip()} on "
                     f"{awg.resource_name}")
            self.awg_sess = awgmod.Session(awg, ib, log=self.log)
        self.awg_sess.never_float = bool(a.get("never_float", True))
        self.awg_sess.require_dry_run = bool(a.get("require_dry_run", True))
        return self.awg_sess

    def _awg_drive_info(self):
        """What this window's AWG plays into the Treks, for a scan's manifest
        - None unless its outputs are ON with a waveform of ours (not parked)."""
        s = self.awg_sess
        if s is None or not s.owned or s.parked or s.wave is None:
            return None
        return self._drive_info(s.wave, s.is_verified(s.wave))

    def _drive_info(self, w, verified):
        out = {"source": "this window's AWG (rampol)", "label": w.label,
               "names": list(awgmod.names(w)), "record_ms": w.period * 1e3,
               "dt_us": w.dt * 1e6, "idle_V": w.idle(), "dry_run_passed": bool(verified),
               "rotation_deg": w.rotation,
               "hold_ms": None if not w.hold else [w.hold[0] * 1e3, w.hold[1] * 1e3]}
        if getattr(w, "ends", None):
            out["ends_deg"] = {"X1": w.ends["EO1"], "X2": w.ends["EO2"]}
        if w.source == "ramp":
            a = self.cfg.get("awg", {})
            out["ramp"] = {k: a.get(k) for k in ("rotation", "split", "edge", "lead_ms",
                                                 "rise_ms", "hold_ms", "fall_ms", "tail_ms")}
        else:
            out["files"] = dict(w.files)
        return out

    # -- a sequence of ramp end points, interleaved with the analyzer ---------------
    def _seq_waves(self, c):
        """(ends, waves, findings) for the AWG tab's sequence: one ramp per
        (X1, X2) end point, the ramp's other settings from the tab."""
        a = c["awg"]
        ends = awgmod.parse_ends(a.get("seq_x1", ""), a.get("seq_x2", ""),
                                 a.get("seq_how", "pairs"))
        idle = self._awg_idle(c)
        waves = [awgmod.ramp_hold(0.0, a, idle=idle, ends={"EO1": e1, "EO2": e2})
                 for e1, e2 in ends]
        found = []
        for w in waves:
            for lv, msg in self._wave_checks(c, w):
                if lv != "INFO":
                    found.append((lv, f"X1 {w.ends['EO1']:g} / X2 {w.ends['EO2']:g}: {msg}"))
        return ends, waves, found

    def _seq_names(self, base, ends):
        return [scanmod.safe_name(f"{base}_X1_{e1:g}_X2_{e2:g}") for e1, e2 in ends]

    def _seq_text(self):
        try:
            a = {k: self.av[k].get() for k in ("seq_x1", "seq_x2")}
            ends = awgmod.parse_ends(a["seq_x1"], a["seq_x2"], self.a_choice["seq_how"].get())
        except (ValueError, KeyError) as exc:
            self.awg_seq_lbl.configure(text=str(exc) if self.av["seq_x1"].get() else "")
            return
        shown = ", ".join(f"{e1:g}/{e2:g}" for e1, e2 in ends[:8])
        self.awg_seq_lbl.configure(text=f"{len(ends)} ramps (X1/X2 deg): {shown}"
                                    + (" ..." if len(ends) > 8 else "")
                                    + " - one ramp scan each, at the Ramp scan tab's angles")

    def do_seq_dry(self):
        """Every ramp of the sequence through the scope (the dry run)."""
        if not self.need(ell=False):
            return
        c = self.gather()
        self.save_settings()
        try:
            ends, waves, found = self._seq_waves(c)
            wiring = self._awg_wiring(c)
        except (ValueError, OSError) as exc:
            self.log(f"Sequence dry run: {exc}")
            return
        if self.report_checks(found, f"AWG sequence ({len(waves)} ramps)") == "FAIL":
            self.log("Dry run not started.")
            return
        if not self._dry_confirm(wiring, len(waves)):
            return
        a = c["awg"]
        self._run_dry(waves, wiring, c, f"sequence X1 {a['seq_x1']} / X2 {a['seq_x2']} "
                                        f"({a['seq_how']})")

    def do_seq_start(self):
        """One ramp scan per (X1, X2) end point, the AWG loaded live between
        them, interleaved with the analyzer: at each angle every ramp in
        turn (or each ramp's scan whole, 'one setting at a time')."""
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        if "PD" not in [r for r, _ in cfgmod.channel_roles(c).values()]:
            self.log("No channel has the PD role.")
            return
        try:
            ends, waves, found = self._seq_waves(c)
        except (ValueError, OSError) as exc:
            self.log(f"Sequence: {exc}")
            return
        if self.report_checks(found, f"AWG sequence ({len(waves)} ramps)") == "FAIL":
            self.log("Sequence not started.")
            return
        s = self.awg_sess
        if c["awg"].get("require_dry_run", True):
            unver = [w for w in waves if s is None or not s.is_verified(w)]
            if unver:
                if messagebox.askyesno(
                        "Not dry-run", f"{len(unver)} of the {len(waves)} ramps have not "
                        f"passed a dry run on the scope this session, and 'require a dry "
                        f"run' is ticked.\n\nDry-run all of them now?", parent=self.root):
                    self.do_seq_dry()
                return
        base = scanmod.safe_name(c.get("scan_name") or "sequence")
        names = self._seq_names(base, ends)
        out = c["outdir"]
        existing = [n for n in names if os.path.isfile(os.path.join(out, n, f"{n}_scan.json"))]
        resume = False
        if existing:
            mans = []
            for n in existing:
                with open(os.path.join(out, n, f"{n}_scan.json"), encoding="utf-8") as fh:
                    mans.append(json.load(fh))
            ours = len(existing) == len(names) and all(
                (m.get("plan", {}).get("series") or {}).get("base") == base for m in mans)
            left = sum(1 for m in mans for x in m["steps"] if x.get("status") != "done")
            if ours and left:
                ans = messagebox.askyesnocancel(
                    "Sequence exists", f"The sequence {base} ({len(names)} scans) has {left} "
                    f"step(s) not measured.\n\nYes = resume it\nNo = a new sequence\n"
                    f"Cancel = do nothing", parent=self.root)
                if ans is None:
                    return
                resume = bool(ans)
            if not resume:
                k = 2
                while True:
                    nb = f"{base}_{k}"
                    if not any(os.path.exists(os.path.join(out, n))
                               for n in self._seq_names(nb, ends)):
                        break
                    k += 1
                self.log(f"{base} is taken - this sequence is {nb}")
                base = nb
                names = self._seq_names(base, ends)
                self.scan_name.set(base)
        peak = max(float(np.max(np.abs(awgmod.predict(w)[1]))) for w in waves)
        if not messagebox.askokcancel(
                "AWG sequence", f"{len(waves)} ramps (X1/X2 end points "
                f"{', '.join(f'{e1:g}/{e2:g}' for e1, e2 in ends[:6])}"
                f"{' ...' if len(ends) > 6 else ''}), one ramp scan each, "
                f"{c['awg'].get('seq_order', SEQ_ORDERS[0])}.\n\nThe AWG outputs go ON "
                f"into the Treks (CH1 -> X1, CH2 -> X2) and each ramp is loaded live as "
                f"the sequence goes (up to {peak:.1f} deg). At the end the AWG is parked. "
                f"Park or Outputs OFF stop it at any time.\n\nStart?", parent=self.root):
            return
        runs = [self._new_run(n) for n in names]
        here = {}
        for r in runs:
            r.here = here
        sc = c["scan"]
        dm, bm = self.dark_mode.get(), self.bg_mode.get()
        pre = [k for k, m in (("dark", dm), ("background", bm)) if m == "measure"]
        if resume:
            for r in runs:
                r.load()
            pre = [k for k in pre if any(x["kind"] == k and x["status"] != "done"
                                         for x in runs[0].manifest["steps"])]
            reuse = []
        else:
            angles = scanmod.ordered(scanmod.angle_list(sc["start"], sc["stop"], sc["step"]),
                                     sc["order"])
            steps = scanmod.build_steps(angles, int(sc["ref_every"]), sc["ref_angle"])
            offs = [{"kind": k, "target": 0.0} for k in pre]
            stray = self._stray_scale(c) if len(pre) == 2 else None
            if stray:
                offs = [{"kind": "dark", "target": 0.0},
                        {"kind": "dark", "target": 0.0, "pd_scale": stray},
                        {"kind": "background", "target": 0.0, "pd_scale": stray},
                        {"kind": "background", "target": 0.0}]
            plan = dict(sc, preset=c["preset"], software=f"rampol {__version__}",
                        dark_mode=dm, bg_mode=bm)
            if (cfgmod.PRESETS.get(c["preset"]) or {}).get("sequence"):
                plan["sequence"] = dict(c.get("sequence") or cfgmod.DEFAULTS["sequence"])
            prov = self._provenance(c)
            order = c["awg"].get("seq_order", SEQ_ORDERS[0])
            for k, (r, w, (e1, e2)) in enumerate(zip(runs, waves, ends)):
                pl = dict(plan, series={"base": base, "index": k, "of": len(runs),
                                        "ends_deg": {"X1": e1, "X2": e2}, "order": order,
                                        "members": names})
                r.new(pl, (offs if k == 0 else []) + [dict(x) for x in steps],
                      extra={"zero_deg": float(c["ell_zero_deg"]), "provenance": prov,
                             "drive": self._drive_info(w, s is not None and s.is_verified(w))})
            reuse = [k for k, m in (("dark", dm), ("background", bm)) if m == "reuse latest"]
            if bm == "reuse latest" and self._stray_scale(c):
                reuse.append("stray")
        self.run = runs[0]
        self.seq_runs = runs
        self.log(f"Sequence {base}: {len(runs)} ramp scans ({', '.join(names[:3])}"
                 f"{' ...' if len(names) > 3 else ''})")

        def start():
            # which scan the plots follow: read here, on the Tk thread
            shown = os.path.normcase(os.path.abspath(self._scan_path(self.show_scan.get()) or ""))
            watch = next((k for k, r in enumerate(runs)
                          if os.path.normcase(os.path.abspath(r.folder)) == shown), 0)
            self.worker(lambda: self._run_seq(runs, waves, c, watch), done=self._seq_done)

        def after_offsets():
            if reuse:
                self.borrow_latest(runs[0], reuse, start)
            else:
                start()
        self.offsets_then(runs[0], pre, after_offsets)

    def _seq_share_offsets(self, runs):
        """Worker: the first scan's dark / background / stray light, lent to
        the others (one beam block serves the whole sequence)."""
        if not any(x["kind"] in an.OFFSET_KINDS and x.get("status") == "done"
                   for x in runs[0].manifest["steps"]) and not runs[0].manifest.get("borrowed"):
            return                         # nothing measured or borrowed to lend
        sg = self.load_sg()
        d0 = an.load_scan(runs[0].folder, sg.load_capture, trim=int(self.cfg["analysis"]["trim"]))
        pd = next(ch for ch, (r, _n) in cfgmod.channel_roles(self.cfg).items() if r == "PD")
        vdiv = self.link.channel_state([pd])[pd][0]
        lv = an.offset_levels(d0, vdiv)
        lend = {}
        for k in an.OFFSET_KINDS:
            e = lv.get(k)
            if e:
                lend[k] = {"level": float(e["level"]), "sem": float(e.get("sem") or 0.0),
                           "n": int(e.get("n") or 0), "vdiv": float(e["vdiv"]),
                           "offset": float(e["offset"]), "measured": e.get("measured", ""),
                           "source": runs[0].name if e["source"] == "this scan" else e["source"]}
        st = an.stray_light(d0)
        if st:
            lend["stray"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in st.items()}
            if st["source"] == "this scan":
                lend["stray"]["source"] = runs[0].name
        if lend:
            for r in runs[1:]:
                r.manifest.setdefault("borrowed", {}).update(lend)
                r.save()

    def _run_seq(self, runs, waves, c, watch=0):
        """Worker: the sequence's steps in order, the AWG loaded live when
        the ramp changes; parked at the end, whatever happens."""
        sess = self._awg_session(c)
        a = c["awg"]
        self._seq_share_offsets(runs)
        steps = [[x for x in r.manifest["steps"] if x["kind"] in ("scan", "ref")] for r in runs]
        n = min(len(x) for x in steps)
        if str(a.get("seq_order", "")).startswith("one"):
            order = [(k, i) for k in range(len(runs)) for i in range(n)]
        else:
            order = [(k, i) for i in range(n) for k in range(len(runs))]
        todo = [(k, i) for k, i in order if steps[k][i].get("status") != "done"]
        settle = float(a.get("seq_settle_s", 1.0) or 0)
        from .bias import eta
        t0, done, cur = time.time(), 0, None
        try:
            for k, i in todo:
                if self.stop_flag.is_set():
                    raise hw.Cancelled()
                if cur != k:
                    if sess.wave is None or awgmod.names(sess.wave) != awgmod.names(waves[k]):
                        sess.load(waves[k], keep_on=True)
                    if not all(sess.outputs().values()):
                        sess.on()
                    cur = k
                    if settle > 0:
                        time.sleep(settle)
                st = steps[k][i]
                self._progress(done, len(todo), f"sequence: {runs[k].name}: {st['kind']} "
                                                f"{st['target']:.2f} deg ({done + 1}/{len(todo)}"
                                                f"{eta(t0, done, len(todo))})")
                f = runs[k].folder
                runs[k].run_step(st, on_step=(lambda _s, f=f: self.call(self.live_refresh, f))
                                 if k == watch else None)
                done += 1
            self._progress(len(todo), len(todo), "sequence done")
        except hw.Cancelled:
            self.log(f"Sequence stopped after {done} of {len(todo)} steps. Start it again "
                     f"under the same name (AWG tab) to resume.")
        finally:
            try:
                sess.end()
            except Exception as exc:
                self.log(f"  AWG park at the end: {exc}")
        return runs

    def _seq_done(self, runs):
        self._awg_status()
        self.log(f"Sequence: {len(runs)} scans - all of them in the Compare tab")
        self.refresh_scan_list()
        names = [r.name for r in runs]
        self.cmp_lb.selection_clear(0, "end")
        for i, n in enumerate(self.cmp_lb.get(0, "end")):
            if n in names:
                self.cmp_lb.selection_set(i)
        self.do_compare_load()

    def do_connect_awg(self):
        """Connect the 4063B now rather than on first use. Only *IDN? is sent:
        the outputs are left exactly as they are (nothing of this window's is
        loaded, so even the never-float rule has nothing to park)."""
        c = self.gather()
        roles = self.roles()

        def go():
            if c["simulate"] and self.bench is None:
                self.ensure_sim(roles)
            self._awg_session(c)
        self.worker(go, done=lambda _r: self._awg_status())

    def _awg_close(self):
        """End the AWG (park under the never-float rule, else off) and let it go."""
        s = self.awg_sess
        if s is None:
            return
        try:
            if s.owned or s.never_float:
                s.end()
            if self.bench is None:
                s.awg.close()
        except Exception as exc:
            self.log(f"AWG close: {exc}")
        self.awg_sess = None

    def _awg_status(self):
        s = self.awg_sess
        if s is None:
            self.awg_lbl.configure(text="AWG: not connected (connects on first use)",
                                   foreground="#666")
            self.awg_hw.configure(text="not connected", foreground="#666")
            return
        dev = s.awg
        if self.bench is not None and getattr(dev, "bench", None) is self.bench:
            where = "simulated 4063B"
        else:
            where = (f"{short_idn(getattr(dev, 'idn', '') or '4063B')} on "
                     f"{getattr(dev, 'resource_name', '') or '?'}")
        self.awg_hw.configure(text=where + ("; outputs ON" if s.owned else ""),
                              foreground="#060")
        on = bool(s.owned)
        if s.wave is None:
            what = "nothing of this window's loaded"
        else:
            what = s.wave.label + ("" if s.wave.source == "park" else
                                   (" - dry run passed" if s.is_verified(s.wave)
                                    else " - NOT dry-run"))
        state = ("parked: idle, outputs ON" if on and s.parked else
                 ("outputs ON" if on else "outputs OFF"))
        self.awg_lbl.configure(text=f"AWG: {what}; {state}",
                               foreground="#c00000" if on and not s.parked else "#060")

    def _awg_wiring(self, c):
        a = c["awg"]
        w = {"EO1": int(a.get("dry_ch1") or 3), "EO2": int(a.get("dry_ch2") or 4)}
        if w["EO1"] == w["EO2"]:
            raise ValueError("the two AWG outputs need two different scope channels")
        return w

    def _dry_confirm(self, wiring, n):
        roles = {ch: r for ch, (r, _n) in cfgmod.channel_roles(self.cfg).items()}
        busy = [f"CH{ch} ({roles[ch]})" for ch in wiring.values() if ch in roles]
        note = (f"\n\nScope {', '.join(busy)} normally carries another signal: its cable comes "
                f"off for the dry run; its V/div and offset are put back after.") if busy else ""
        return messagebox.askokcancel(
            "Dry run on the scope",
            f"Dry run of {n} waveform{'s' if n > 1 else ''}:\n\n"
            f"   AWG CH1 -> scope CH{wiring['EO1']}\n   AWG CH2 -> scope CH{wiring['EO2']}\n\n"
            "A BNC tee at each AWG output keeps the next stage's input driven.\n"
            "The Treks must NOT drive the EOMs: HV disabled, or their outputs "
            "disconnected.\n\nThe outputs go ON into the scope; each is first played "
            "alone (the other at idle) to check the cabling. The scope's trigger stays "
            f"as it is (the bench trigger).{note}\n\nStart?", parent=self.root)

    def do_awg_dry(self):
        if not self.need(ell=False):
            return
        c = self.gather()
        self.save_settings()
        try:
            wave, found = self._awg_build(c)
            wiring = self._awg_wiring(c)
        except (ValueError, OSError) as exc:
            self.log(f"Dry run: {exc}")
            return
        self.awg_wave, self.awg_found = wave, found
        self.awg_set = None
        if self.report_checks(found, f"AWG waveform: {wave.label}") == "FAIL":
            self.log("Dry run not started.")
            return
        if not self._dry_confirm(wiring, 1):
            return
        self._run_dry([wave], wiring, c, wave.label)

    def _run_dry(self, waves, wiring, c, label):
        shots = int(c["awg"].get("dry_shots") or 4)
        wait_s = float(c["scan"].get("wait_s", 10.0))
        sim_mode = self.bench is not None

        def go():
            sess = self._awg_session(c)
            if sim_mode:                       # the simulated re-cabling
                self.bench.wiring = "scope"
                self.bench.awg_scope = {1: wiring["EO1"], 2: wiring["EO2"]}
            reps = []
            from .bias import eta
            t_run = time.time()
            try:
                for i, w in enumerate(waves):
                    self._progress(i, len(waves), f"dry run {i + 1}/{len(waves)}: {w.label}"
                                                  f"{eta(t_run, i, len(waves))}")
                    swing = max(float(np.ptp(w.u[k])) for k in w.u)
                    ident = swing > 0.05 and not any(r["identified"] for r in reps)
                    rep = awgmod.dry_run(sess, self.link, w, wiring, shots=shots,
                                         wait_s=wait_s, cancelled=self.stop_flag.is_set,
                                         log=self.log, identify=ident)
                    rep["identified"] = ident
                    reps.append(rep)
                    if not rep["ok"]:
                        break
                self._progress(len(waves), len(waves), "dry run done")
            finally:
                sess.end()
                if sim_mode:
                    self.bench.wiring = "treks"
            return reps

        def done(reps):
            self.awg_dry, self.awg_dry_all = reps[-1], reps
            ok = len(reps) == len(waves) and all(r["ok"] for r in reps)
            self._save_dry(reps, label, c)
            s = self.awg_sess
            end = ("parked at idle, outputs ON" if s is not None and s.never_float
                   else "outputs OFF")
            if ok:
                self.log(f"Dry run PASSED: {len(reps)} waveform(s), {label}")
                messagebox.showinfo(
                    "Dry run passed",
                    f"{len(reps)} waveform(s) played as meant ({label}).\n\nThe AWG is now "
                    f"{end}. To drive the Treks: take the scope off the tees (or put the "
                    f"cables back) and re-enable the Treks, then Outputs ON or start the run.",
                    parent=self.root)
            else:
                bad = reps[-1]["problems"]
                self.log(f"Dry run FAILED ({reps[-1]['label']}):")
                for pr in bad:
                    self.log(f"  {pr}")
                messagebox.showwarning("Dry run failed", f"{reps[-1]['label']}:\n\n"
                                       + "\n\n".join(bad[:6]), parent=self.root)
            self._awg_status()
            self.plot_dirty.add(self.fig_awg._frame)
            self.nb.select(self.fig_awg._frame)
            self.draw_visible()
        self.worker(go, done=done)

    def _save_dry(self, reps, label, c):
        """outdir/awg_dryrun/<time>_<label>.json + .npz (the captured traces),
        and a lab-log row."""
        import datetime
        when = datetime.datetime.now()
        stem = when.strftime("%Y%m%d-%H%M%S") + "_" + scanmod.safe_name(label)[:60]
        out = os.path.join(c["outdir"], "awg_dryrun")
        try:
            os.makedirs(out, exist_ok=True)
            traces, slim = {}, []
            for i, r in enumerate(reps):
                rr = {k: v for k, v in r.items() if k != "steps"}
                rr["steps"] = {}
                for step, res in r["steps"].items():
                    rr["steps"][step] = {}
                    for n, x in res.items():
                        t, v = x["trace"]
                        traces[f"w{i}_{step}_{n}_t"] = t
                        traces[f"w{i}_{step}_{n}_v"] = v
                        rr["steps"][step][n] = {k: v_ for k, v_ in x.items()
                                                if k not in ("trace", "model")}
                slim.append(rr)
            with open(os.path.join(out, stem + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"label": label, "when": when.isoformat(timespec="seconds"),
                           "calibration": calib.get(c), "reports": slim}, fh, indent=1,
                          default=_jsonable)
            np.savez_compressed(os.path.join(out, stem + ".npz"), **traces)
            self._lab_upsert(c["outdir"], lablog.dry_row(slim, label, when))
            self.log(f"  dry run record: {os.path.join(out, stem)}.json/.npz")
        except OSError as exc:
            self.log(f"  dry run record not written: {exc}")

    def do_awg_load(self):
        if not self.need(ell=False):
            return
        c = self.gather()
        self.save_settings()
        try:
            wave, found = self._awg_build(c)
        except (ValueError, OSError) as exc:
            self.log(f"AWG: {exc}")
            return
        self.awg_wave, self.awg_found = wave, found
        self.awg_set = None
        if self.report_checks(found, f"AWG waveform: {wave.label}") == "FAIL":
            self.log("Not loaded.")
            return
        fit_tb = bool(c["awg"].get("fit_timebase"))

        def go():
            sess = self._awg_session(c)
            names = sess.load(wave)
            if fit_tb and self.link is not None:
                div, pos = self._awg_scope_tb(c, wave)
                sc = self.link.scope
                sc.put(":TIMebase:REFerence", "LEFT")
                sc.put(":TIMebase:SCALe", f"{div:.6g}")
                sc.put(":TIMebase:POSition", f"{pos:.6g}")
                self.log(f"  scope timebase {div*1e3:g} ms/div from {(pos - div)*1e3:+.2f} ms "
                         f"to {(pos + 9 * div)*1e3:+.2f} ms (the record is "
                         f"{wave.period*1e3:.3f} ms)")
            return names

        def done(names):
            self.log(f"AWG loaded: {wave.label} ({', '.join(names.values())})")
            self._awg_status()
            self.plot_dirty.add(self.fig_awg._frame)
            self.draw_visible()
        self.worker(go, done=done)

    def do_awg_on(self):
        s = self.awg_sess
        if s is None or s.wave is None:
            self.log("AWG: load a waveform first (Load to AWG).")
            return
        if s.require_dry_run and not s.is_verified(s.wave):
            if messagebox.askyesno(
                    "Not dry-run", f"'{s.wave.label}' has not passed a dry run on the scope "
                    f"this session, and 'require a dry run' is ticked.\n\nRun the dry run "
                    f"now?", parent=self.root):
                self.do_awg_dry()
            return
        _, rot = awgmod.predict(s.wave)
        if not messagebox.askokcancel(
                "AWG outputs ON",
                f"Switch both AWG outputs ON into the Treks?\n\nCH1 -> X1 and CH2 -> X2 play\n"
                f"{s.wave.label}\non every bench trigger (up to "
                f"{float(np.max(np.abs(rot))):.1f} deg of rotation).\n\nPark or Outputs "
                f"OFF stop it at any time.", parent=self.root):
            return
        self.worker(s.on, done=lambda _o: self._awg_status())

    def _bg_awg(self, fn, what):
        """An AWG action beside the worker, so Park / OFF work while a
        measurement runs. One thread takes them in the order they were
        pressed: two threads per click once let a Park that started first
        finish last, switching the outputs back on after an OFF."""
        if getattr(self, "_awg_q", None) is None:
            self._awg_q = queue.Queue()

            def loop():
                while True:
                    f, w = self._awg_q.get()
                    try:
                        f()
                    except Exception as exc:
                        self.log(f"AWG {w} failed: {exc}")
                    self.call(self._awg_status)
            threading.Thread(target=loop, daemon=True, name="awg-actions").start()
        self._awg_q.put((fn, what))

    def do_awg_park(self):
        """Stop driving without letting anything float: an idle-level
        waveform, outputs ON."""
        c = self.gather()
        try:
            idle = self._awg_idle(c)
        except ValueError as exc:
            self.log(f"Park: {exc}")
            return
        base = awgmod.ramp_hold(0.0, c["awg"], idle=idle)
        s = self.awg_sess
        if s is not None:
            self._bg_awg(lambda: s.park(like=s.wave or base), "park")
        else:
            self.worker(lambda: self._awg_session(c).park(like=base),
                        done=lambda _o: self._awg_status())

    def do_awg_off(self):
        s = self.awg_sess
        if s is None:
            self.log("AWG not connected - nothing to switch off from here.")
            return
        if s.never_float and not messagebox.askyesno(
                "Outputs OFF", "The never-float rule is on. An output switched OFF lets its "
                "cable float, and a stage that drives high on a floating input (the "
                "FPGA/buffer stage: -4 to -5.7 kV) will do so.\n\nPark stops the drive "
                "safely. Switch OFF only with that stage bypassed on both channels.\n\n"
                "Switch both outputs OFF?", parent=self.root):
            return
        self._bg_awg(lambda: s.off(force=True), "OFF")

    def do_awg_find(self, kind):
        s = self.awg_sess
        if not self.need():
            return
        if s is None or s.wave is None or s.wave.hold is None:
            self.log("AWG: load a ramp waveform (it has a hold) first.")
            return
        if s.owned != set(awgmod.CHANNELS.values()) or s.parked:
            self.log("AWG: switch the outputs ON with the waveform first (Outputs ON).")
            return
        c = self.gather()
        a = c["awg"]
        settle = float(a.get("settle_ms", 4.0)) * 1e-3
        t0, t1 = s.wave.hold[0] + settle, s.wave.hold[1] - 0.2e-3
        if t1 - t0 < 0.3e-3:
            self.log(f"AWG: the hold leaves {max(t1 - t0, 0)*1e3:.2f} ms after the settle - "
                     f"lengthen the hold")
            return
        roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(c).items()}
        chroles = self.roles()
        use_preset = bool(self.find_preset.get())
        zoom = bool(self.find_zoom.get())
        sc_plan = c["scan"]
        plan = {"shots": int(a.get("shots") or 8),
                "wait_s": float(sc_plan.get("wait_s") or 10.0),
                "dither_codes": int(sc_plan.get("dither_codes", 3))}
        rotation = s.wave.rotation
        wave = s.wave

        def go():
            from . import bias as biasmod
            st = (self._scope_like_scan(c, chroles, f"AWG find {kind}", keep_timebase=True)
                  if use_preset else self.link.scope.read_settings())
            src = str(st.get(":TRIGger:EDGE:SOURce", "")).upper()
            if src.startswith("LINE"):
                raise RuntimeError("the scope triggers on LINE: the AWG bursts on the bench "
                                   "trigger - set the trigger source to it (EXT)")
            def run():
                return biasmod.find_extremum(
                    self.link, self.rot, roles, kind, window_s=(t0, t1), plan=plan,
                    log=self.log, cancelled=self.stop_flag.is_set, ask=self.ask_main,
                    progress=self._progress)
            post = self._analyzer_offsets(c, roles, plan)
            if zoom:
                plan["points"] = 20000
                with biasmod.window_timebase(self.link, (t0, t1), self.log):
                    out = post(run(), (t0, t1), "awg hold")
            else:
                # the AWG tab's scope span, whatever the preset's timebase
                div, pos = self._awg_scope_tb(c, wave)
                sc = self.link.scope
                sc.put(":TIMebase:REFerence", "LEFT")
                sc.put(":TIMebase:SCALe", f"{div:.6g}")
                sc.put(":TIMebase:POSition", f"{pos:.6g}")
                self._window_in_record(sc.read_settings(), (t0, t1), "hold window")
                out = post(run(), (t0, t1), "awg hold")
            out["bias"] = rotation
            return out
        self.worker(go, done=lambda out: self._find_done(out, c["outdir"]))

    def draw_awg(self, fig):
        if self.awg_set:
            return self._draw_awg_set(fig, self.awg_set)
        w = self.awg_wave
        if w is None and self.awg_sess is not None:
            w = self.awg_sess.wave
        if w is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No waveform yet - AWG tab: Preview", ha="center",
                    va="center", transform=ax.transAxes, color="#888")
            ax.set_axis_off()
            return
        dry = getattr(self, "awg_dry", None)
        if dry is not None and tuple(dry["names"]) == awgmod.names(w) and "both" in dry["steps"]:
            return self._draw_dry(fig, w, dry)
        mon, rot = awgmod.predict(w)
        t = w.t * 1e3
        ax = fig.add_subplot(211)
        for name, col in (("EO1", "#1f77b4"), ("EO2", "#2ca02c")):
            ax.plot(t, w.u[name], color=col, lw=0.9,
                    label=f"CH{awgmod.CHANNELS[name]} -> {name} (idle {w.u[name][0]*1e3:+.1f} mV)")
        ax.set_ylabel("AWG output (V)")
        ax.set_title(f"{w.label}: {w.n} points, {w.period*1e3:.3f} ms", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)
        ax2 = fig.add_subplot(212, sharex=ax)
        ax2.plot(t, rot, color="k", lw=0.9, label="rotation (EOM calibration)")
        for name, col in (("EO1", "#1f77b4"), ("EO2", "#2ca02c")):
            ax2.plot(t, 90 * mon[name] / awgmod.biasmod.CHAN[name]["v90"], color=col,
                     lw=0.6, ls="--", label=f"{name} share")
        if w.hold:
            ax2.axvspan(w.hold[0] * 1e3, w.hold[1] * 1e3, color="0.9", lw=0, label="hold")
            try:
                settle = float(self.av["settle_ms"].get())
            except ValueError:
                settle = 4.0
            if w.hold[1] - w.hold[0] > settle * 1e-3 + 0.5e-3:
                ax2.axvspan(w.hold[0] * 1e3 + settle, w.hold[1] * 1e3 - 0.2,
                            color="#9ecae1", alpha=0.5, lw=0, label="Find window")
        try:
            c_ = self.gather()
            if c_["awg"].get("fit_timebase"):
                div, pos = self._awg_scope_tb(c_, w)
                for a_ in (ax, ax2):
                    a_.axvline((pos - div) * 1e3, color="#d62728", lw=0.8, ls=":")
                    a_.axvline((pos + 9 * div) * 1e3, color="#d62728", lw=0.8, ls=":")
                ax2.plot([], [], color="#d62728", lw=0.8, ls=":", label="the scope's record")
                ax.set_xlim((pos - div) * 1e3, (pos + 9 * div) * 1e3)
        except (ValueError, KeyError):
            pass
        ax2.set_xlabel("time from the trigger (ms)")
        ax2.set_ylabel("rotation (deg)")
        ax2.legend(fontsize=7, loc="upper right")
        ax2.grid(alpha=0.3)

    def _draw_dry(self, fig, w, dry):
        """The dry run: what the scope saw on each output against what was
        meant, and what is left after the fitted delay, scale, gain and
        offset."""
        both = dry["steps"]["both"]
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax)
        for name, col in (("EO1", "#1f77b4"), ("EO2", "#2ca02c")):
            r = both[name]
            t, v = r["trace"]
            meant = awgmod._model(t, w.u[name], w.dt, 0.0, 1.0)
            ch = dry["wiring"][name]
            ax.plot(t * 1e3, meant, color="k", lw=0.6, ls="--")
            ax.plot(t * 1e3, v, color=col, lw=0.9,
                    label=f"AWG CH{awgmod.CHANNELS[name]} on scope CH{ch}: gain "
                          f"{r['gain']:.4f}, delay {r['delay_us']:.1f} us, scale "
                          f"{r['stretch']:.5f}")
            d, st, g, o = r["model"]
            model = g * awgmod._model(t, w.u[name], w.dt, d, st) + o
            tt, rr = self._decimate(t, (v - model) * 1e3, 3000)
            ax2.plot(tt * 1e3, rr, color=col, lw=0.6,
                     label=f"{name}: {r['rms_mV']:.1f} mV rms, idle {r['idle_meas_V']*1e3 if r['idle_meas_V'] is not None else float('nan'):+.0f} "
                           f"mV (meant {r['idle_meant_V']*1e3:+.0f})")
        ax.plot([], [], color="k", lw=0.6, ls="--", label="meant (no delay, gain 1)")
        verdict = "PASSED" if dry["ok"] else "FAILED"
        ax.set_title(f"Dry run {verdict}: {w.label} ({dry['shots']} shots per step)", fontsize=9)
        ax.set_ylabel("scope (V)")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel("seen - fitted (mV)")
        ax2.set_xlabel("time from the trigger (ms)")
        ax2.legend(fontsize=6, loc="upper right")
        ax2.grid(alpha=0.3)
        if dry["problems"]:
            ax2.set_title("; ".join(dry["problems"])[:160], fontsize=7, color="#c00000")

    # -- EOM calibration ---------------------------------------------------------------------------
    def open_calibration(self):
        win = getattr(self, "cal_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        cal = calib.get(self.cfg)
        w = tk.Toplevel(self.root)
        w.title("EOM calibration")
        self.cal_win = w
        ttk.Label(w, justify="left", wraplength=640, text=(
            "AWG volts -> Trek monitor volts -> kV at the EOM -> rotation, per crystal:\n"
            "    monitor V = gain x (AWG V - idle),   kV = monitor V / (monitor V per kV),"
            "   deg = 90 x kV / V90.\nThe pair turns the light by the sum of the two. Used "
            "by the AWG waveforms and fixed-rotation runs (rotation -> AWG volts), the scans' "
            "rotation from the monitors and the ILC-target comparison.")).grid(
            row=0, column=0, columnspan=6, sticky="w", padx=8, pady=(8, 6))
        for j, h in enumerate(("", "AWG -> monitor (V/V)", "monitor V per kV", "V90 (kV)",
                               "deg per AWG V", "deg per monitor V")):
            ttk.Label(w, text=h).grid(row=1, column=j, padx=4, sticky="w")
        self.cal_vars, self.cal_der = {}, {}
        for i, n in enumerate(calib.NAMES):
            ttk.Label(w, text=f"{n}  (AWG CH{i + 1} -> X{i + 1})").grid(
                row=2 + i, column=0, sticky="w", padx=8)
            for j, k in enumerate(("gain", "mon_per_kv", "v90_kv")):
                v = tk.StringVar(value=f"{cal[n][k]:.6g}")
                self.cal_vars[(n, k)] = v
                ttk.Entry(w, textvariable=v, width=11).grid(row=2 + i, column=1 + j, padx=4,
                                                            sticky="w")
                v.trace_add("write", lambda *_: self._cal_derived())
            for j, k in enumerate(("awg", "mon")):
                lab = ttk.Label(w, text="", width=12)
                lab.grid(row=2 + i, column=4 + j, padx=4, sticky="w")
                self.cal_der[(n, k)] = lab
        self.cal_source = (cal.get("source", ""), cal.get("date", ""))
        self.cal_src = CopyLabel(w, text="", foreground="#666", width=90)
        self.cal_src.grid(row=4, column=0, columnspan=6, sticky="w", padx=8, pady=(4, 0))
        cf = ttk.LabelFrame(w, text="Convert - type a value in one box, press Enter")
        cf.grid(row=5, column=0, columnspan=6, sticky="we", padx=8, pady=6)
        self.conv_ch = tk.StringVar(value="EO1")
        ttk.Combobox(cf, textvariable=self.conv_ch, values=calib.NAMES, width=5,
                     state="readonly").pack(side="left", padx=4, pady=4)
        self.conv = {}
        for key, label in (("awg", "AWG V above idle"), ("mon", "monitor V"),
                           ("kv", "kV"), ("deg", "deg")):
            ttk.Label(cf, text=label).pack(side="left", padx=(8, 2))
            v = tk.StringVar()
            self.conv[key] = v
            e = ttk.Entry(cf, textvariable=v, width=9)
            e.pack(side="left")
            e.bind("<Return>", lambda _e, k=key: self._cal_convert(k))
        bf = ttk.Frame(w)
        bf.grid(row=6, column=0, columnspan=6, sticky="we", padx=8, pady=(2, 8))
        ttk.Button(bf, text="1 Sep 2026 values", command=lambda: self._cal_fill(
            calib.DEFAULT)).pack(side="left")
        ttk.Button(bf, text="From EOM-ILC", command=self._cal_from_eomilc).pack(side="left", padx=4)
        ttk.Button(bf, text="Fit to the loaded fixed-rotation run", command=self._cal_from_bias).pack(
            side="left")
        ttk.Button(bf, text="Close", command=w.destroy).pack(side="right")
        ttk.Button(bf, text="Apply and save", command=self._cal_apply).pack(side="right", padx=4)
        self._cal_derived()

    def _cal_read(self):
        cal = calib.get(self.cfg)
        for (n, k), v in self.cal_vars.items():
            cal[n][k] = float(v.get())
        return cal

    def _cal_derived(self):
        try:
            cal = calib.validate(self._cal_read())
        except (ValueError, KeyError) as exc:
            for lab in self.cal_der.values():
                lab.configure(text="?")
            self.cal_src.configure(text=f"! {exc}", foreground="#c00000")
            return
        for n in calib.NAMES:
            self.cal_der[(n, "awg")].configure(text=f"{calib.deg_per_awg_v(cal, n):.4f}")
            self.cal_der[(n, "mon")].configure(text=f"{calib.deg_per_mon_v(cal, n):.4f}")
        src, date = self.cal_source
        self.cal_src.configure(text=f"source: {src} ({date})", foreground="#666")

    def _cal_fill(self, cal, source=None):
        for (n, k), v in self.cal_vars.items():
            v.set(f"{cal[n][k]:.6g}")
        self.cal_source = (source or cal.get("source", ""), cal.get("date", ""))
        self._cal_derived()

    def _cal_convert(self, key):
        try:
            cal = calib.validate(self._cal_read())
            val = float(self.conv[key].get())
        except ValueError as exc:
            self.log(f"Convert: {exc}")
            return
        res = calib.convert(cal, self.conv_ch.get(), **{key: val})
        for k, v in self.conv.items():
            v.set(f"{res[k]:.6g}")

    def _cal_from_eomilc(self):
        try:
            hw.load_eomilc(self.cfg["eomilc_path"])
            self._cal_fill(calib.from_eomilc())
        except Exception as exc:
            self.log(f"Calibration from EOM-ILC: {exc}")

    def _cal_from_bias(self):
        r = getattr(self, "bias_result", None)
        if not r:
            self.log("Load a fixed-rotation run first (Fixed rotations tab: Load...).")
            return
        try:
            cal, rep = calib.from_bias(r, self._cal_read())
        except ValueError as exc:
            self.log(f"Calibration fit: {exc}")
            return
        self.log(f"Calibration fit to {r.get('name')}:")
        for line in rep:
            self.log(f"  {line}")
        self._cal_fill(cal)

    def _cal_apply(self):
        import datetime
        try:
            cal = self._cal_read()
            cal["source"] = self.cal_source[0] or "typed"
            cal["date"] = self.cal_source[1] or datetime.date.today().isoformat()
            calib.apply(calib.validate(cal), self.cfg)
        except ValueError as exc:
            messagebox.showerror("EOM calibration", str(exc), parent=self.cal_win)
            return
        self.save_settings()
        self.log("EOM calibration applied: " + calib.summary(cal))
        self.awg_wave = None
        self.plot_dirty.add(self.fig_awg._frame)
        if self.result:
            self.reanalyse()
        self.draw_visible()

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
            self.log(f"{name} exists - this fixed-rotation run is {new}")
            self.bv["name"].set(new)
            c["bias"]["name"] = new
            name = new
        prov = self._provenance(c)

        try:
            plan["idle"] = self._awg_idle(c)
        except ValueError as exc:
            self.log(f"Fixed rotations: {exc}")
            return
        plan["end"] = "park" if c["awg"].get("never_float", True) else "off"
        if c["awg"].get("require_dry_run", True):
            waves = self._bias_waves(plan)
            s = self.awg_sess
            missing = [w for w in waves if s is None or not s.is_verified(w)]
            if missing:
                messagebox.showwarning(
                    "Dry run first", f"{len(missing)} of the {len(waves)} plateaus in this "
                    f"plan have not passed a dry run on the scope (e.g. "
                    f"{', '.join(w.label for w in missing[:3])}).\n\nFixed rotations tab: 'Dry "
                    f"run on scope' plays every plateau into the scope first.",
                    parent=self.root)
                return

        def go():
            from . import bias as biasmod
            sess = self._awg_session(c)
            self.bias_live = {"points": [], "name": name, "plan": plan}
            run = biasmod.BiasRun(c["outdir"], name, self.link, self.rot, sess.awg, roles,
                                  plan=plan, log=self.log, cancelled=self.stop_flag.is_set,
                                  ask=self.ask_main, progress=self._progress,
                                  on_point=lambda p: self.call(self._bias_point, p),
                                  eomilc=self.awg_eom, ilc_bench=sess.ib, provenance=prov,
                                  session=sess)
            try:
                run.run()
            finally:
                self.call(self._awg_status)
            return run.folder

        def done(folder):
            self.load_bias(folder)
        self.worker(go, done=done)

    def _bias_plan(self, c):
        plan = dict(c["bias"])
        plan.pop("name", None)
        plan["idle"] = self._awg_idle(c)
        return plan

    def _bias_waves(self, plan):
        """One plateau wave per distinct bias of the plan, as the run plays them."""
        from . import bias as biasmod
        p = dict(biasmod.PLAN, **plan)
        seen, out = set(), []
        for b in biasmod.order_biases(biasmod.parse_biases(p["biases"]), p["order"]):
            if b not in seen:
                seen.add(b)
                out.append(biasmod.plateau_wave(b, p))
        return out

    def _show_awg_set(self, title, waves, window=None, found=()):
        """Draw a set of waveforms on the AWG plot tab and log what each is."""
        s = self.awg_sess
        self.awg_set = {"title": title, "waves": waves, "window": window,
                        "found": list(found)}
        w0 = waves[0]
        self.log(f"{title}: {len(waves)} waveforms, {w0.n} points at {w0.dt*1e6:g} us = "
                 f"{w0.period*1e3:.3f} ms each" + (
                     "" if all(abs(w.period - w0.period) < 1e-12 for w in waves)
                     else " (records differ)"))
        for w in waves:
            _, rot = awgmod.predict(w)
            pk = {k: float(np.max(np.abs(u))) for k, u in w.u.items()}
            self.log(f"  {w.label}: CH1 peak {pk['EO1']:.3f} V, CH2 {pk['EO2']:.3f} V, "
                     f"rotation {float(np.max(np.abs(rot))):.2f} deg; dry run "
                     + ("passed" if s is not None and s.is_verified(w) else "not yet"))
        self.plot_dirty.add(self.fig_awg._frame)
        self.nb.select(self.fig_awg._frame)
        self.draw_visible()

    def do_bias_preview(self):
        """Every plateau of the Fixed rotations plan, drawn on the AWG plot
        tab with the window it measures in - before any dry run."""
        c = self.gather()
        from . import bias as biasmod
        try:
            plan = self._bias_plan(c)
            p = dict(biasmod.PLAN, **plan)
            biases = biasmod.parse_biases(p["biases"])
            waves = self._bias_waves(plan)
            window = biasmod.windows(p)[0]
        except (ValueError, OSError) as exc:
            self.log(f"Fixed rotations preview: {exc}")
            return
        if not waves:
            self.log("Fixed rotations preview: no rotations in the plan")
            return
        found = []
        try:
            for _b, _pk, txt in biasmod.check_plateaus(biases, p, self._eom(c)):
                found.append(("INFO", txt))
        except ValueError as exc:
            found.append(("FAIL", str(exc)))
        self.report_checks(found, "Fixed rotations plan", popup=False)
        order = biasmod.order_biases(biases, p["order"])
        self._show_awg_set(f"Fixed rotations plan: {', '.join(f'{b:g}' for b in order[:10])}"
                           f"{' ...' if len(order) > 10 else ''} deg ({p['order']})",
                           waves, window, found)

    def do_seq_preview(self):
        """Every ramp of the AWG tab's sequence, drawn together."""
        c = self.gather()
        try:
            ends, waves, found = self._seq_waves(c)
        except (ValueError, OSError) as exc:
            self.log(f"Sequence preview: {exc}")
            return
        self.report_checks(found, f"AWG sequence ({len(waves)} ramps)", popup=False)
        win = None
        w0 = waves[0]
        try:
            settle = float(self.av["settle_ms"].get())
            if w0.hold and w0.hold[1] - w0.hold[0] > settle * 1e-3 + 0.5e-3:
                win = (w0.hold[0] + settle * 1e-3, w0.hold[1] - 0.2e-3)
        except (ValueError, KeyError):
            pass
        self._show_awg_set(f"AWG sequence: {len(waves)} ramps (X1/X2 deg)", waves, win, found)

    def _draw_awg_set(self, fig, S):
        """A set of waveforms: the AWG outputs (solid CH1 -> X1, dashed CH2
        -> X2) and the rotation each gives, one colour per waveform."""
        from matplotlib.lines import Line2D
        waves = S["waves"]
        n = len(waves)
        cmap = matplotlib.colormaps["viridis"]
        s = self.awg_sess
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax)
        for i, w in enumerate(waves):
            col = cmap(0.9 * i / max(n - 1, 1))
            t = w.t * 1e3
            ax.plot(t, w.u["EO1"], color=col, lw=0.9)
            ax.plot(t, w.u["EO2"], color=col, lw=0.9, ls="--")
            _, rot = awgmod.predict(w)
            ok = s is not None and s.is_verified(w)
            ax2.plot(t, rot, color=col, lw=0.9,
                     label=w.label.split(" (")[0] + (" - dry run passed" if ok else ""))
        if S.get("window"):
            a, b = S["window"]
            ax2.axvspan(a * 1e3, b * 1e3, color="#9ecae1", alpha=0.5, lw=0,
                        label="where it measures")
        bad = [m for lv, m in S.get("found", ()) if lv == "FAIL"]
        w0 = waves[0]
        ax.set_title(S["title"] + (f" - FAILS: {bad[0][:80]}" if bad else
                                   f"; {w0.period*1e3:.3f} ms records, {w0.n} points"),
                     fontsize=9, color="#c00000" if bad else "black")
        ax.set_ylabel("AWG output (V)")
        ax.legend(handles=[Line2D([], [], color="k", lw=0.9, label="CH1 -> X1"),
                           Line2D([], [], color="k", lw=0.9, ls="--", label="CH2 -> X2")],
                  fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False)
        ax2.set_xlabel("time from the trigger (ms)")
        ax2.set_ylabel("rotation (deg, EOM calibration)")
        if n <= 14:
            ax2.legend(fontsize=6, loc="upper right", ncol=2 if n > 7 else 1)
        else:
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, n - 1))
            fig.colorbar(sm, ax=[ax, ax2], label="waveform (in order)")
        ax2.grid(alpha=0.3)

    def do_bias_dry(self):
        """Every plateau of the bias plan through the scope, before the run."""
        if not self.need(ell=False):
            return
        c = self.gather()
        self.save_settings()
        from . import bias as biasmod
        try:
            plan = self._bias_plan(c)
            biasmod.check_plateaus(biasmod.parse_biases(plan["biases"]),
                                   dict(biasmod.PLAN, **plan), self._eom(c))
            waves = self._bias_waves(plan)
            wiring = self._awg_wiring(c)
        except (ValueError, OSError) as exc:
            self.log(f"Fixed rotations dry run: {exc}")
            return
        if not self._dry_confirm(wiring, len(waves)):
            return
        self._run_dry(waves, wiring, c, f"bias plan {plan['biases']}")

    def _bias_point(self, p):
        self.bias_live.setdefault("points", []).append(p)
        from . import bias as biasmod
        self.bias_result = dict(self.bias_live, transfer=biasmod.transfer(
            self.bias_live["points"]))
        self.plot_dirty.add(self.fig_bias._frame)
        self.nb.select(self.fig_bias._frame)
        self.draw_visible()

    def do_load_bias(self):
        d = filedialog.askdirectory(title="A fixed-rotation run folder (holds bias.json)",
                                    parent=self.root, initialdir=self.outdir.get())
        if d:
            self.load_bias(d)

    def load_bias(self, folder):
        from . import bias as biasmod
        try:
            self.bias_result = biasmod.load(folder)
        except (OSError, ValueError) as exc:
            self.log(f"Fixed-rotation run not loaded: {exc}")
            return
        tf = self.bias_result.get("transfer")
        n = len(self.bias_result.get("points", []))
        self._lab_upsert(os.path.dirname(os.path.abspath(folder)),
                         lablog.bias_row(self.bias_result))
        self.log(f"Fixed-rotation run {self.bias_result['name']}: {n} points"
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
            ax.text(0.5, 0.5, "No fixed-rotation run yet - Fixed rotations tab: Start, or Load",
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
                xs, ys = [x[i] for i in lb], [ps[i]["er_lower"] for i in lb]
                ax.plot(xs, ys, "o", mfc="none", color=cols[d_],
                        label="lower bound (Imin unresolved; dotted line up)")
                ax.vlines(xs, ys, [y * 2.5 for y in ys], color=cols[d_], lw=1.0,
                          linestyles=(0, (1, 1.6)))
            cv = [i for i in ok if ps[i].get("malus_ratio")]
            ax.plot([x[i] for i in cv],
                    [(ps[i]["imin"] + ps[i]["fit"]["k"]) / ps[i]["imin"] for i in cv],
                    "x", color=cols[d_], alpha=0.6, label="from the null's curvature")
        self._logy(fig, ax)
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
        self._logy(fig, ax)
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

    def _stray_scale(self, c):
        """The PD setting the stray light is read at, or None: the ramp
        tab's mV/div, 0 V one division below centre (a PD a few mV
        negative stays on screen)."""
        s = c["scan"]
        if not s.get("stray_on", True):
            return None
        pd = next((ch for ch, (r, _n) in cfgmod.channel_roles(c).items() if r == "PD"), None)
        v = float(s.get("stray_vdiv", 0.005) or 0)
        if pd is None or v <= 0:
            return None
        return {"ch": pd, "vdiv": v, "offset": 1.0 * v}

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
            if "stray" in kinds:
                st = an.find_stray(outdir, sg.load_capture, vdiv, exclude=run.folder)
                if st is None:
                    self.log(f"  no earlier stray-light pair (dark and background at a fine "
                             f"V/div) in {outdir}")
                else:
                    got["stray"] = st
                    self.log(f"  stray light: reusing {st['level']*1e3:+.3f} +- "
                             f"{st['sem']*1e3:.3f} mV from {st['source']} (read at "
                             f"{st['vdiv']*1e3:g} mV/div, {st['measured']})")
            for kind in kinds:
                if kind == "stray":
                    continue
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
        self.shots_info = CopyLabel(side, text="", foreground="#666", width=34)
        self.shots_info.pack(anchor="w", pady=(4, 0))
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=right)
        toolbar = NavigationToolbar2Tk(canvas, right)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._canvas, fig._toolbar, fig._ctl, fig._frame = canvas, toolbar, side, frame
        self._hook_save(fig, toolbar)
        canvas.mpl_connect("button_press_event", lambda ev: self._shots_click(ev))
        canvas.mpl_connect("button_press_event", self.copy_coords)
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

    @staticmethod
    def _band(t, lo, hi, n=4000):
        """A band thinned to ~n bins, keeping its envelope (min of lo, max
        of hi per bin): fill_between polygons are not simplified when drawn."""
        if len(t) <= 2 * n:
            return t, lo, hi
        k = len(t) // n
        m = (len(t) // k) * k
        return (t[:m].reshape(-1, k).mean(axis=1), lo[:m].reshape(-1, k).min(axis=1),
                hi[:m].reshape(-1, k).max(axis=1))

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
        self._logy(fig, ax, "level", 1.0)
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
        ttk.Label(rr, text="transmission of").pack(side="left")
        self.find_light = tk.StringVar(value=FIND_LIGHT[0])
        ttk.Combobox(rr, textvariable=self.find_light, values=FIND_LIGHT, width=24,
                     state="readonly").pack(side="left", padx=4)
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        for label, key, w, after in (("window", "window", 11, "ms"), ("line", "line_hz", 4, "Hz"),
                                     ("scan step", "step", 4, "deg")):
            ttk.Label(rr, text=label).pack(side="left", padx=(0, 2))
            self.fv[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.fv[key], width=w).pack(side="left")
            ttk.Label(rr, text=after).pack(side="left", padx=(1, 8))
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        for label, key, w in (("+-deg", "half", 4), ("points", "points", 3), ("shots", "shots", 3)):
            ttk.Label(rr, text=label).pack(side="left", padx=(0, 2))
            self.fv[key] = tk.StringVar()
            ttk.Entry(rr, textvariable=self.fv[key], width=w).pack(side="left", padx=(0, 8))
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=(3, 1))
        self._btn(rr, "Malus scan 0-180", self.do_malus_scan)
        self._btn(rr, "Find and go there", self.do_find_angle, padx=(4, 0))
        self.find_zero_btn = ttk.Button(rr, text="Make it analyzer 0", state="disabled",
                                        command=self.do_find_zero)
        self.find_zero_btn.pack(side="left", padx=4)
        self.find_lbl = CopyLabel(f, text="", foreground="#060", width=47)
        self.find_lbl.pack(anchor="w", padx=6)
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        self.find_preset = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="preset first", variable=self.find_preset).pack(side="left")
        self.find_zoom = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="timebase to the window", variable=self.find_zoom).pack(
            side="left", padx=(6, 0))
        self._btn(rr, "Set scope as ramp scan", self.do_apply_preset, padx=(6, 0))
        rr = ttk.Frame(f)
        rr.pack(fill="x", padx=6, pady=1)
        self.find_off = {}
        for label, kind, default in (("Dark", "dark", "none"),
                                     ("Background", "background", "measure after")):
            ttk.Label(rr, text=label).pack(side="left", padx=(0, 2))
            v = tk.StringVar(value=default)
            ttk.Combobox(rr, textvariable=v, values=ANALYZER_OFFSETS, width=12,
                         state="readonly").pack(side="left", padx=(0, 8))
            self.find_off[kind] = v
        ttk.Label(f, foreground="#666", justify="left", wraplength=330, text=(
            "Static light: nothing ramping, LINE trigger, PD mean over one line period. "
            "Dark/background: measured after, at every V/div used, and subtracted.")).pack(
            anchor="w", padx=6, pady=(2, 4))

    def _run_in_window(self, run, static, win, line_hz, zoom, st, post=None):
        """Worker: run(window) with the scope set for it - static light (LINE
        trigger, one line period), or the record window with the timebase
        zoomed onto it (zoom) or checked to cover it - then post(out, window)
        (the dark / background) in the same scope state, and everything
        changed put back afterwards."""
        from . import bias as biasmod
        post = post or (lambda o, w_, mode: o)
        if static:
            with biasmod.static_light(self.link, line_hz, self.log) as w:
                out = run(w)
                out["static"] = True
                return post(out, w, "static")
        if zoom and win is not None:
            with biasmod.window_timebase(self.link, win, self.log):
                return post(run(win), win, "window")
        self._window_in_record(st, win)
        return post(run(win), win, "window")

    def _analyzer_offsets(self, c, roles, plan):
        """post() for Find and the Malus scan: the dark (PD covered) and / or
        background (beam blocked) at every setting the readings used -
        measured now, with prompts, or the newest stored ones - subtracted
        (background before dark: it includes the stray light)."""
        from . import bias as biasmod
        modes = {k: c["find"].get(f"{k}_mode", d) for k, d in
                 (("dark", "none"), ("background", "measure after"))}
        outdir = c["outdir"]
        b = self.bench

        def post(out, w, light):
            if out["kind"] == "scan":
                sets = {k: tuple(st_) for k, st_ in zip(out["keys"], out["settings"])}
            else:
                sets = {out["keys"][-1]: tuple(out["setting"])}
            kinds = [k for k in ("background", "dark") if modes[k] != "none"]
            if not kinds:
                return biasmod.apply_offsets(out, {}, log=self.log)
            measured = {}
            for kind in ("dark", "background"):
                if modes[kind] != "measure after":
                    continue
                title, ask, undo = self.OFFSET_PROMPTS[kind]
                if not self.ask_main(title, ask + f"\n\n({len(sets)} setting(s); the "
                                     f"analyzer stays where it is.)"):
                    self.log(f"  {kind} skipped")
                    continue
                if b is not None:
                    if kind == "dark":
                        b.covered = True
                    else:
                        b._imax_saved, b.imax = b.imax, 0.0
                try:
                    measured[kind] = biasmod.measure_offsets(
                        self.link, self.rot, roles, sets, w, plan, log=self.log,
                        cancelled=self.stop_flag.is_set, progress=self._progress, label=kind)
                finally:
                    if b is not None:
                        if kind == "dark":
                            b.covered = False
                        else:
                            b.imax = b._imax_saved
                self.ask_main(title, undo)
                try:
                    biasmod.save_offsets(outdir, kind, light, measured[kind], sets)
                except OSError as exc:
                    self.log(f"  {kind} not stored ({exc})")
            picked = biasmod.pick_offsets(outdir, sets, light, measured, kinds)
            for k, v in picked.items():
                self.log(f"  subtract {v['kind']} {v['level']*1e3:+.3f} mV at {k} "
                         f"({v['when']})")
            out = biasmod.apply_offsets(out, picked, log=self.log)
            if out.get("missing"):
                self.log(f"  no {' / '.join(kinds)} for {', '.join(out['missing'])} - those "
                         f"readings stay raw ('measure after' measures them)")
            return out
        return post

    def _find_setup(self, c, zoom=False):
        """(static, window s, line Hz, plan) from the Find settings. The
        acquisition is the ramp scan's - its trigger wait, readout points
        and offset dither - with the Find tab's shots: Find used to wait
        10 s for a trigger, and the spin-echo sequence comes every ~10 s."""
        fcfg = c["find"]
        sc = c["scan"]
        static = fcfg.get("light") == FIND_LIGHT[1]
        win = None
        line_hz = float(fcfg.get("line_hz") or 60.0)
        if not static:
            txt = str(fcfg.get("window", "")).strip()
            if txt:
                a, b = (float(x) * 1e-3 for x in txt.split(":"))
                win = (min(a, b), max(a, b))
        plan = {"shots": int(fcfg.get("shots") or 8),
                "points": int(sc.get("points") or 20000),
                "wait_s": float(sc.get("wait_s") or 10.0),
                "dither_codes": int(sc.get("dither_codes", 3))}
        if static:
            # a line trigger every 16.7 ms: no long wait, and the mean of a
            # 20 ms record needs few points
            plan.update(points=2000, wait_s=2.0)
        elif zoom and win is not None:
            # the record now spans the window: the scan's 100k points would
            # only slow the readout
            plan["points"] = min(plan["points"], 20000)
        return static, win, line_hz, plan

    TIMEBASE = (":TIMebase:SCALe", ":TIMebase:POSition", ":TIMebase:REFerence")

    def _scope_like_scan(self, c, chroles, label, keep_timebase=False):
        """Worker: the scope as a ramp scan takes it - the selected preset
        written and read back (what 'Apply to scope' does), then the scan's
        pre-run settings check; a FAIL asks before going on. keep_timebase:
        leave the timebase (the AWG tab sets it to its own record). Returns
        the settings as they then are."""
        name = c.get("preset")
        pre = cfgmod.all_presets(c).get(name)
        if pre is None:
            self.log(f"{label}: no preset {name!r} - the scope is used as it is")
        else:
            writes = self.link.preset_writes(pre, chroles)
            if keep_timebase:
                writes = {k: v for k, v in writes.items() if k not in self.TIMEBASE}
            bad, errs = self.link.apply_checked(writes)
            self.log(f"{label}: scope set from the preset '{name}' ({len(writes)} settings"
                     + (", every one read back)" if not bad and not errs else ")"))
            for root, (want, got) in bad.items():
                self.log(f"  ! {root}: wrote {want}, scope reads {got}")
            for e in errs:
                self.log(f"  ! scope error: {e}")
        st = self.link.scope.read_settings()
        found = checks.settings_checks(st, self.link.prof, chroles, dict(c["scan"]))
        bad = [f"{lv}: {msg}" for lv, msg in found if lv in ("WARN", "FAIL")]
        for b in bad:
            self.log(f"  {b}")
        if checks.summary(found) == "FAIL" and not self.ask_main(
                "Scope check", f"The scope check before the {label} found:\n\n"
                + "\n\n".join(bad) + "\n\nGo on anyway?"):
            raise hw.Cancelled()
        return st

    def _window_in_record(self, st, win, what="window"):
        """Refuse a window the record does not cover (the timebase decides)."""
        if win is None:
            return
        from .config import record_span
        sc = self.link.scope

        def get(k):
            v = st.get(k)
            return v if v not in (None, "") else sc.get(k)
        t0, t1 = record_span(get(":TIMebase:SCALe"), get(":TIMebase:POSition"),
                             get(":TIMebase:REFerence") or "LEFT")
        if win[0] < t0 - 1e-9 or win[1] > t1 + 1e-9:
            raise RuntimeError(f"the {what} {win[0]*1e3:.2f}..{win[1]*1e3:.2f} ms is not inside "
                               f"the record the scope takes ({t0*1e3:.2f}..{t1*1e3:.2f} ms) - "
                               f"pick a window in it, or a preset whose timebase shows it")

    def do_find_angle(self):
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        fcfg = c["find"]
        roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(c).items()}
        chroles = self.roles()
        use_preset = bool(self.find_preset.get())
        zoom = bool(self.find_zoom.get())
        kind = self.find_kind.get()
        try:
            static, win, line_hz, plan = self._find_setup(c, zoom)
            half = float(fcfg["half"]) if str(fcfg.get("half", "")).strip() else None
            points = int(fcfg["points"]) if str(fcfg.get("points", "")).strip() else None
        except ValueError:
            self.log("Find: window as from:to in ms (e.g. -10:-0.5); numbers elsewhere")
            return

        def go():
            from . import bias as biasmod
            st = (self._scope_like_scan(c, chroles, f"Find {kind}") if use_preset
                  else self.link.scope.read_settings())

            def run(w):
                return biasmod.find_extremum(
                    self.link, self.rot, roles, kind, window_s=w, plan=plan, half_deg=half,
                    points=points, log=self.log, cancelled=self.stop_flag.is_set,
                    ask=self.ask_main, progress=self._progress)
            return self._run_in_window(run, static, win, line_hz, zoom, st,
                                       self._analyzer_offsets(c, roles, plan))

        self.worker(go, done=lambda out: self._find_done(out, c["outdir"]))

    def do_malus_scan(self):
        if not self.need():
            return
        c = self.gather()
        self.save_settings()
        fcfg = c["find"]
        roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(c).items()}
        chroles = self.roles()
        use_preset = bool(self.find_preset.get())
        zoom = bool(self.find_zoom.get())
        try:
            static, win, line_hz, plan = self._find_setup(c, zoom)
            step = float(fcfg.get("step") or 10.0)
            if not 0.5 <= step <= 45:
                raise ValueError
        except ValueError:
            self.log("Malus scan: a window as from:to in ms, a step of 0.5-45 deg")
            return
        angles = np.arange(0.0, 180.0 - 1e-9, step)

        def go():
            from . import bias as biasmod
            st = (self._scope_like_scan(c, chroles, "Malus scan") if use_preset
                  else self.link.scope.read_settings())

            def run(w):
                return biasmod.malus_scan(self.link, self.rot, roles, angles, window_s=w,
                                          plan=plan, log=self.log,
                                          cancelled=self.stop_flag.is_set,
                                          progress=self._progress)
            return self._run_in_window(run, static, win, line_hz, zoom, st,
                                       self._analyzer_offsets(c, roles, plan))

        self.worker(go, done=lambda out: self._find_done(out, c["outdir"]))

    def _find_done(self, out, outdir):
        self.find_result = out
        self._lab_upsert(outdir, lablog.find_row(out))
        if out["kind"] == "scan":
            out["angle"] = out["angle_min"]
            self.find_lbl.configure(
                text=f"Malus scan: maximum at {out['angle_max']:.2f} deg, minimum at "
                     f"{out['angle_min']:.2f} deg; ER {'>' if out['er_lower'] else ''}"
                     f"{out['er']:.0f} from the fit ({out['offset_note']})")
            self.find_zero_btn.configure(state="normal")
            if self.rot is not None:
                self.show_pos(self.rot.position())
        else:
            sub = out.get("level_raw", out["level"]) - out["level"]
            self.find_lbl.configure(
                text=f"{out['kind']} at analyzer {out['angle']:.3f} +- "
                     f"{out['sig']*1e3:.0f} mdeg; level {out['level']*1e3:.3f} mV"
                     + (f" ({out['offset_note']}: {sub*1e3:+.3f} mV)" if out.get("offsets")
                        else " (raw)") + "; the analyzer is there now")
            self.find_zero_btn.configure(state="normal" if out["kind"] == "min" else "disabled")
            self.show_pos(out["angle"])
        self.plot_dirty.add(self.fig_find._frame)
        self.nb.select(self.fig_find._frame)
        self.draw_visible()

    def do_find_zero(self):
        out = getattr(self, "find_result", None)
        if not out or out["kind"] not in ("min", "scan") or self.rot is None:
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
            ax.text(0.5, 0.5, "Nothing found yet - Analyzer tab", ha="center",
                    transform=ax.transAxes, color="#888")
            ax.set_axis_off()
            return
        th = np.array(out["theta"])
        I = np.array(out["I"])
        w = out.get("window")
        if out.get("static"):
            where = "static light, LINE trigger, one line period"
        elif out.get("bias") is not None:
            where = f"AWG hold at {out['bias']:g} deg"
        else:
            where = "whole record" if not w else f"window {w[0]*1e3:.2f}..{w[1]*1e3:.2f} ms"
        ax.errorbar(th, I * 1e3, np.array(out["sem"]) * 1e3, fmt="o", ms=4,
                    label="PD mean per angle")
        if out["kind"] == "scan":
            fig.clear()
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
            Ic = np.asarray(out["I"]) * 1e3
            Sc = np.asarray(out.get("sem_corr", out["sem"])) * 1e3
            vds = np.asarray(out.get("vdivs", [out["vdiv"]] * len(th)))
            xx = np.linspace(0, 180, 721)
            f_ = out["fit"]
            model = (f_["a0"] + f_["c2"] * np.cos(np.deg2rad(2 * xx))
                     + f_["s2"] * np.sin(np.deg2rad(2 * xx))) * 1e3
            cmap = matplotlib.colormaps["viridis"]
            levels = sorted(set(vds))
            for i_, v in enumerate(levels):
                m = vds == v
                col = cmap(i_ / max(len(levels) - 1, 1))
                for a_ in (ax, ax2):
                    a_.errorbar(th[m], Ic[m], Sc[m], fmt="o", ms=4, color=col,
                                label=f"read at {v*1e3:g} mV/div" if a_ is ax else None)
            for a_ in (ax, ax2):
                a_.plot(xx, model, color="k", lw=0.8)
                a_.axvline(out["angle_max"], color="#2ca02c", lw=0.8, ls="--")
                a_.axvline(out["angle_min"], color="#9467bd", lw=0.8, ls="--")
                a_.grid(alpha=0.3, which="both")
            ax.plot([], [], color="k", lw=0.8, label="a0 + B cos 2(theta - psi), weighted")
            ax.set_ylabel("PD (mV)")
            ax.set_title(f"Malus scan ({where}); {out['offset_note']}", fontsize=9)
            ax.legend(fontsize=6, ncol=2, loc="upper right")
            ax.tick_params(labelbottom=False)
            pos = Ic > 0
            ax2.set_yscale("log")
            lo = max(min(np.min(Ic[pos]) if pos.any() else 1e-3, model.min() if model.min() > 0
                         else 1e9) * 0.3, 1e-3)
            ax2.set_ylim(lo, max(Ic.max(), model.max()) * 2)
            ax2.set_ylabel("PD (mV, log)")
            ax2.set_xlabel("analyzer angle (deg)")
            ax2.set_title(f"maximum {out['angle_max']:.2f} deg, minimum {out['angle_min']:.2f} "
                          f"deg; Imax {out['imax']:.4f} V, Imin {out['imin']*1e3:.3f} +- "
                          f"{out['sig_imin']*1e3:.3f} mV, ER {'>' if out['er_lower'] else ''}"
                          f"{out['er']:.0f}", fontsize=8)
            return
        else:
            f_ = out["fit"]
            xx = np.linspace(th.min(), th.max(), 300)
            model = f_["imin"] + f_["k"] * np.sin(np.deg2rad(xx - f_["theta_n"])) ** 2
            ax.plot(xx, (model if out["kind"] == "min" else -model) * 1e3, color="k", lw=0.9)
            ax.axvline(out["angle"] if abs(out["angle"] - th.mean()) < 90 else out["angle"] + 180,
                       color="#d62728", lw=0.8, ls="--")
            ax.set_ylabel(f"PD (mV, raw, at {out['vdiv']*1e3:g} mV/div)")
            ax.set_title(f"{out['kind']} transmission at {out['angle']:.3f} +- "
                         f"{out['sig']*1e3:.0f} mdeg ({where})", fontsize=9)
        ax.set_xlabel("analyzer angle (deg)")
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
        seq = self.result["d"].manifest.get("plan", {}).get("sequence") or c.get("sequence") or {}
        gap = float(seq.get("spacing_ms", 16.667))
        self.log(f"ILC target: legs {gap:g} ms apart (from the "
                 f"{'scan' if self.result['d'].manifest.get('plan', {}).get('sequence') else 'sequence fields'})")

        def go():
            from . import ilc_target
            return ilc_target.compare(
                folder, i["x1"], i["x2"], f_cut=float(i["f_cut"]),
                lock_tol=float(c["analysis"].get("lock_tol", 0.006)),
                pd_delay_us=float(i["pd_delay_us"]), split=float(i["split"]),
                line_ref=i.get("line_ref") or None,
                leg_gap_ms=gap,
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
        self.awg_addr.set(c.get("awg_addr", ""))
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
        self.stray_on.set(bool(s.get("stray_on", True)))
        self.stray_vdiv.set(f"{float(s.get('stray_vdiv', 0.005)) * 1e3:g}")
        for k, v in self.fv.items():
            v.set(str(c["find"].get(k, "")))
        self.find_kind.set(c["find"].get("kind", "min"))
        self.find_light.set(c["find"].get("light", FIND_LIGHT[0]))
        self.find_preset.set(bool(c["find"].get("use_preset", True)))
        self.find_zoom.set(bool(c["find"].get("zoom", True)))
        for k, v in self.find_off.items():
            v.set(c["find"].get(f"{k}_mode", v.get()))
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
        a = c["awg"]
        for k, v in self.av.items():
            v.set(str(a.get(k, "")))
        seq = c.get("sequence") or cfgmod.DEFAULTS["sequence"]
        for k, v in self.seq.items():
            v.set(f"{float(seq.get(k, cfgmod.DEFAULTS['sequence'][k])):g}")
        self.root.after_idle(self._show_sequence)
        for k in self.a_choice:
            self.a_choice[k].set(str(a.get(k, awgmod.DEFAULTS.get(k, ""))))
        self.a_fit_tb.set(bool(a.get("fit_timebase", True)))
        self.a_never.set(bool(a.get("never_float", True)))
        self.a_require.set(bool(a.get("require_dry_run", True)))

    def gather(self):
        """The window's values into self.cfg (validated where it matters)."""
        c = self.cfg
        c["scope_addr"] = self.scope_addr.get().strip()
        c["awg_addr"] = self.awg_addr.get().strip()
        c["ell_port"] = self.ell_port.get().strip()
        c["ell_address"] = self.ell_addr.get().strip() or "0"
        c["simulate"] = bool(self.simulate.get())
        c["autoconnect"] = bool(self.autoconnect.get())
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
        s["stray_on"] = bool(self.stray_on.get())
        if _isnum(self.stray_vdiv.get()) and float(self.stray_vdiv.get()) > 0:
            s["stray_vdiv"] = float(self.stray_vdiv.get()) * 1e-3
        fd = c["find"]
        for k, v in self.fv.items():
            fd[k] = v.get().strip()
        fd["kind"], fd["light"] = self.find_kind.get(), self.find_light.get()
        fd["use_preset"] = bool(self.find_preset.get())
        fd["zoom"] = bool(self.find_zoom.get())
        for k, v in self.find_off.items():
            fd[f"{k}_mode"] = v.get()
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
        seq = c.setdefault("sequence", dict(cfgmod.DEFAULTS["sequence"]))
        for k, v in self.seq.items():
            if _isnum(v.get()) and float(v.get()) >= 0:
                seq[k] = float(v.get())
        a = c["awg"]
        for k, v in self.av.items():
            txt = v.get().strip()
            if k in ("idle1", "idle2", "file1", "file2", "seq_x1", "seq_x2"):
                a[k] = txt
            elif _isnum(txt):
                a[k] = int(float(txt)) if k in ("shots", "dry_shots") else float(txt)
        for k in self.a_choice:
            a[k] = self.a_choice[k].get()
        a["fit_timebase"] = bool(self.a_fit_tb.get())
        a["never_float"] = bool(self.a_never.get())
        a["require_dry_run"] = bool(self.a_require.get())
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

    def _show_sequence(self):
        """The sequence fields under the preset, for a preset built from them."""
        p = cfgmod.all_presets(self.cfg).get(self.preset.get()) or {}
        if p.get("sequence"):
            if not self.seq_row.winfo_ismapped():
                self.seq_row.pack(fill="x", padx=6, pady=1, after=self._seq_anchor)
                self.seq_lbl.pack(anchor="w", padx=6, after=self.seq_row)
            sc = p["scope"]
            from .config import record_span
            t0, t1 = record_span(sc[":TIMebase:SCALe"], sc[":TIMebase:POSition"], "LEFT")
            self.seq_lbl.configure(text=f"record {t0*1e3:+.1f} .. {t1*1e3:+.1f} ms at "
                                        f"{float(sc[':TIMebase:SCALe'])*1e3:g} ms/div "
                                        f"('Apply to scope' writes it)")
        else:
            self.seq_row.pack_forget()
            self.seq_lbl.pack_forget()

    def sequence_changed(self):
        seq = self.cfg.setdefault("sequence", dict(cfgmod.DEFAULTS["sequence"]))
        for k, v in self.seq.items():
            try:
                x = float(v.get())
                if x < 0:
                    raise ValueError
                seq[k] = x
            except ValueError:
                v.set(f"{seq.get(k, cfgmod.DEFAULTS['sequence'][k]):g}")
        self._show_sequence()

    def preset_picked(self):
        self._show_sequence()
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

    def _open_scope(self, c, roles):
        """Worker thread: connect the scope; returns the status text."""
        sg = self.load_sg()
        if sg is None:
            raise RuntimeError("Scope Grab is not loaded")
        if c["simulate"]:
            scope, _ = self.ensure_sim(roles)
        else:
            prof = sg.scope_profiles.get_profile(c["scope_model"])
            scope = hw.share_rm(sg.Scope(prof), sg.pyvisa)
            scope.connect(c["scope_addr"] or None)
        self.link = hw.ScopeLink(scope, log=self.log)
        self.log(f"Scope: {scope.idn.strip()} at {scope.addr}")
        return short_idn(scope.idn)

    def _open_ell(self, c, roles):
        """Worker thread: connect the analyzer mount; (status text, position)."""
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
        return (f"ELL{info['type']} S/N {info['serial']}", pos)

    def do_connect_scope(self):
        c = self.gather()
        roles = self.roles()
        self.worker(lambda: self._open_scope(c, roles),
                    done=lambda txt: self.scope_status.configure(text=txt, foreground="#060"))

    def do_connect_ell(self):
        c = self.gather()
        roles = self.roles()

        def done(out):
            self.ell_status.configure(text=out[0], foreground="#060")
            self.show_pos(out[1])
        self.worker(lambda: self._open_ell(c, roles), done=done)

    def auto_connect(self):
        """On open: the scope and the analyzer, each on its own - one that is
        off or held by another program is logged and the other still comes
        up. 'Connect on open' turns it off."""
        if self.busy or self.link is not None or self.rot is not None:
            return
        c = self.gather()
        roles = self.roles()
        self.log("Connecting on open (untick 'Connect on open' to stop this)...")

        def go():
            out = {}
            try:
                out["scope"] = self._open_scope(c, roles)
            except Exception as exc:
                self.log(f"  scope not connected: {exc}")
            try:
                out["ell"] = self._open_ell(c, roles)
            except Exception as exc:
                self.log(f"  analyzer not connected: {exc}")
            return out

        def done(out):
            if "scope" in out:
                self.scope_status.configure(text=out["scope"], foreground="#060")
            if "ell" in out:
                self.ell_status.configure(text=out["ell"][0], foreground="#060")
                self.show_pos(out["ell"][1])
        self.worker(go, done=done)

    def do_disconnect(self):
        def go():
            self._awg_close()
            if self.link is not None and self.bench is None:
                self.link.scope.close()
            if self.rot is not None and self.bench is None:
                self.rot.close()
            self.link = self.rot = None
            self.bench = None

        def done(_):
            self.scope_status.configure(text="not connected", foreground="#666")
            self.ell_status.configure(text="not connected", foreground="#666")
            self._awg_status()
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
            if unfinished and (man.get("plan") or {}).get("series"):
                base = man["plan"]["series"].get("base", "")
                if not messagebox.askyesno(
                        "Part of an AWG sequence", f"{run.name} is one scan of the AWG "
                        f"sequence {base}: resumed here it would be measured with whatever "
                        f"the AWG plays now, not its own ramp. Resume it with the AWG tab's "
                        f"Start sequence (same scan name).\n\nStart a new scan {new} here "
                        f"instead?", parent=self.root):
                    return
                unfinished = []
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
        offs = [{"kind": k, "target": 0.0} for k in pre]
        stray = self._stray_scale(c) if len(pre) == 2 else None
        if stray:
            # each also read at a fine V/div, in the same prompt: their
            # difference there is the stray light (dark_level). The order
            # puts the fine background next to the fine dark.
            offs = [{"kind": "dark", "target": 0.0}, {"kind": "dark", "target": 0.0,
                                                      "pd_scale": stray},
                    {"kind": "background", "target": 0.0, "pd_scale": stray},
                    {"kind": "background", "target": 0.0}]
        steps = offs + steps
        plan = dict(s, preset=c["preset"], software=f"rampol {__version__}",
                    dark_mode=dm, bg_mode=bm)
        if (cfgmod.PRESETS.get(c["preset"]) or {}).get("sequence"):
            plan["sequence"] = dict(c.get("sequence") or cfgmod.DEFAULTS["sequence"])
        extra = {"zero_deg": float(c["ell_zero_deg"]),
                 "precheck": getattr(self, "last_check", None),
                 "provenance": self._provenance(c)}
        drive = self._awg_drive_info()
        if drive:
            extra["drive"] = drive
            self.log(f"  the EOMs are driven by this window's AWG: {drive['label']} - recorded "
                     f"in the scan (the ILC state files in its provenance are not what plays)")
        run.new(plan, steps, extra=extra)
        self.run = run
        self.log(f"Scan {run.name}: {len(angles)} angles, {len(steps)} steps -> {run.folder}")
        reuse = [k for k, m in (("dark", dm), ("background", bm)) if m == "reuse latest"]
        if bm == "reuse latest" and self._stray_scale(c):
            reuse.append("stray")

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

    # -- rename / correct a scan ------------------------------------------------
    SEQ_FIELDS = (("legs apart", "spacing_ms"), ("motion", "motion_ms"),
                  ("before", "before_ms"), ("after", "after_ms"))

    def open_edit_scan(self):
        """Rename the shown scan and correct what its manifest says about it
        (the preset, the spin-echo sequence, notes). Every correction is kept
        in the manifest's 'edits' with the old value."""
        path = self._scan_path(self.show_scan.get())
        name = os.path.basename(os.path.normpath(path)) if path else ""
        mp = os.path.join(path, f"{name}_scan.json") if path else ""
        if not path or not os.path.isfile(mp):
            self.log("Rename / edit: pick a scan in the Scan box first.")
            return
        with open(mp, encoding="utf-8") as fh:
            man = json.load(fh)
        plan = man.get("plan", {})
        w = tk.Toplevel(self.root)
        w.title(f"Rename / edit - {name}")
        w.transient(self.root)
        self.edit_win = w
        f = ttk.Frame(w, padding=8)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Name").grid(row=0, column=0, sticky="w")
        v_name = tk.StringVar(value=name)
        ttk.Entry(f, textvariable=v_name, width=44).grid(row=0, column=1, sticky="ew", pady=2)
        preview = ttk.Label(f, foreground="#666", text="")
        preview.grid(row=1, column=1, sticky="w")

        def show_name(*_):
            n = scanmod.safe_name(v_name.get())
            preview.configure(text=(f"saved as {n}" if n != v_name.get().strip() else "")
                              + "  (spaces become _ : Scope Grab's Compare box splits at "
                                "spaces)")
        v_name.trace_add("write", show_name)
        show_name()
        ttk.Label(f, text="Preset").grid(row=2, column=0, sticky="w")
        v_preset = tk.StringVar(value=plan.get("preset", ""))
        ttk.Entry(f, textvariable=v_preset, width=44).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(f, text="Sequence (ms)").grid(row=3, column=0, sticky="w")
        sr = ttk.Frame(f)
        sr.grid(row=3, column=1, sticky="w", pady=2)
        seq = plan.get("sequence") or {}
        v_seq = {}
        for label, key in self.SEQ_FIELDS:
            ttk.Label(sr, text=label).pack(side="left", padx=(0, 2))
            v_seq[key] = tk.StringVar(value=f"{seq[key]:g}" if key in seq else "")
            ttk.Entry(sr, textvariable=v_seq[key], width=7).pack(side="left", padx=(0, 6))
        ttk.Label(f, foreground="#666", justify="left", wraplength=420, text=(
            "The spin-echo sequence the scan ran: leg spacing, one motion's length, and the "
            "record kept before the trigger and after the second motion. "
            + ("Not recorded for this scan (before 7 Oct 2026, or not a spin-echo preset): "
               "fill it in to correct it, or leave it blank." if not seq else
               "Recorded when the scan started.")
            + " The ILC-target comparison takes its leg spacing from here.")).grid(
            row=4, column=1, sticky="w")
        ttk.Label(f, text="Notes").grid(row=5, column=0, sticky="nw", pady=(4, 0))
        notes = tk.Text(f, width=52, height=4, wrap="word", font="TkDefaultFont")
        notes.grid(row=5, column=1, sticky="ew", pady=(4, 2))
        notes.insert("1.0", man.get("notes", ""))
        ed = man.get("edits") or []
        hist = "\n".join(f"{e.get('when', '')}  {e.get('field')}: {e.get('from')!r} -> "
                         f"{e.get('to')!r}" for e in ed[-6:]) or "none yet"
        ttk.Label(f, text="Corrections").grid(row=6, column=0, sticky="nw")
        CopyLabel(f, text=hist, foreground="#666", width=60).grid(row=6, column=1, sticky="w")
        msg = ttk.Label(f, foreground="#c00000", text="")
        msg.grid(row=7, column=1, sticky="w")

        def save():
            if self.busy:
                msg.configure(text="Busy - wait for the current operation (or Stop it) first.")
                return
            seq_new = {}
            for _label, key in self.SEQ_FIELDS:
                txt = v_seq[key].get().strip()
                if txt:
                    if not _isnum(txt) or float(txt) < 0:
                        msg.configure(text=f"Sequence {key}: a number of ms >= 0")
                        return
                    seq_new[key] = float(txt)
            if seq_new and len(seq_new) < len(self.SEQ_FIELDS):
                msg.configure(text="Sequence: fill in all four, or none")
                return
            fields = {"plan.preset": v_preset.get().strip(),
                      "plan.sequence": seq_new or None,
                      "notes": notes.get("1.0", "end").strip()}
            try:
                new = self.edit_scan(path, fields, v_name.get())
            except (OSError, ValueError) as exc:
                msg.configure(text=str(exc))
                return
            w.destroy()
            self.refresh_scan_list(select=new)
            self.do_load_shown()
        b = ttk.Frame(f)
        b.grid(row=8, column=1, sticky="e", pady=(6, 0))
        ttk.Button(b, text="Save", command=save).pack(side="left", padx=4)
        ttk.Button(b, text="Cancel", command=w.destroy).pack(side="left")

    def edit_scan(self, path, fields, new_name):
        """Correct the metadata, then rename (folder, files, manifest, the
        lab log, other scans that cite it). Returns the scan's name after.
        Tk thread: it is a few hundred renames, and nothing may be measuring."""
        outdir = os.path.dirname(os.path.normpath(path))
        old = os.path.basename(os.path.normpath(path))
        if self.run is not None and os.path.normcase(self.run.folder) == os.path.normcase(path) \
                and self.busy:
            raise ValueError(f"{old} is being measured")
        scanmod.edit_metadata(path, fields, log=self.log)
        new = scanmod.safe_name(new_name)
        if new != old:
            new = scanmod.rename(outdir, old, new, log=self.log)
            new_folder = os.path.join(outdir, new)
            try:
                n = lablog.rename(outdir, old, new, new_folder)
                if n:
                    self.log(f"  lab log: {n} row(s) updated")
            except OSError as exc:
                self.log(f"  lab log NOT updated ({exc}): its row still says {old}. Close "
                         f"it in Excel and correct the row by hand (the next analysis of "
                         f"{new} adds a row under the new name).")
            # what the window holds that points at the old folder
            self.scan_cache.pop(os.path.normcase(os.path.abspath(path)), None)
            # the Compare picks are re-read from the list (it shows the new name)
            self.cmp_sel, self.cmp_results = [], {}
            lr = self.iv["line_ref"].get().strip()
            if lr and os.path.normcase(os.path.normpath(lr)) == os.path.normcase(os.path.normpath(path)):
                self.iv["line_ref"].set(new_folder)
            if self.run is not None and os.path.normcase(self.run.folder) == os.path.normcase(path):
                self.run = None
            self.log("  Export brief again for a brief that says the new name.")
        return new

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
                     f"with every Apply switch on, a fixed-rotation run finishes or an angle is found.")
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
            "box. The scans compared here are also drawn on the Extinction, "
            "Malus, Angle, Poincaré and Diagnostics tabs when their 'compare "
            "scans' box is ticked.")).pack(anchor="w", pady=(4, 0))
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=right)
        toolbar = NavigationToolbar2Tk(canvas, right)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._canvas, fig._toolbar, fig._ctl, fig._frame = canvas, toolbar, side, frame
        self._hook_save(fig, toolbar)
        canvas.mpl_connect("button_press_event", self.copy_coords)
        self.plot_tabs[frame] = (fig, self.draw_compare)
        self.plot_dirty.add(frame)
        self.fig_cmp = fig
        self._ylog_box(fig, True, "log y (ER)", parent=None)

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
            self._cmp_dirty()
            self.nb.select(self.fig_cmp._frame)
            self.redraw(self.fig_cmp)
        self.worker(go, done=done)

    def do_compare_clear(self):
        self.cmp_sel, self.cmp_results = [], {}
        self.cmp_lb.selection_clear(0, "end")
        self._cmp_dirty()
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
        # the difference is from the shown scan, or from the first compared
        # one when nothing is loaded
        ref = shown if shown is not None else items[0][0]
        diff_on = bool(self.cmp_diff.get()) and len(items) > 1
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
            if diff_on and r is not ref:
                tr = ref["d"].t
                m = (tr >= d.t[0]) & (tr <= d.t[-1])
                if m.sum() > 10:
                    dd = (np.interp(tr[m], d.t, pol["rotation"])
                          - ref["pol"]["rotation"][m]) * 1e3
                    tt, dd = self._decimate(tr[m], self.smooth(dd, tr[m]), 3000)
                    ax2.plot(tt * 1e3, dd, color=c, lw=0.8,
                             label=f"{d.name} - " + ("shown" if ref is shown else ref["d"].name))
            er, ok, us = self.smoothed_er(pol, d.t)
            x = pol["rotation"] if by_rot else d.t * 1e3
            k = max(1, len(x) // 4000)
            ax3.plot(x[::k], np.where(ok, er, np.nan)[::k], color=c, lw=0.6, alpha=0.6)
            cr = [p for p in r.get("direct", []) if p["kind"] == "crossing"]
            for lower in (False, True):
                sel = [p for p in cr if p["lower"] == lower]
                if not sel:
                    continue
                xs = [p["rotation"] if by_rot else p["t_ms"] for p in sel]
                ys = [p["er"] for p in sel]
                ax3.plot(xs, ys, "o", ms=3.5, color=c, mfc=c if not lower else "none",
                         ls="none")
                if lower:
                    ax3.vlines(xs, ys, [y * 2.5 for y in ys], color=c, lw=1.0,
                               linestyles=(0, (1, 1.6)))
        ax1.set_ylabel("rotation from rest (deg)")
        ax1.set_title("Polarization rotation, each scan from its own rest azimuth", fontsize=9)
        ax1.legend(fontsize=7, loc="best")
        ax1.grid(alpha=0.3)
        if diff_on:
            sig = ref["pol"]["sig_psi"] * 1e3
            t3 = ref["d"].t * 1e3
            kk = max(1, len(t3) // 4000)
            ax2.fill_between(t3[::kk], -sig[::kk], sig[::kk], color="0.85", lw=0,
                             label="+-1 SD per sample, shown scan's fit")
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_ylabel("difference (mdeg)")
            ax2.legend(fontsize=7, loc="best")
            ax2.grid(alpha=0.3)
            ax1.tick_params(labelbottom=False)
        self._logy(fig, ax3)
        ax3.set_xlabel("rotation from rest (deg)" if by_rot else "time (ms)")
        if not by_rot:
            ax3.set_xlabel("time (ms)")
            if diff_on:
                ax2.tick_params(labelbottom=False)
        ax3.set_ylabel("extinction ratio")
        ax3.set_title(f"Lines: ER_fit ({us:g} us mean); dots: measured directly at the "
                      f"crossings (hollow with a dotted line up: lower bounds)" if us else
                      "Lines: ER_fit per sample; dots: measured directly at the crossings "
                      "(hollow with a dotted line up: lower bounds)", fontsize=8)
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
        idx = np.linspace(0, len(t) - 1, min(len(t), 2500)).astype(int)
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
        cur, = ax3.plot([0], [0], [0], "o", color="#d62728", ms=7,
                        label="cursor (largest chi until one is set)")
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
        # chi is already a running mean over the Smooth span: every k-th sample
        # draws the same line for a fraction of the cost
        k = max(1, len(tt) // 6000)
        tt, chi_d, ok_d = tt[::k], chi[::k], ok[::k]
        ax.plot(tt, np.where(ok_d, chi_d, np.nan), lw=0.7, color="#1f77b4",
                label=f"chi ({us:g} us mean)" if us else "chi (per sample)")
        ax.plot(tt, np.where(~ok_d, chi_d, np.nan), lw=0.5, alpha=0.35, color="#1f77b4",
                label="upper bound (Imin < 2 sigma)")
        for r_, c_ in self._compared(fig):
            t_r = r_["d"].t
            er_r, ok_r, _ = self.smoothed_er(r_["pol"], t_r)
            chi_r = np.rad2deg(np.arctan(1.0 / np.sqrt(np.maximum(er_r, 1.0))))
            k_r = max(1, len(t_r) // 6000)
            ax.plot(t_r[::k_r] * 1e3, np.where(ok_r, chi_r, np.nan)[::k_r], lw=0.6, color=c_,
                    label=r_["d"].name)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("ellipticity angle chi (deg)")
        ax.set_title("If fully polarized: tan chi = sqrt(Imin / Imax)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
        self._cursor(ax)
        self._poin_tax = ax
        axe = fig.add_subplot(gs[1, 1])
        ph = np.linspace(0, 2 * np.pi, 200)

        def ellipse(jj):
            c_ = np.deg2rad(chi[jj])
            az = np.deg2rad(st["azimuth_deg"][jj] - st["azimuth_deg"][j0])
            x = np.cos(c_) * np.cos(ph)
            y = np.sin(c_) * np.sin(ph)
            return ((x * np.cos(az) - y * np.sin(az), x * np.sin(az) + y * np.cos(az)),
                    ([-np.cos(az), np.cos(az)], [-np.sin(az), np.sin(az)]))
        (ex, ey), (axx, axy) = ellipse(j0)
        axe.plot(ex, ey, color="0.5", ls="--", lw=1.0, label="rest")
        axe.plot(axx, axy, color="0.5", lw=0.5, ls=":")
        el, = axe.plot(ex, ey, color="#d62728", lw=1.0, label="cursor")
        ea, = axe.plot(axx, axy, color="#d62728", lw=0.5, ls=":")
        axe.set_aspect("equal")
        axe.set_xlim(-1.1, 1.1)
        axe.set_ylim(-1.1, 1.1)
        axe.axhline(0, color="k", lw=0.4)
        axe.axvline(0, color="k", lw=0.4)
        etitle = axe.set_title("", fontsize=8)
        axe.set_xlabel("rest polarization direction")
        axe.legend(fontsize=6, loc="lower right")

        def update():
            j = (int(np.argmin(np.abs(t - self.cursor_t))) if self.cursor_t is not None else jw)
            cur.set_data_3d([st["s1"][j]], [st["s2"][j]], [st["s3"][j]])
            (ex_, ey_), (axx_, axy_) = ellipse(j)
            el.set_data(ex_, ey_)
            ea.set_data(axx_, axy_)
            az_j = st["azimuth_deg"][j] - st["azimuth_deg"][j0]
            etitle.set_text(f"{t[j]*1e3:.3f} ms: azimuth {az_j:+.2f} deg\n"
                            f"chi {'<' if not ok[j] else ''}{chi[j]:.2f} deg, ER_fit "
                            f"{'>' if not ok[j] else ''}{er[j]:.0f}")
        fig._cursor_update = update
        update()

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
            self.cursor_dirty.discard(frame)
            fig, draw = self.plot_tabs[frame]
            if fig is None:
                draw()
            else:
                self.redraw(fig, draw)
        elif frame in self.cursor_dirty:
            self.cursor_dirty.discard(frame)
            fig = self.plot_tabs[frame][0]
            if fig is not None:
                self.cursor_refresh(fig)

    def cursor_refresh(self, fig):
        """Move the cursor on a drawn figure without rebuilding it: the
        cursor lines move, a tab's own update (Malus, Poincare) refreshes its
        cursor-dependent artists, and the figure is drawn once with its
        layout held (constrained layout re-measures every label otherwise)."""
        x = None if self.cursor_t is None else self.cursor_t * 1e3
        for ax in getattr(fig, "_cursor_axes", []):
            ln = getattr(ax, "_cline", None)
            if x is None:
                if ln is not None:
                    ln.set_visible(False)
                continue
            if ln is None:
                xl = ax.get_xlim()
                ax._cline = ax.axvline(x, color="#d62728", lw=0.8, ls=":")
                ax.set_xlim(xl)
            else:
                ln.set_xdata([x, x])
                ln.set_visible(True)
        upd = getattr(fig, "_cursor_update", None)
        if upd is not None:
            try:
                upd()
            except Exception as exc:
                self.log(f"cursor: {exc}")
        self.fast_draw(fig)

    @staticmethod
    def fast_draw(fig):
        eng = fig.get_layout_engine()
        try:
            fig.set_layout_engine("none")
            fig._canvas.draw()
        finally:
            try:
                fig.set_layout_engine(eng)
            except Exception:
                pass

    def copy_coords(self, ev):
        """Right-click on a plot: its x, y at the mouse to the clipboard."""
        if getattr(ev, "button", None) != 3 or ev.inaxes is None or ev.xdata is None \
                or getattr(ev.inaxes, "name", "") == "3d":
            return
        txt = f"{ev.xdata:.6g}\t{ev.ydata:.6g}"
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        lx, ly = ev.inaxes.get_xlabel() or "x", ev.inaxes.get_ylabel() or "y"
        self.plot_status.configure(text=f"copied {lx} = {ev.xdata:.6g}, {ly} = {ev.ydata:.6g}",
                                   foreground="#060")

    def redraw(self, fig, draw=None):
        draw = draw or self.plot_tabs[fig._frame][1]
        fig.clear()
        fig._cursor_axes, fig._cursor_update = [], None
        if fig._frame in getattr(self, "free_tabs", ()):
            try:
                draw(fig)
            except Exception as exc:
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(0.02, 0.5, f"Could not draw: {exc}", transform=ax.transAxes)
                ax.set_axis_off()
                self.log(f"draw: {exc}")
        elif not self.result and hasattr(fig, "_cmp_on") and self._compared():
            # nothing loaded but scans compared: the first stands in for the
            # shown one, the others are drawn against it
            first = self._compared()[0][0]
            self.result, fig._force_cmp = first, True
            try:
                draw(fig)
            except Exception as exc:
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(0.02, 0.5, f"Could not draw: {exc}", transform=ax.transAxes)
                ax.set_axis_off()
                self.log(f"draw: {exc}")
            finally:
                self.result, fig._force_cmp = None, False
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
        if getattr(ev, "button", 1) != 1:
            return
        if ev.inaxes is None or ev.xdata is None or fig._toolbar.mode:
            return
        if fig is self.fig_ext and self.ext_x.get() != "time":
            return
        if fig is self.fig_poin and ev.inaxes is not getattr(self, "_poin_tax", None):
            return
        self.cursor_t = ev.xdata * 1e-3
        self.cursor_var.set(f"{ev.xdata:.3f}")
        self.cursor_moved()

    def cursor_moved(self):
        frames = {self.fig_malus._frame, self.fig_map._frame, self.fig_angle._frame,
                  self.fig_ext._frame, self.fig_poin._frame, self.fig_traces._frame}
        self.cursor_dirty |= frames - self.plot_dirty
        self.plot_dirty.add(self.fig_build._frame)
        self.draw_visible()

    def set_cursor_text(self):
        try:
            self.cursor_t = float(self.cursor_var.get()) * 1e-3
        except ValueError:
            return
        self.cursor_moved()

    def smooth_samples(self, t):
        try:
            us = max(float(self.smooth_us.get()), 0.0)
        except ValueError:
            us = 0.0
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        return us, max(1, int(round(us * 1e-6 / dt)))

    def smooth(self, y, t):
        """Running mean over the Smooth box's span (edges padded): the same
        window as np.convolve(..., 'same'), from a cumulative sum - O(N)
        whatever the span."""
        _, n = self.smooth_samples(t)
        if n <= 1:
            return y
        y = np.asarray(y, float)
        yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
        if not np.all(np.isfinite(yp)):
            return np.convolve(yp, np.ones(n) / n, mode="same")[n:-n]
        c = np.concatenate([[0.0], np.cumsum(yp)])
        i = np.arange(n, n + len(y))
        return (c[i + (n - 1) // 2 + 1] - c[i - n // 2]) / n

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
        """The cursor line on a time axis, registered so a cursor move
        updates it in place (cursor_refresh) instead of redrawing."""
        ax._cline = None
        if self.cursor_t is not None:
            ax._cline = ax.axvline(self.cursor_t * 1e3, color="#d62728", lw=0.8, ls=":")
        fig = ax.figure
        if not hasattr(fig, "_cursor_axes"):
            fig._cursor_axes = []
        fig._cursor_axes.append(ax)

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
        self._logy(fig, ax, "level", 1e-3)
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
            # block means over ~4000 columns: a 19 x 100k mesh took 2.4 s to
            # draw on test-4 and the screen has ~1000 pixels across anyway
            n = max(1, len(t) // 4000)
            m = (len(t) // n) * n
            if n > 1:
                I = I[:, :m].reshape(len(th), -1, n).mean(axis=2)
                tb = t[:m].reshape(-1, n)
                tm = np.r_[tb[:, 0], tb[-1, -1]]
            else:
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
        th = pol["theta"]
        thw = wrap_angle(th)
        thr = np.deg2rad(th)
        g = np.linspace(-5, 355, 721)
        gr = np.deg2rad(g)
        zero_k, zero_g = np.zeros(len(th)), np.zeros(len(g))
        ax = fig.add_subplot(211)
        pts, = ax.plot(thw, zero_k, "o", ms=4, color="#1f77b4", label="measured")
        mod, = ax.plot(g, zero_g, color="k", lw=0.9, label="a0 + 2-theta terms")
        full = None
        if "c1" in pol:
            full, = ax.plot(g, zero_g, color="#ff7f0e", lw=0.7, ls="--",
                            label="with 1- and 4-theta terms")
        ov = []
        for r_, c_ in self._compared(fig):
            ln, = ax.plot([], [], "s", ms=3.5, color=c_, mfc="none", label=r_["d"].name)
            ov.append((r_, ln))
        self._logy(fig, ax, "level", 1.0)
        ax.set_ylabel("PD - dark (mV)")
        title = ax.set_title("")
        txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=7,
                      family="monospace")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        ax2 = fig.add_subplot(212, sharex=ax)
        rpts, = ax2.plot(thw, zero_k, "o", ms=3, color="#1f77b4")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set_xlabel("analyzer angle (deg)")
        ax2.set_ylabel("residual to 2-theta fit (mV)")
        ax2.grid(alpha=0.3)

        def update():
            tc = self.cursor_t if self.cursor_t is not None else t[len(t) // 2]
            j = int(np.argmin(np.abs(t - tc)))
            y = pol["I"][:, j]
            a0, c2, s2 = pol["a0"][j], pol["c2"][j], pol["s2"][j]
            model = a0 + c2 * np.cos(2 * gr) + s2 * np.sin(2 * gr)
            pts.set_ydata(y * 1e3)
            mod.set_ydata(model * 1e3)
            if full is not None:
                extra = sum(pol[k][j] * f for k, f in
                            (("c1", np.cos(gr)), ("s1", np.sin(gr)),
                             ("c4", np.cos(4 * gr)), ("s4", np.sin(4 * gr))))
                full.set_ydata((model + extra) * 1e3)
            fit_at = a0 + c2 * np.cos(2 * thr) + s2 * np.sin(2 * thr)
            rpts.set_ydata((y - fit_at) * 1e3)
            for r_, ln in ov:
                p_ = r_["pol"]
                tr = r_["d"].t
                if tr[0] <= t[j] <= tr[-1]:
                    j2 = int(np.argmin(np.abs(tr - t[j])))
                    ln.set_data(wrap_angle(p_["theta"]), p_["I"][:, j2] * 1e3)
                else:
                    ln.set_data([], [])
            title.set_text(f"Transmission vs analyzer angle at t = {t[j] * 1e3:.3f} ms ({d.name})")
            txt.set_text(f"psi = {pol['psi'][j]:+.3f} +- {pol['sig_psi'][j] * 1e3:.1f} mdeg\n"
                         f"Imax = {pol['imax'][j] * 1e3:.1f} mV, Imin = {pol['imin'][j] * 1e3:.2f} "
                         f"+- {pol['sig_imin'][j] * 1e3:.2f} mV\nER_fit = "
                         f"{'>' if pol['er_lower'][j] else ''}{pol['er'][j]:.0f}, visibility "
                         f"{pol['vis'][j]:.5f}")
            for a in (ax, ax2):
                a.relim()
                a.autoscale_view()
        fig._cursor_update = update
        update()

    def draw_angle(self, fig):
        res = self.result
        pol, d = res["pol"], res["d"]
        t = d.t * 1e3
        ax = fig.add_subplot(211)
        rot, sig = pol["rotation"], pol["sig_psi"]
        ax.fill_between(*self._band(t, rot - sig, rot + sig), color="#1f77b4", alpha=0.3,
                        lw=0)
        ax.plot(t, rot, color="#1f77b4", lw=0.9, label=f"measured ({d.name})")
        ov = self._compared(fig)
        if res["mon"] is not None:
            # the monitors' green would read as a compared scan
            ax.plot(t, res["mon"][0], color="#2ca02c" if not ov else "0.4", lw=0.8, ls="--",
                    label="from Trek monitors (zero at rest, like the light)")
        for r_, c_ in ov:
            tt_, rr_ = self._decimate(r_["d"].t, r_["pol"]["rotation"], 4000)
            ax.plot(tt_ * 1e3, rr_, color=c_, lw=0.8, label=r_["d"].name)
        ax.set_ylabel("rotation from rest (deg)")
        ax.set_title(f"Polarization rotation vs time (rest azimuth {pol['psi_rest']:+.3f} deg "
                     f"in the analyzer frame)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        self._cursor(ax)
        ax2 = fig.add_subplot(212, sharex=ax)
        if res["mon"] is not None:
            ax2.plot(t, res["mon"][1] * 1e3, color="#2ca02c" if not ov else "#1f77b4", lw=0.7,
                     label="measured - monitor prediction" + (f" ({d.name})" if ov else ""))
        for r_, c_ in ov:
            if r_.get("mon") is not None:
                tt_, rr_ = self._decimate(r_["d"].t, self.smooth(r_["mon"][1], r_["d"].t) * 1e3,
                                          4000)
                ax2.plot(tt_ * 1e3, rr_, color=c_, lw=0.7, label=f"{r_['d'].name} (smoothed)")
        ax2.plot(t, sig * 1e3, color="k", lw=0.6, ls="--", label="+-1 SD of the fit")
        ax2.plot(t, -sig * 1e3, color="k", lw=0.6, ls="--")
        ax2.set_ylabel("difference (mdeg)")
        ax2.set_xlabel("time (ms)")
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(alpha=0.3)
        self._cursor(ax2)

    def draw_extinction(self, fig):
        """Extinction ratio along the ramp, each measurement method a marker,
        each transport (leg) a colour, +-1 sigma error bars, lower bounds as
        a marker with a dotted line going up from it. 'What are these?' explains
        every family (ER_HELP). Compared scans: one colour each."""
        res = self.result
        pol, d = res["pol"], res["d"]
        by_rot = self.ext_x.get() == "rotation"
        x = pol["rotation"] if by_rot else d.t * 1e3
        ax = fig.add_subplot(111)
        show = {k: v.get() for k, v in self.ext_show.items()}
        log = self._logy(fig, ax)
        ov = self._compared(fig)
        segs = an.segments(pol["t"], pol["rotation"])
        legs = sorted({an.leg_of(segs, s["t0"]) for s in segs if s["base"] in ("up", "down")})
        leg_col = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd"}
        # leg shading on the time axis
        if not by_rot:
            for s in segs:
                if s["base"] in ("up", "down"):
                    ax.axvspan(s["t0"] * 1e3, s["t1"] * 1e3, lw=0, alpha=0.06,
                               color=leg_col.get(an.leg_of(segs, s["t0"]), "0.5"))
        fit_top = 0.0
        if show["fit"]:
            for r_, c_ in [(res, None)] + ov:
                p_ = r_["pol"]
                er, ok, us = self.smoothed_er(p_, r_["d"].t)
                xx_ = p_["rotation"] if by_rot else r_["d"].t * 1e3
                k = max(1, len(xx_) // 6000) if c_ else 1
                ax.plot(xx_[::k], np.where(ok, er, np.nan)[::k], color=c_ or "0.55",
                        lw=0.6, alpha=0.5 if c_ else 1, zorder=1)
                if c_ is None:
                    ax.plot(x, np.where(~ok, er, np.nan), color="0.55", lw=0.5, alpha=0.35,
                            zorder=1)
                if ok.any():
                    fit_top = max(fit_top, float(np.percentile(er[ok], 99.5)))
            dr = pol.get("drift_resid")
            if dr:
                ax.axhline(1 / dr, color="0.55", lw=0.8, ls=":", zorder=1)

        def pts(family, r):
            """[(x, er, (lo, hi), lower, leg, direction)] for a method of result r."""
            p_, d_ = r["pol"], r["d"]
            sg = segs if r is res else an.segments(p_["t"], p_["rotation"])
            out = []
            if family == "crossing" or family == "static":
                for p in r.get("direct", []):
                    if p["kind"] != family or p.get("offset_limited"):
                        continue
                    t_s = p["t_ms"] * 1e-3
                    e = an.er_sigma(p["er"], p["imin_mV"], p["sig_mV"], p["imax_V"] * 1e3,
                                    p.get("sig_imax_V", 0.0) * 1e3)
                    out.append((p["rotation"] if by_rot else p["t_ms"], p["er"], e,
                                p["lower"], an.leg_of(sg, t_s),
                                an.direction(sg, t_s) if family == "crossing" else "static"))
            elif family == "dip":
                for p in r["dips"]:
                    e = an.er_sigma(p["er"], p["imin"], p["sig_imin"])
                    out.append((p["rotation"] if by_rot else p["t"] * 1e3, p["er"], e,
                                p["er_lower"], an.leg_of(sg, p["t"]),
                                an.direction(sg, p["t"])))
            elif family == "refine":
                for rf in r["refine"]:
                    if "er" not in rf:
                        continue
                    m = (d_.t >= rf["t0"]) & (d_.t <= rf["t1"])
                    tm = 0.5 * (rf["t0"] + rf["t1"])
                    e = an.er_sigma(rf["er"], rf["imin"], rf["sig_imin"])
                    out.append((float(np.mean(p_["rotation"][m])) if by_rot else tm * 1e3,
                                rf["er"], e, rf["er_lower"], an.leg_of(sg, tm), "static"))
            if not show["lower"]:
                out = [o for o in out if not o[3]]
            return out

        fams = [("crossing", "o", "direct", "Measured at a crossing: Imin and Imax both read"),
                ("dip", "^", "dips", "Dip fit: Imin fitted around the crossing, Imax from the fit"),
                ("static", "s", "direct", "Measured, static: analyzer angle nearest crossed"),
                ("refine", "D", "refine", "Null refine: angles stepped around crossed")]
        # everything to draw first, so a lower bound's dotted line can be sized to the axis
        todo = []
        n_fam = {}
        for r_, c_ in [(res, None)] + ov:
            for fam, mk, key, _lbl in fams:
                if not show[key]:
                    continue
                P = pts(fam, r_)
                if r_ is res:
                    n_fam[fam] = len(P)
                todo += [(mk, c_, *q) for q in P]
        vals = [q[3] for q in todo]
        top = max(vals + [fit_top, 1.0])
        # a lower bound's dotted line: a factor 2.5 up on a log axis, 7 % of
        # the axis on a linear one. Not an arrow: its head read as a triangle,
        # the dip fit's marker (7 Oct 2026)
        stub = (lambda v: 1.5 * v) if log else (lambda v: 0.07 * top)
        n_lower = 0
        for mk, c_, xx, er_, (lo, hi), lower, leg, how in todo:
            col = c_ or leg_col.get(leg, "0.3")
            fill = dict(mfc=col) if how == "away" else (
                dict(mfc="white") if how == "back" else
                dict(mfc=col, fillstyle="bottom", markerfacecoloralt="white"))
            ms, lw, al = (4.5, 0.8, 1.0) if c_ is None else (3.5, 0.6, 0.75)
            if lower:
                n_lower += 1
                ax.plot([xx, xx], [er_, er_ + stub(er_)], color=col, lw=1.1, ls=(0, (1, 1.6)),
                        alpha=al, zorder=2)
                ax.plot([xx], [er_], mk, ms=ms, color=col, mec=col, alpha=al, zorder=3, **fill)
            else:
                hi_ = hi if np.isfinite(hi) else er_
                ax.errorbar([xx], [er_], yerr=[[lo], [hi_]], fmt=mk, ms=ms, color=col,
                            mec=col, elinewidth=lw, capsize=1.5, alpha=al, zorder=3, **fill)
        if show["direct"]:
            off = [p for p in res.get("direct", []) if p.get("offset_limited")]
            for p in off:
                ax.plot([p["rotation"] if by_rot else p["t_ms"]], [p["er"]], "s", ms=4.5,
                        mfc="none", mec="0.6", zorder=2)
        if log:
            lows = [q[3] - q[4][0] for q in todo if not q[5]] + vals
            floor = min([v for v in lows if v > 0] + [top])
            ax.set_ylim(max(floor * 0.5, 1.0), max(top * 2.8, floor * 10))
        else:
            ax.set_ylim(0, top * 1.12)
        lim = self.cfg["analysis"]["polarizer_er"]
        on_scale = log and lim <= 10 * ax.get_ylim()[1]
        if on_scale:
            ax.axhline(lim, color="k", lw=0.8, ls="--")
        # the key: methods (black), then legs (colour), then the lines
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        H = []
        for fam, mk, key, lbl in fams:
            if show[key] and n_fam.get(fam):
                H.append(Line2D([], [], marker=mk, color="k", ls="none", ms=5,
                                label=f"{lbl} ({n_fam[fam]})"))
        if show["direct"] and any(p.get("offset_limited") for p in res.get("direct", [])):
            H.append(Line2D([], [], marker="s", mfc="none", mec="0.6", ls="none", ms=5,
                            label="Static, angle too far from crossed (not the light's ER)"))
        H.append(Line2D([], [], marker="o", mfc="k", mec="k", ls="none", ms=5,
                        label="filled: on a ramp out from rest"))
        H.append(Line2D([], [], marker="o", mfc="white", mec="k", ls="none", ms=5,
                        label="hollow: on the ramp back toward rest"))
        H.append(Line2D([], [], marker="o", mfc="k", fillstyle="bottom",
                        markerfacecoloralt="white", mec="k", ls="none", ms=5,
                        label="half: rotation standing still (rest, hold, after)"))
        if n_lower:
            H.append(Line2D([], [], marker="$\u22ee$", color="k", ls="none", ms=9,
                            label=f"dotted line up from a marker: lower bound, the ER is "
                                  f"above it (Imin < 2 sigma; {n_lower})"))
        for leg in legs or [1]:
            H.append(Patch(color=leg_col.get(leg, "0.3"), label=f"leg {leg}"
                           + (" (first transport)" if leg == 1 and len(legs) > 1 else
                              " (second transport)" if leg == 2 else "")))
        for r_, c_ in ov:
            H.append(Patch(color=c_, label=f"compared: {r_['d'].name}"))
        if show["fit"]:
            H.append(Line2D([], [], color="0.55", lw=0.8,
                            label="per-sample Malus fit (ER_fit)"))
            if pol.get("drift_resid"):
                H.append(Line2D([], [], color="0.55", lw=0.8, ls=":",
                                label=f"ER_fit's drift limit ({1 / pol['drift_resid']:.0f})"))
        if on_scale:
            H.append(Line2D([], [], color="k", lw=0.8, ls="--",
                            label=f"analyzer's own ER ({lim:.1e})"))
        else:
            ax.text(0.99, 0.01, f"analyzer's own ER at 843 nm: {lim:.1e}, off scale",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#666")
        ax.legend(handles=H, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
                  title="+-1 sigma bars; 'What are these?' explains", title_fontsize=7)
        ax.set_xlabel("rotation from rest (deg)" if by_rot else "time (ms)")
        ax.set_ylabel("extinction ratio Imax / Imin")
        ax.set_title(f"Extinction ratio along the ramp ({d.name}"
                     + (f", with {len(ov)} compared" if ov else "") + ")")
        ax.grid(alpha=0.3, which="both")
        if not by_rot:
            self._cursor(ax)

    def show_er_help(self):
        win = getattr(self, "er_help_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        w = tk.Toplevel(self.root)
        w.title("Extinction ratio: what each point is")
        self.er_help_win = w
        txt = tk.Text(w, wrap="word", width=96, height=34, font=("TkDefaultFont", 9),
                      relief="flat", padx=10, pady=8)
        txt.insert("1.0", ER_HELP)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)
        ttk.Button(w, text="Close", command=w.destroy).pack(pady=(0, 8))

    def draw_diagnostics(self, fig):
        """Four checks on the scan, each with a line under it saying what it
        shows and what it should look like."""
        res = self.result
        pol, d = res["pol"], res["d"]
        plan = d.manifest.get("plan", {})
        t = d.t * 1e3
        subs = fig.subfigures(2, 2)

        def panel(k):
            return subs.flat[k].add_subplot(111)

        def caption(k, text):
            # wrap=True wraps at the panel's own edge, so it follows the window
            subs.flat[k].supxlabel(text, x=0.01, ha="left", fontsize=7, color="#444",
                                   wrap=True)
        ax = panel(0)
        dr = pol.get("drift_resid")
        caption(0, f"The analyzer goes back to {float(plan.get('ref_angle', 45)):g} deg every "
                   f"{plan.get('ref_every', '?')} angles; each point is that return's mean "
                   f"photodiode level. The light there does not change, so any trend is laser "
                   f"intensity drift: it is interpolated in time and divided out of every angle "
                   f"('drift-correct from refs'). How well the returns predict each other "
                   + (f"({dr * 1e3:.2f}e-3) caps the trustworthy ER_fit at ~{1 / dr:.0f}."
                      if dr else "sets the cap on a trustworthy ER_fit."))
        ov = self._compared(fig)
        for r_, c_ in [(res, "#1f77b4")] + ov:
            p_ = r_["pol"]
            if len(p_["ref_clocks"]):
                c0 = p_["ref_clocks"].min()
                lv = p_["ref_levels"]
                ax.plot((p_["ref_clocks"] - c0) / 60, (lv / lv.mean() - 1) * 1e3, "o-", ms=3,
                        color=c_, label=r_["d"].name)
        if ov:
            ax.legend(fontsize=6)
        ax.set_xlabel("time into scan (min)")
        ax.set_ylabel("ref level - mean (1e-3)")
        ax.set_title("Reference-angle returns")
        ax.grid(alpha=0.3)
        ax = panel(1)
        if pol.get("angle_gain") is not None:
            caption(1, "How much light reaches the photodiode at each analyzer angle "
                       "relative to the mean, fitted along with the Malus law and divided "
                       "out. The mount turning moves the beam on the detector, so a smooth "
                       "once-per-turn variation of a few % is expected; a jump at one angle "
                       "points at that capture.")
            g = pol["angle_gain"]
            order = np.argsort(wrap_angle(pol["theta"]))
            ax.plot(wrap_angle(pol["theta"])[order], (g[order] - 1) * 100, "o-", ms=3,
                    label=d.name)
            for r_, c_ in ov:
                g_ = r_["pol"].get("angle_gain")
                if g_ is not None:
                    o_ = np.argsort(wrap_angle(r_["pol"]["theta"]))
                    ax.plot(wrap_angle(r_["pol"]["theta"])[o_], (g_[o_] - 1) * 100, "o-", ms=3,
                            color=c_, label=r_["d"].name)
            if ov:
                ax.legend(fontsize=6)
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
            caption(1, "Shown with the per-angle transmission switched off: the parts of the "
                       "signal varying once (1-theta) or four times (4-theta) per analyzer "
                       "turn, which the Malus law a0 + B cos 2(theta - psi) cannot make, "
                       "relative to B. 1-theta is the beam walking on the detector; both "
                       "should sit near the noise.")
            ax.set_ylabel("relative amplitude (1e-3)")
            ax.set_xlabel("time (ms)")
            ax.set_title("Harmonics outside the Malus law")
            ax.grid(alpha=0.3)
        ax = panel(2)
        caption(2, "Along the record: the rms of what the per-sample Malus fit leaves over, "
                   "against the median statistical error of one capture block (dashed). On "
                   "the line, the Malus law explains the data to the noise; above it, "
                   "something the model lacks (drift, an angle-dependent transmission, a "
                   "fast change within the averaging) - those times' ER_fit is less certain.")
        sem_med = np.nanmedian([np.nanmedian(s["sem"]["PD"]) for s in pol["steps"]])
        src = pol.get("err_source", "residual")
        ax.plot(t, pol["rms"] * 1e3, lw=0.6,
                label="fit residual rms" if src == "residual"
                else f"error scale from {src}")
        for r_, c_ in ov:
            tt_, rr_ = self._decimate(r_["d"].t, r_["pol"]["rms"] * 1e3, 4000)
            ax.plot(tt_ * 1e3, rr_, lw=0.6, color=c_, alpha=0.8, label=r_["d"].name)
        if np.isfinite(sem_med):
            ax.axhline(sem_med * 1e3, color="k", ls="--", lw=0.8, label="median block SEM")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("mV")
        ax.set_title("Residual vs measurement noise")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = panel(3)
        caption(3, "Per step, in measuring order: where the analyzer landed against where it "
                   "was sent (mdeg, blue; one ELL14 pulse is 2.5 mdeg), and the share of "
                   "photodiode samples that were off the scope screen (red x). Off-screen "
                   "samples are clipped: that step's V/div or offset did not hold the light.")
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
        # the worker's own cleanup (a bias run switching the AWG off) runs
        # before anything is closed under it
        th = self._worker_thread
        if th is not None and th.is_alive():
            th.join(timeout=20.0)
        self._awg_close()
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

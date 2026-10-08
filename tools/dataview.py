"""Data viewer for the polarimetry runs: every integrated number next to the
traces it was computed from.

    python tools/dataview.py [outdir]

Left: the runs in the data folder (bias runs = Fixed rotations / grids /
hold series / compensations / transfer functions; ramp scans = edges,
suite, slew sweeps), grouped by series. Right, for a bias run: the points
table (click a row), the null scan with its fit and the PD traces behind
each scan point, the tracking traces (slope pairs, azimuth, monitors). For a
ramp scan: the PD traces per analyzer angle with the monitors and command,
the harmonic fit's outputs against time, the extinction ratio against time
from the angles near crossed (with those angles' traces), and the summary.
"""
import glob, json, os, sys, traceback
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rampol import analysis as an, bias as biasmod, config as cfgmod, hw  # noqa: E402

K_MON = {"MonX1": 90.0 / 5.1283, "MonX2": 90.0 / 5.1374}


def run_kind(folder):
    if os.path.isfile(os.path.join(folder, "bias.json")):
        return "bias"
    if glob.glob(os.path.join(folder, "*_scan.json")):
        return "scan"
    return None


def series_of(name):
    """A run's series: the name up to its last _X1_, _h, _r, _s or stem part."""
    for tag in ("_X1_", "_h", "_it", "_r"):
        i = name.rfind(tag)
        if i > 0 and (tag != "_h" or name[i + 2:i + 3].isdigit() or name[i + 2:i + 3] == "0"):
            return name[:i]
    if name.startswith(("suite0928_", "N15_s")):
        return name.split("_", 1)[0] if name.startswith("suite") else name.rsplit("_", 1)[0]
    return name


class Plot(ttk.Frame):
    """A matplotlib figure with a toolbar in a frame."""

    def __init__(self, master, **kw):
        super().__init__(master)
        self.fig = Figure(figsize=(9, 6), dpi=96, **kw)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self):
        self.fig.clear()
        return self.fig

    def draw(self):
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()


class Viewer(tk.Tk):
    def __init__(self, outdir):
        super().__init__()
        self.title(f"rampol data viewer - {outdir}")
        self.geometry("1500x900")
        self.outdir = outdir
        cfg = cfgmod.load()
        self.sg = hw.load_scope_grab(cfg["scope_grab_path"])
        self.cache = {}
        self.cur = None            # (kind, folder, data)
        self.sel_point = 0
        self.raw = tk.BooleanVar(value=False)
        self._build()
        self._fill_tree()

    # ---------------------------------------------------------------- layout
    def _build(self):
        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True)
        left = ttk.Frame(pan, width=330)
        pan.add(left, weight=0)
        ttk.Label(left, text="filter").pack(anchor="w", padx=4)
        self.filt = tk.StringVar()
        e = ttk.Entry(left, textvariable=self.filt)
        e.pack(fill="x", padx=4)
        e.bind("<KeyRelease>", lambda _e: self._fill_tree())
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.status = tk.StringVar(value="pick a run")
        ttk.Label(left, textvariable=self.status, wraplength=320).pack(anchor="w", padx=4, pady=4)
        right = ttk.Frame(pan)
        pan.add(right, weight=1)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.tabs = {}
        for name in ("Points", "Null scan", "Tracking", "Traces", "Fit", "ER(t)", "Summary"):
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=name)
            self.tabs[name] = f
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._refresh_tab())
        # points table
        f = self.tabs["Points"]
        cols = ("i", "X1", "X2", "ER", "Imin mV", "Imax V", "null", "creep", "after 1 ms", "tau", "V/div", "win 1st/2nd", "flags")
        self.ptable = ttk.Treeview(f, columns=cols, show="headings", height=14)
        for c in cols:
            self.ptable.heading(c, text=c)
            self.ptable.column(c, width=80, anchor="center")
        self.ptable.pack(fill="x", padx=4, pady=4)
        self.ptable.bind("<<TreeviewSelect>>", self._on_point)
        self.map_plot = Plot(f)
        self.map_plot.pack(fill="both", expand=True)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="map:").pack(side="left")
        self.map_key = tk.StringVar(value="er")
        for k in ("er", "imin", "theta_n", "creep", "after_1ms", "tau"):
            ttk.Radiobutton(row, text=k, value=k, variable=self.map_key, command=self._draw_map).pack(side="left")
        # null scan / tracking / traces / fit / er / summary
        self.null_plot = Plot(self.tabs["Null scan"]); self.null_plot.pack(fill="both", expand=True)
        self.track_plot = Plot(self.tabs["Tracking"]); self.track_plot.pack(fill="both", expand=True)
        tf = self.tabs["Traces"]
        top = ttk.Frame(tf); top.pack(fill="x")
        ttk.Label(top, text="angles (ctrl-click to add):").pack(side="left")
        self.angle_list = tk.Listbox(top, selectmode="extended", height=4, width=60, exportselection=False)
        self.angle_list.pack(side="left", padx=4)
        self.angle_list.bind("<<ListboxSelect>>", lambda _e: self._draw_traces())
        ttk.Checkbutton(top, text="raw (no dark subtraction)", variable=self.raw, command=self._on_raw).pack(side="left", padx=8)
        ttk.Button(top, text="all", command=lambda: (self.angle_list.select_set(0, "end"), self._draw_traces())).pack(side="left")
        ttk.Button(top, text="near crossed", command=self._select_near).pack(side="left")
        self.trace_plot = Plot(tf); self.trace_plot.pack(fill="both", expand=True)
        self.fit_plot = Plot(self.tabs["Fit"]); self.fit_plot.pack(fill="both", expand=True)
        ef = self.tabs["ER(t)"]
        top = ttk.Frame(ef); top.pack(fill="x")
        ttk.Label(top, text="angles within +-").pack(side="left")
        self.er_half = tk.DoubleVar(value=4.0)
        ttk.Entry(top, textvariable=self.er_half, width=5).pack(side="left")
        ttk.Label(top, text="deg of crossed; boxcar").pack(side="left")
        self.er_box = tk.DoubleVar(value=20.0)
        ttk.Entry(top, textvariable=self.er_box, width=6).pack(side="left")
        ttk.Label(top, text="us").pack(side="left")
        ttk.Button(top, text="recompute", command=self._draw_er).pack(side="left", padx=8)
        self.er_plot = Plot(ef); self.er_plot.pack(fill="both", expand=True)
        self.summary = tk.Text(self.tabs["Summary"], wrap="none", font=("Consolas", 9))
        self.summary.pack(fill="both", expand=True)

    def _fill_tree(self):
        self.tree.delete(*self.tree.get_children())
        pat = self.filt.get().strip().lower()
        groups = {}
        for f in sorted(os.listdir(self.outdir)):
            p = os.path.join(self.outdir, f)
            if not os.path.isdir(p):
                continue
            k = run_kind(p)
            if not k or (pat and pat not in f.lower()):
                continue
            groups.setdefault(series_of(f), []).append((f, k))
        for g, items in groups.items():
            if len(items) == 1 and items[0][0] == g:
                self.tree.insert("", "end", iid=items[0][0], text=f"{items[0][0]}  [{items[0][1]}]")
                continue
            node = self.tree.insert("", "end", iid="grp:" + g, text=f"{g} ({len(items)})", open=False)
            for f, k in items:
                self.tree.insert(node, "end", iid=f, text=f"{f}  [{k}]")

    # ---------------------------------------------------------------- loading
    def _on_select(self, _e=None):
        sel = self.tree.selection()
        if not sel or sel[0].startswith("grp:"):
            return
        name = sel[0]
        folder = os.path.join(self.outdir, name)
        kind = run_kind(folder)
        self.status.set(f"loading {name} ...")
        self.update_idletasks()
        try:
            if kind == "bias":
                man = biasmod.load(folder)
                self.cur = ("bias", folder, man)
                self.sel_point = 0
                self._fill_points(man)
            else:
                d = an.load_scan(folder, self.sg.load_capture, cache=self.cache.setdefault(folder, {}))
                d.subtract_dark = not self.raw.get()
                note = ""
                try:
                    pol = an.polarization(d)
                except ValueError as exc:
                    # one or two analyzer angles (the slew sweeps): traces only
                    th, I, sem, steps = an.scan_matrix(d, "scan")
                    pol = {"theta": th, "I": I, "steps": steps, "rotation": None, "no_fit": str(exc)}
                    note = f" - {exc}: traces only, no fit"
                self.cur = ("scan", folder, {"d": d, "pol": pol, "er": None, "direct": None})
                self._fill_angles(d)
            self.status.set(f"{name} ({kind}){note}")
            # land on a tab that applies to this kind of run
            tab = self.nb.tab(self.nb.select(), "text")
            bias_tabs, scan_tabs = ("Points", "Null scan", "Tracking", "Summary"), ("Traces", "Fit", "ER(t)", "Summary")
            if kind == "bias" and tab not in bias_tabs:
                self.nb.select(self.tabs["Points"])
            elif kind != "bias" and tab not in scan_tabs:
                self.nb.select(self.tabs["Traces"])
        except Exception as exc:
            self.status.set(f"{name}: {exc}")
            traceback.print_exc()
            return
        self._refresh_tab()

    def _refresh_tab(self):
        if not self.cur:
            return
        tab = self.nb.tab(self.nb.select(), "text")
        kind = self.cur[0]
        try:
            if kind == "bias":
                {"Points": self._draw_map, "Null scan": self._draw_null, "Tracking": self._draw_track,
                 "Summary": self._bias_summary}.get(tab, lambda: self._note(f"{tab}: for ramp scans"))()
            else:
                {"Traces": self._draw_traces, "Fit": self._draw_fit, "ER(t)": self._draw_er,
                 "Summary": self._scan_summary}.get(tab, lambda: self._note(f"{tab}: for bias runs"))()
        except Exception as exc:
            self.status.set(f"{tab}: {exc}")
            traceback.print_exc()

    def _note(self, text):
        self.status.set(text)

    # ---------------------------------------------------------------- bias runs
    def _fill_points(self, man):
        self.ptable.delete(*self.ptable.get_children())
        for p in man["points"]:
            tk_ = p.get("track") or {}
            flags = []
            if p.get("er") is None and p.get("er_lower"):
                flags.append(f"> {p['er_lower']:.0f}")
            if p.get("track_only"):
                flags.append("track only")
            if not p.get("null_converged", True):
                flags.append("null not converged")
            ratio = self._window_ratio(man, p)
            if ratio and ratio > 1.05:
                flags.append("still recovering in the window")
            self.ptable.insert("", "end", iid=str(p["i"]), values=(
                p["i"], f"{p['x1']:g}", f"{p['x2']:g}", f"{p['er']:.0f}" if p.get("er") else "-",
                f"{p['imin']*1e3:.2f}" if p.get("imin") is not None else "-",
                f"{p['imax']:.3f}" if p.get("imax") else "-", f"{p['theta_n']:.3f}",
                f"{tk_.get('hold_slope_mdeg_ms', float('nan')):+.1f}", f"{tk_.get('after_1ms_mdeg', float('nan')):+.0f}",
                f"{tk_.get('tau_ms', float('nan')):.0f}", f"{(p.get('scan') or {}).get('vdiv', 0)*1e3:g}" if p.get("scan") else "-",
                f"{ratio:.2f}" if ratio else "-", ", ".join(flags)))
        if man["points"]:
            self.ptable.selection_set("0")

    def _window_ratio(self, man, p):
        """The nearest-null raw trace's mean over the window's first half over
        its second: above 1 the scope is still recovering from the bright
        rest light inside the ER window (8 Oct 2026, every 90 deg point)."""
        sc, f = p.get("scan"), p.get("fit")
        if not sc or not f:
            return None
        try:
            npz = np.load(os.path.join(self.cur[1], f"point_{p['i']:02d}.npz"))
        except OSError:
            return None
        th = np.array(sc["theta"]); k = int(np.argmin(np.abs(th - f["theta_n"]))); key = f"null_{th[k]:.3f}"
        if key not in npz.files:
            return None
        t = npz["t"]; w = man["window_s"]; mid = 0.5 * (w[0] + w[1])
        y = npz[key]
        a = y[(t > w[0]) & (t < mid)].mean(); b = y[(t > mid) & (t < w[1])].mean()
        dark = next((v[0] for kk, v in (man.get("dark") or {}).items() if kk.startswith(f"{sc['vdiv']:.6g}V/div")), 0.0)
        return float((a - dark) / (b - dark)) if (b - dark) > 0 else None

    def _on_point(self, _e=None):
        sel = self.ptable.selection()
        if sel:
            self.sel_point = int(sel[0])
            self._refresh_tab()

    def _point(self):
        man = self.cur[2]
        return next(p for p in man["points"] if p["i"] == self.sel_point)

    def _draw_map(self):
        man = self.cur[2]
        pts = man["points"]
        key = self.map_key.get()
        fig = self.map_plot.clear()
        ax = fig.add_subplot(111)
        x1 = np.array([p["x1"] for p in pts]); x2 = np.array([p["x2"] for p in pts])

        def val(p):
            if key in ("creep", "after_1ms", "tau"):
                return (p.get("track") or {}).get({"creep": "hold_slope_mdeg_ms", "after_1ms": "after_1ms_mdeg", "tau": "tau_ms"}[key], np.nan)
            if key == "er":
                return p.get("er") or p.get("er_lower") or np.nan
            if key == "imin":
                return (p.get("imin") or np.nan) * 1e3
            return p.get(key, np.nan)
        v = np.array([val(p) for p in pts], float)
        xs, ys = sorted(set(x1)), sorted(set(x2))
        if len(xs) > 1 and len(ys) > 1 and len(pts) == len(xs) * len(ys):
            g = np.full((len(ys), len(xs)), np.nan)
            for a, b, c in zip(x1, x2, v):
                g[ys.index(b), xs.index(a)] = c
            im = ax.imshow(g, origin="lower", aspect="auto", cmap="viridis",
                           norm=matplotlib.colors.LogNorm() if key == "er" else None,
                           extent=(min(xs) - 7.5, max(xs) + 7.5, min(ys) - 7.5, max(ys) + 7.5))
            fig.colorbar(im, ax=ax)
            for a, b, c in zip(x1, x2, v):
                ax.text(a, b, f"{c:.0f}" if key in ("er", "after_1ms", "tau") else f"{c:.2f}", ha="center", va="center", fontsize=7, color="w")
            ax.set(xlabel="X1 (deg)", ylabel="X2 (deg)")
        else:
            lab = [f"{a:g}/{b:g}" for a, b in zip(x1, x2)]
            ax.plot(range(len(v)), v, "o-")
            ax.set_xticks(range(len(v))); ax.set_xticklabels(lab, rotation=90, fontsize=7)
            if key == "er":
                ax.set_yscale("log")
        p = self._point()
        ax.plot([p["x1"]], [p["x2"]] if len(xs) > 1 and len(ys) > 1 and len(pts) == len(xs) * len(ys) else [val(p)], "r+", ms=16, mew=2)
        ax.set_title(f"{man['name']}: {key} (red cross: the selected point)")
        self.map_plot.draw()

    def _draw_null(self):
        man = self.cur[2]
        p = self._point()
        folder = self.cur[1]
        fig = self.null_plot.clear()
        npz = np.load(os.path.join(folder, f"point_{p['i']:02d}.npz"))
        w = man["window_s"]
        ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222); ax3 = fig.add_subplot(212)
        sc = p.get("scan")
        if sc:
            th = np.array(sc["theta"]); I = np.array(sc["I"]) * 1e3; sem = np.array(sc["sem"]) * 1e3
            ax1.errorbar(th, I, sem, fmt="o", ms=4, label=f"window means at {sc['vdiv']*1e3:g} mV/div")
            f = p.get("fit") or {}
            if f.get("k"):
                tt = np.linspace(th.min(), th.max(), 200)
                ax1.plot(tt, (f["imin"] + f["k"] * np.sin(np.deg2rad(tt - f["theta_n"])) ** 2) * 1e3, "-", lw=1,
                         label=f"fit: Imin {f['imin']*1e3:.3f} +- {f['sig_imin']*1e3:.3f} mV at {f['theta_n']:.3f} deg, rms {f['rms']*1e3:.3f} mV")
            dark = (man.get("dark") or {}).get(biasmod._key((sc["vdiv"], None)) if False else None)
            dark_lv = None
            for key_, val_ in (man.get("dark") or {}).items():
                if key_.startswith(f"{sc['vdiv']:.6g}V/div"):
                    dark_lv = val_
                    ax1.axhline(0, color="k", lw=0.5)
                    ax1.text(th.min(), 0, f"dark {val_[0]*1e3:+.3f} +- {val_[1]*1e3:.3f} mV subtracted", fontsize=7, va="bottom")
            ax1.set(title=f"null scan, point {p['i']} (X1 {p['x1']:g} / X2 {p['x2']:g}): ER {p['er'] or 0:.0f}" +
                    (f" +- {p['sig_er']:.0f}" if p.get("sig_er") else ""), xlabel="analyzer (deg)", ylabel="I (mV, dark-subtracted)")
            ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
            # the traces behind each window mean
            t = npz["t"] * 1e3
            cols = matplotlib.cm.viridis(np.linspace(0, 0.95, max(len(th), 2)))
            for k, (a, c) in enumerate(zip(th, cols)):
                key_ = f"null_{a:.3f}"
                if key_ in npz.files:
                    ax3.plot(t, npz[key_] * 1e3, color=c, lw=0.7, label=f"{a:.2f} deg")
            ax3.axvspan(w[0] * 1e3, w[1] * 1e3, color="orange", alpha=0.15, label="ER window")
            if dark_lv:
                ax3.axhline(dark_lv[0] * 1e3, color="k", lw=0.8, ls="--", label=f"dark {dark_lv[0]*1e3:+.3f} mV (the fit's zero)")
            ax3.set(title="RAW PD traces at the null-scan angles (window mean minus the dark = the points above)", xlabel="t (ms)", ylabel="mV (raw)")
            ax3.legend(fontsize=6, ncol=3); ax3.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "no null scan (track-only point)", ha="center", transform=ax1.transAxes)
        # coarse 4 angles and bright
        t = npz["t"] * 1e3
        for key_ in [k for k in npz.files if k.startswith("coarse_") or k == "bright"]:
            ax2.plot(t, npz[key_], lw=0.7, label=key_)
        ax2.axvspan(w[0] * 1e3, w[1] * 1e3, color="orange", alpha=0.15)
        ax2.set(title=f"coarse scan at {man['coarse'][0]:g} V/div: Imax {p.get('imax') or 0:.3f} V, psi {p.get('psi_coarse') or 0:.2f}", xlabel="t (ms)", ylabel="V")
        ax2.legend(fontsize=6); ax2.grid(alpha=0.3)
        self.null_plot.draw()

    def _draw_track(self):
        man = self.cur[2]
        p = self._point()
        npz = np.load(os.path.join(self.cur[1], f"point_{p['i']:02d}.npz"))
        fig = self.track_plot.clear()
        w = man["window_s"]
        ax1 = fig.add_subplot(311); ax2 = fig.add_subplot(312, sharex=ax1); ax3 = fig.add_subplot(313, sharex=ax1)
        t = npz["t"] * 1e3
        for k in ("slope_hold_plus", "slope_hold_minus", "slope_rest_plus", "slope_rest_minus"):
            if k in npz.files:
                ax1.plot(t, npz[k], lw=0.7, label=k)
        ax1.set(title=f"slope-pair traces (null +- 45 deg), point {p['i']} X1 {p['x1']:g} / X2 {p['x2']:g}", ylabel="V")
        ax1.axvspan(w[0] * 1e3, w[1] * 1e3, color="orange", alpha=0.15); ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
        if "track_t" in npz.files:
            tt = npz["track_t"] * 1e3
            sense = p.get("sense", -1.0)
            ax2.plot(tt, sense * npz["track_dpsi"], lw=0.7, label="light (hold pair, monitors' sense)")
            ax2.plot(tt, npz["track_mon_rot"], lw=0.7, label="monitors (90 V / V90 summed)")
            if "track_dpsi_rest" in npz.files:
                ax2.plot(tt, sense * npz["track_dpsi_rest"], lw=0.7, alpha=0.7, label="light (rest pair)")
            ax2.set(ylabel="rotation (deg)"); ax2.legend(fontsize=6); ax2.grid(alpha=0.3)
            lm = npz["track_lm"] * 1e3
            v = npz["track_valid"].astype(bool)
            ax3.plot(tt[v], lm[v], lw=0.7, label="light - monitors, hold pair (valid)")
            if "track_lm_rest" in npz.files:
                vr = npz["track_valid_rest"].astype(bool)
                ax3.plot(tt[vr], npz["track_lm_rest"][vr] * 1e3, lw=0.7, label="rest pair (valid)")
            tk_ = p.get("track") or {}
            ax3.set(ylabel="mdeg", xlabel="t (ms)", ylim=(-1500, 800),
                    title=f"creep {tk_.get('hold_slope_mdeg_ms', float('nan')):+.1f} mdeg/ms, after 1 ms {tk_.get('after_1ms_mdeg', float('nan')):+.0f}, "
                          f"extreme {tk_.get('after_extreme_mdeg', float('nan')):+.0f} at {tk_.get('after_extreme_ms', float('nan')):.0f} ms, tau {tk_.get('tau_ms', float('nan')):.0f} ms")
            ax3.legend(fontsize=6); ax3.grid(alpha=0.3)
        self.track_plot.draw()

    def _bias_summary(self):
        man = self.cur[2]
        self.summary.delete("1.0", "end")
        pl = {k: v for k, v in man["plan"].items() if k not in ("correction", "nulls", "modulation")}
        txt = [f"{man['name']}  created {man.get('created')}  finished {man.get('finished')}", "",
               "plan: " + json.dumps(pl, indent=0)[:3000], "",
               f"coarse {man.get('coarse')}  null settings {man.get('null_settings')}", f"window {man.get('window_s')}  darks {json.dumps(man.get('dark'))}", ""]
        for c in man.get("limit_checks") or []:
            txt.append("limit check: " + str(c))
        self.summary.insert("1.0", "\n".join(txt))

    # ---------------------------------------------------------------- ramp scans
    def _fill_angles(self, d):
        self.angle_list.delete(0, "end")
        self._angles = []
        for s in d.steps:
            if s["kind"] in ("scan",) and "landed" in s:
                self._angles.append(s)
                self.angle_list.insert("end", f"{s['landed']:.2f}")
        self.angle_list.select_set(0, "end")

    def _on_raw(self):
        if self.cur and self.cur[0] == "scan":
            self._on_select()

    def _select_near(self):
        data = self.cur[2]
        pol = data["pol"]
        if pol.get("no_fit"):
            return
        crossed = (pol["psi_rest"] + 90) % 180
        self.angle_list.select_clear(0, "end")
        for k, s in enumerate(self._angles):
            if abs((s["landed"] - crossed + 90) % 180 - 90) <= 8:
                self.angle_list.select_set(k)
        self._draw_traces()

    def _draw_traces(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        fig = self.trace_plot.clear()
        ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212, sharex=ax1)
        t = d.t * 1e3
        sel = [self._angles[i] for i in self.angle_list.curselection()]
        th = pol["theta"]; I = pol["I"]
        cols = matplotlib.cm.hsv(np.linspace(0, 0.95, max(len(th), 2)))
        for s in sel:
            k = int(np.argmin(np.abs(th - s["landed"])))
            ax1.plot(t, I[k] * 1e3, color=cols[k], lw=0.6, label=f"{th[k]:.1f} deg")
        ax1.set(ylabel="PD (mV, dark-subtracted, drift-corrected)" if d.subtract_dark else "PD (mV, raw)",
                title=f"{d.name}: PD traces per analyzer angle ({len(sel)} of {len(th)} shown)")
        if sel:
            ax1.legend(fontsize=6, ncol=4)
        ax1.grid(alpha=0.3)
        for r in ("MonX1", "MonX2"):
            if r in d.roles:
                m = np.mean([s["v"][r] for s in pol["steps"]], axis=0)
                ax2.plot(t, K_MON[r] * (m - m[d.t < -0.1e-3].mean()), lw=0.7, label=f"{r} (deg)")
        for r in ("CmdX1", "CmdX2"):
            if r in d.roles:
                m = np.mean([s["v"][r] for s in pol["steps"]], axis=0)
                ax2.plot(t, m - m[d.t < -0.1e-3].mean(), lw=0.7, ls=":", label=f"{r} (AWG V)")
        mon_sum = np.zeros(len(t))
        for r in ("MonX1", "MonX2"):
            if r in d.roles:
                m = np.mean([s_["v"][r] for s_ in pol["steps"]], axis=0)
                mon_sum += K_MON[r] * (m - m[d.t < -0.1e-3].mean())
        sign = -1.0 if pol.get("rotation") is not None and np.ptp(mon_sum) > 1 and np.corrcoef(pol["rotation"], mon_sum)[0, 1] < 0 else 1.0
        if pol.get("rotation") is not None:
            ax2.plot(t, sign * pol["rotation"], "k", lw=0.7, label="light rotation (fit, monitors' sense, deg)")
        ax2.set(xlabel="t (ms)", ylabel="deg / V"); ax2.legend(fontsize=6); ax2.grid(alpha=0.3)
        self.trace_plot.draw()

    def _draw_fit(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        if pol.get("no_fit"):
            self.status.set(f"{d.name}: {pol['no_fit']} - only the Traces tab applies")
            fig = self.fit_plot.clear(); fig.text(0.5, 0.5, pol['no_fit'] + ' - only the Traces tab applies', ha='center'); self.fit_plot.draw(); return
        fig = self.fit_plot.clear()
        t = d.t * 1e3
        ax1 = fig.add_subplot(411); ax2 = fig.add_subplot(412, sharex=ax1); ax3 = fig.add_subplot(413, sharex=ax1); ax4 = fig.add_subplot(414, sharex=ax1)
        mon = np.zeros(len(t))
        for r in ("MonX1", "MonX2"):
            if r in d.roles:
                m = np.mean([s["v"][r] for s in pol["steps"]], axis=0)
                mon += K_MON[r] * (m - m[d.t < -0.1e-3].mean())
        rot = pol["rotation"]
        sign = -1.0 if np.corrcoef(rot, mon)[0, 1] < 0 else 1.0
        ax1.plot(t, sign * rot, lw=0.7, label="light (monitors' sense)"); ax1.plot(t, mon, lw=0.7, label="monitors")
        ax1.set(ylabel="rotation (deg)", title=f"{d.name}: harmonic fit against time ({len(pol['theta'])} angles)"); ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
        lm = (sign * rot - mon) * 1e3
        ax2.plot(t, lm - np.median(lm[d.t < -0.1e-3]), lw=0.6, color="C3"); ax2.set(ylabel="light - monitors (mdeg)", ylim=(-2000, 2000)); ax2.grid(alpha=0.3)
        ax3.plot(t, pol["imax"], lw=0.6, label="Imax (V)"); ax3.plot(t, pol["imin"] * 100, lw=0.6, label="Imin x 100 (V)")
        ax3.set(ylabel="V"); ax3.legend(fontsize=6); ax3.grid(alpha=0.3)
        ax4.plot(t, pol["rms"] * 1e3, lw=0.6, label="fit rms (mV)"); ax4.plot(t, pol["sig_psi"] * 1e3, lw=0.6, label="sig psi (mdeg)")
        ax4.set(xlabel="t (ms)", ylabel="mV / mdeg", yscale="log"); ax4.legend(fontsize=6); ax4.grid(alpha=0.3)
        self.fit_plot.draw()

    def _draw_er(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        if pol.get("no_fit"):
            self.status.set(f"{d.name}: {pol['no_fit']} - only the Traces tab applies")
            fig = self.er_plot.clear(); fig.text(0.5, 0.5, pol['no_fit'] + ' - only the Traces tab applies', ha='center'); self.er_plot.draw(); return
        half, box = float(self.er_half.get()), float(self.er_box.get())
        er = an.er_vs_time(d, pol, half_deg=half, box_us=box, stride=1, gains=pol.get("angle_gain"))
        data["er"] = er
        if data.get("direct") is None:
            data["direct"] = an.direct_er(d, pol, gains=pol.get("angle_gain"))
        fig = self.er_plot.clear()
        t = d.t * 1e3
        ax1 = fig.add_subplot(311); ax2 = fig.add_subplot(312, sharex=ax1); ax3 = fig.add_subplot(313, sharex=ax1)
        if len(er["t"]):
            te = er["t"] * 1e3
            par = er["method"] == 1
            ax1.plot(te[par], er["er"][par], ".", ms=2, label=f"parabola through >= 3 angles within +-{half:g} deg")
            ax1.plot(te[~par], er["er"][~par], ".", ms=2, alpha=0.5, label="nearest angle, offset-corrected")
            ax1.plot(te[er["er_lower"]], er["er"][er["er_lower"]], "k_", ms=3, alpha=0.4, label="lower bound (Imin < 2 sigma)")
            ax2.plot(te, er["imin"] * 1e3, ".", ms=2, label="Imin (mV)")
            ax2.plot(te, 2 * er["sig_imin"] * 1e3, "-", lw=0.5, color="gray", label="2 sigma")
        for p in data["direct"]:
            if p["kind"] == "static":
                ax1.plot([p["t_ms"]], [p["er"]], "r*" if not p["lower"] else "r_", ms=10,
                         label=f"static point {p['seg']}: {p['er']:.0f}{' (bound)' if p['lower'] else ''} at angle {p['theta']:.1f}, {p['off_deg']:+.2f} deg off")
        # the floor: the dark level's uncertainty at the PD's V/div, and one 8-bit code
        vdiv = an._pd_vdiv(d, "scan") or 1.0
        dk = (d.manifest.get("borrowed") or {}).get("background") or {}
        sem_dark = float(dk.get("sem") or 0.0)
        floor = max(sem_dark, 0.0)
        if floor > 0 and len(er["t"]):
            ax1.axhline(float(np.nanmedian(er["imax"])) / floor, color="gray", lw=0.8, ls="--",
                        label=f"Imax / dark sem ({sem_dark*1e3:.2f} mV at {vdiv:g} V/div): above this the ER is not resolved")
            ax2.axhline(floor * 1e3, color="gray", lw=0.8, ls="--")
        ax2.axhline(vdiv * 8 / 256 * 1e3, color="gray", lw=0.5, ls=":", label=f"one 8-bit code at {vdiv:g} V/div")
        ax1.set(yscale="log", ylabel="ER", title=f"{d.name}: extinction ratio against time from the angles near crossed (Imax from the fit)")
        ax1.legend(fontsize=6, loc="upper right"); ax1.grid(alpha=0.3, which="both")
        ax2.set(ylabel="mV", yscale="symlog", title="Imin and its 2-sigma noise"); ax2.legend(fontsize=6); ax2.grid(alpha=0.3)
        # the angle traces that feed it
        th = pol["theta"]; I = pol["I"]
        used = sorted({a for _n, angs in er["sets"] for a in angs})
        for a in used:
            k = int(np.argmin(np.abs(th - a)))
            ax3.plot(t, I[k] * 1e3, lw=0.6, label=f"{a:.1f} deg")
        ax3.plot(t, np.where(len(er["t"]) and True, np.interp(d.t, er["t"], er["imin"], left=np.nan, right=np.nan) * 1e3, np.nan), "k", lw=0.8, label="Imin(t)")
        ax3.set(xlabel="t (ms)", ylabel="mV", yscale="symlog", title="the near-crossed angle traces behind it"); ax3.legend(fontsize=6, ncol=4); ax3.grid(alpha=0.3)
        self.er_plot.draw()

    def _scan_summary(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        if pol.get("no_fit"):
            self.status.set(f"{d.name}: {pol['no_fit']} - only the Traces tab applies")
            self.summary.delete('1.0', 'end'); self.summary.insert('1.0', f'{d.name}: ' + pol['no_fit'] + ' - no fit, no ER; see the Traces tab'); return
        self.summary.delete("1.0", "end")
        if data.get("direct") is None:
            data["direct"] = an.direct_er(d, pol, gains=pol.get("angle_gain"))
        lines = [f"{d.name}  ({len(pol['theta'])} angles; dark {pol['dark']*1e3:+.2f} mV; gains {'fitted' if pol.get('angle_gain') is not None else 'none'} {pol.get('gain_note', '')})", "",
                 f"rest azimuth {pol['psi_rest']:.3f} deg; crossed {(pol['psi_rest'] + 90) % 180:.3f}", "",
                 "direct ER points:"]
        for p in data["direct"]:
            lines.append(f"  {p['kind']:8s} {p['seg']:8s} t {p['t_ms']:7.2f} ms  angle {p['theta']:7.2f}  rotation {p['rotation']:+7.2f}  Imin {p['imin_mV']:7.3f} +- {p['sig_mV']:.3f} mV  Imax {p['imax_V']:.3f} V  ER {p['er']:8.0f}{' (bound: ' + p.get('bound_from', '') + ')' if p['lower'] else ''}"
                         + (f"  off {p['off_deg']:+.2f} deg (Imax sin^2 = {p['imin_from_offset_mV']:.3f} mV)" if p["kind"] == "static" else ""))
        man = d.manifest
        lines += ["", "drive: " + json.dumps(man.get("drive", {}), indent=0)[:1500], "", "plan: " + json.dumps({k: v for k, v in man.get("plan", {}).items() if k != "series"}, indent=0)[:1500]]
        self.summary.insert("1.0", "\n".join(lines))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else cfgmod.load()["outdir"]
    app = Viewer(outdir)
    app.mainloop()


if __name__ == "__main__":
    main()

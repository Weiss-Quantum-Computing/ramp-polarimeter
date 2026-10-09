"""Data viewer for the polarimetry runs: every integrated number next to the
traces it was computed from.

    python tools/dataview.py [outdir]

Left: the runs in the data folder, newest first and grouped by series, with
their type, size and date; below it what the selected run is (plan, drive,
note). Right, above the plots, the run's numbers: for a bias run (Fixed
rotations / grids / hold series / compensations / transfer functions) the
points table - pick a point there, or click it in the Overview plot, and every
tab follows it; for a ramp scan the analyzer angles to show and the direct
extinction-ratio points - pick one and the plots mark its time and show its
angle. Only the tabs that apply to the run are shown:

    bias run:   Overview | Null scan | Tracking | Details
    ramp scan:  Traces | Fit | ER(t) | Details

Runs load in the background (a 19-angle scan takes ~10 s) and stay cached;
the toolbar's save button offers <run>_<tab>.png in the run's folder (its
saved_figures/ when that exists). The viewer never writes to the data.
"""
import glob, json, os, queue, sys, threading, time, traceback
import tkinter as tk
from tkinter import ttk
import warnings
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="No artists with labels found")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rampol import analysis as an, bias as biasmod, config as cfgmod, hw  # noqa: E402

K_MON = {"MonX1": 90.0 / 5.1283, "MonX2": 90.0 / 5.1374}
MAX_PTS = 3000             # points per line on screen: longer traces are thinned to a min/max envelope
TABS = {"bias": ("Overview", "Null scan", "Tracking", "Details"),
        "scan": ("Traces", "Fit", "ER(t)", "Details")}
# Overview quantities: key -> (label, unit-bearing axis label)
QUANT = {"er": ("ER", "extinction ratio"),
         "imin": ("Imin", "Imin (mV)"),
         "null": ("crossed angle", "crossed analyzer angle (deg)"),
         "creep": ("drift in hold", "polarization drift in the hold (mdeg/ms)"),
         "after": ("1 ms after fall", "polarization change 1 ms after the fall (mdeg)"),
         "tau": ("relaxation", "relaxation time after the fall (ms)")}


# --------------------------------------------------------------------- helpers
def thin(x, y, n=MAX_PTS):
    """(x, y) reduced to about 2 n points keeping every bin's min and max, so
    spikes survive; untouched when short enough."""
    x, y = np.asarray(x), np.asarray(y, float)
    if len(x) <= 2 * n:
        return x, y
    m = len(x) // n
    k = m * n
    yb = y[:k].reshape(n, m); xb = x[:k].reshape(n, m)
    lo = np.argmin(np.where(np.isnan(yb), np.inf, yb), axis=1); hi = np.argmax(np.where(np.isnan(yb), -np.inf, yb), axis=1)
    r = np.arange(n)
    i1, i2 = np.minimum(lo, hi), np.maximum(lo, hi)
    xs = np.column_stack([xb[r, i1], xb[r, i2]]).ravel(); ys = np.column_stack([yb[r, i1], yb[r, i2]]).ravel()
    return xs, ys


def plot_thin(ax, x, y, *args, **kw):
    xs, ys = thin(x, y)
    return ax.plot(xs, ys, *args, **kw)


def legend(ax, **kw):
    """A legend outside the axes on the right, so it never covers data."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, **kw)


def robust_ylim(ax, *ys, lo=0.5, hi=99.5, pad=0.12, min_span=None):
    """y limits from percentiles of the data, so a few spikes at the edges do
    not flatten the rest (they are drawn, just off the axis)."""
    v = np.concatenate([np.asarray(y, float).ravel() for y in ys if y is not None and len(y)] or [np.array([])])
    v = v[np.isfinite(v)]
    if not len(v):
        return
    a, b = np.percentile(v, [lo, hi])
    if min_span and b - a < min_span:
        c = 0.5 * (a + b); a, b = c - min_span / 2, c + min_span / 2
    s = (b - a) or abs(a) or 1.0
    ax.set_ylim(a - pad * s, b + pad * s)


def message(fig, text):
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10, transform=ax.transAxes)


def fnum(v, fmt, dash="-"):
    try:
        return dash if v is None or not np.isfinite(v) else format(v, fmt)
    except (TypeError, ValueError):
        return dash


def run_kind(folder):
    if os.path.isfile(os.path.join(folder, "bias.json")):
        return "bias"
    if glob.glob(os.path.join(folder, "*_scan.json")):
        return "scan"
    return None


def manifest_path(folder, kind):
    if kind == "bias":
        return os.path.join(folder, "bias.json")
    g = glob.glob(os.path.join(folder, "*_scan.json"))
    return g[0] if g else None


def series_of(name):
    """A run's series: the name up to its last _X1_, _h, _it or _r part."""
    for tag in ("_X1_", "_h", "_it", "_r"):
        i = name.rfind(tag)
        if i > 0 and (tag != "_h" or name[i + 2:i + 3].isdigit() or name[i + 2:i + 3] == "0"):
            return name[:i]
    if name.startswith(("suite0928_", "N15_s")):
        return name.split("_", 1)[0] if name.startswith("suite") else name.rsplit("_", 1)[0]
    return name


def describe(folder, kind):
    """What a run is, from its manifest alone: type, size, date, one-line
    label, the plan's note, and lines for the info box."""
    with open(manifest_path(folder, kind), encoding="utf-8") as fh:
        m = json.load(fh)
    plan = m.get("plan") or {}
    out = {"date": (m.get("created") or "")[5:16].replace("T", " "), "note": plan.get("note", ""), "lines": []}
    L = out["lines"]
    if kind == "bias":
        pts = m.get("points") or []
        out["type"] = ("transfer fn" if plan.get("modulation") else "compensation" if plan.get("correction")
                       else "tracking" if plan.get("track_only") else "ER grid" if plan.get("how") == "grid" else "fixed rot.")
        out["n"] = f"{len(pts)} pt"
        x1 = sorted({p["x1"] for p in pts}); x2 = sorted({p["x2"] for p in pts})
        hold = sorted({p.get("hold_ms") or plan.get("hold_ms") for p in pts} - {None}) or [plan.get("hold_ms")]
        rng = lambda v: (f"{v[0]:g}" if len(v) == 1 else f"{v[0]:g}..{v[-1]:g} ({len(v)})") if v else "-"
        out["label"] = f"X1 {rng(x1)} / X2 {rng(x2)} deg, hold {rng(hold)} ms, {len(pts)} points"
        L += [f"fixed-rotation run ({out['type']}), {len(pts)} points",
              f"X1 {rng(x1)} deg, X2 {rng(x2)} deg", f"hold {rng(hold)} ms, rise {plan.get('rise_ms', '-')} ms, "
              f"tracked {plan.get('track_ms', '-')} ms",
              f"ER window {', '.join(f'{w*1e3:.2f}' for w in m.get('window_s') or [])} ms; {plan.get('shots', '-')} shots",
              f"created {m.get('created', '-')}, finished {m.get('finished', 'no')}"]
        if plan.get("darks_from"):
            L.append(f"darks from {plan['darks_from']}")
    else:
        steps = [s for s in m.get("steps") or [] if s.get("kind") == "scan"]
        angles = sorted({round(s.get("landed", s.get("angle", 0)), 2) + 0.0 for s in steps})
        out["type"] = "ramp scan"
        out["n"] = f"{len(angles)} ang"
        drive = m.get("drive") or {}
        out["label"] = drive.get("label") or plan.get("preset") or ""
        L += [f"ramp scan, {len(angles)} analyzer angles" + (f" ({angles[0]:g}..{angles[-1]:g} deg)" if angles else ""),
              f"drive: {drive.get('label') or 'not recorded (scope preset ' + str(plan.get('preset')) + ')'}",
              f"{plan.get('shots', '-')} shots per angle, preset {plan.get('preset', '-')}",
              f"created {m.get('created', '-')}"]
        if (m.get("borrowed") or {}):
            L.append("darks/backgrounds borrowed: " + ", ".join(f"{k} from {(v or {}).get('source', '?')}"
                                                                for k, v in m["borrowed"].items() if isinstance(v, dict)))
    if out["note"]:
        L += ["", "note: " + out["note"]]
    return out


def window_ratio(folder, man, p):
    """The nearest-null raw trace's mean over the ER window's first half over
    its second: above 1 the scope is still recovering from the bright rest
    light inside the window (8 Oct 2026, every 90 deg point)."""
    sc, f = p.get("scan"), p.get("fit")
    if not sc or not f:
        return None
    try:
        npz = np.load(os.path.join(folder, f"point_{p['i']:02d}.npz"))
    except OSError:
        return None
    th = np.array(sc["theta"]); k = int(np.argmin(np.abs(th - f["theta_n"]))); key = f"null_{th[k]:.3f}"
    if key not in npz.files:
        return None
    t = npz["t"]; w = man["window_s"]; mid = 0.5 * (w[0] + w[1])
    y = npz[key]
    a = y[(t > w[0]) & (t < mid)].mean(); b = y[(t > mid) & (t < w[1])].mean()
    dark = dark_for(man, sc["vdiv"])
    dark = dark[0] if dark else 0.0
    return float((a - dark) / (b - dark)) if (b - dark) > 0 else None


def dark_for(man, vdiv):
    """(level, sem) of the bias run's dark at this V/div, or None."""
    return next((v for k, v in (man.get("dark") or {}).items() if k.startswith(f"{vdiv:.6g}V/div")), None)


def mon_rotation(d, steps):
    """{role: rotation (deg) from the mean monitor trace, zeroed before t=0}
    and their sum."""
    out = {}
    for r in ("MonX1", "MonX2"):
        if r in d.roles:
            m = np.mean([s["v"][r] for s in steps], axis=0)
            out[r] = K_MON[r] * (m - m[d.t < -0.1e-3].mean())
    total = sum(out.values()) if out else np.zeros(len(d.t))
    return out, total


def light_sign(pol, mon_sum):
    """+1 or -1: the fitted light rotation's sign that matches the monitors."""
    rot = pol.get("rotation")
    if rot is None or np.ptp(mon_sum) <= 1:
        return 1.0
    return -1.0 if np.corrcoef(rot, mon_sum)[0, 1] < 0 else 1.0


# ---------------------------------------------------------------------- plots
class Toolbar(NavigationToolbar2Tk):
    def save_figure(self, *args):
        d = getattr(self.canvas, "save_dir", None)
        if d:
            sub = os.path.join(d, "saved_figures")
            matplotlib.rcParams["savefig.directory"] = sub if os.path.isdir(sub) else d
        return super().save_figure(*args)


class Plot(ttk.Frame):
    """A matplotlib figure with a toolbar in a frame."""

    def __init__(self, master, **kw):
        super().__init__(master)
        self.fig = Figure(figsize=(9, 6), dpi=96, **kw)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.save_dir = None
        self.canvas.save_name = "figure.png"
        self.canvas.get_default_filename = lambda: self.canvas.save_name
        self.toolbar = Toolbar(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self, folder=None, name=None):
        self.fig.clear()
        if folder:
            self.canvas.save_dir = folder
        if name:
            self.canvas.save_name = name.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        return self.fig

    def draw(self):
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()


# --------------------------------------------------------------------- viewer
class Viewer(tk.Tk):
    def __init__(self, outdir):
        super().__init__()
        self.title(f"rampol data viewer - {outdir}")
        self.geometry("1500x940")
        self.outdir = outdir
        cfg = cfgmod.load()
        self.sg = hw.load_scope_grab(cfg["scope_grab_path"])
        self.cache = {}            # folder -> load_scan's step cache
        self.loaded = {}           # (folder, raw) -> data: runs already loaded
        self.info = {}             # run name -> describe()
        self.drawn = {}            # tab -> key of what it shows (skip a redraw when unchanged)
        self.cur = None            # (kind, folder, data)
        self.sel_point = 0
        self.mark = None           # (t_ms, theta) picked in the direct-ER table
        self.raw = tk.BooleanVar(value=False)
        self._q = queue.Queue()
        self._job = 0
        self._pending = None
        self._after = None
        self._build()
        self._fill_tree()
        threading.Thread(target=self._describe_all, daemon=True).start()
        self.after(50, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self):
        style = ttk.Style(self)
        style.configure("Head.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Sub.TLabel", foreground="#444")
        self.status = tk.StringVar(value="pick a run on the left")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(6, 2)).pack(side="bottom", fill="x")
        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True)

        # left: filter, run tree, run info
        left = ttk.Frame(pan, width=420)
        pan.add(left, weight=0)
        row = ttk.Frame(left); row.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(row, text="filter").pack(side="left")
        self.filt = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.filt)
        e.pack(side="left", fill="x", expand=True, padx=4)
        e.bind("<KeyRelease>", lambda _e: self._fill_tree())
        self.newest = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="newest first", variable=self.newest, command=self._fill_tree).pack(side="left")
        lp = ttk.PanedWindow(left, orient="vertical")
        lp.pack(fill="both", expand=True, padx=4, pady=4)
        tf = ttk.Frame(lp)
        lp.add(tf, weight=3)
        self.tree = ttk.Treeview(tf, columns=("type", "n", "date"), show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="run"); self.tree.column("#0", width=230, stretch=True)
        for c, w in (("type", 75), ("n", 45), ("date", 75)):
            self.tree.heading(c, text=c); self.tree.column(c, width=w, stretch=False, anchor="w")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.run_info = tk.Text(lp, height=11, wrap="word", font=("Segoe UI", 9), relief="flat", background=self.cget("background"))
        lp.add(self.run_info, weight=1)

        # right: header, the run's numbers (points / angles + direct ER), tabs
        right = ttk.Frame(pan)
        pan.add(right, weight=1)
        self.head = tk.StringVar(value="no run loaded")
        self.subhead = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.head, style="Head.TLabel").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(right, textvariable=self.subhead, style="Sub.TLabel").pack(anchor="w", padx=6)
        rp = ttk.PanedWindow(right, orient="vertical")
        rp.pack(fill="both", expand=True, padx=4, pady=4)
        self.sel_area = ttk.Frame(rp, height=170)
        rp.add(self.sel_area, weight=0)
        self._build_point_table()
        self._build_angle_picker()
        self.nb = ttk.Notebook(rp)
        rp.add(self.nb, weight=1)
        self.tabs = {}
        for name in ("Overview", "Null scan", "Tracking", "Traces", "Fit", "ER(t)", "Details"):
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=name)
            self.tabs[name] = f
        for f in self.tabs.values():
            self.nb.hide(f)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._refresh_tab())

        f = self.tabs["Overview"]
        row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="show:").pack(side="left")
        self.ov_key = tk.StringVar(value="er")
        self._ov_user = "er"       # the quantity picked by hand; a run without it shows another for itself only
        for k, (lab, _) in QUANT.items():
            ttk.Radiobutton(row, text=lab, value=k, variable=self.ov_key, command=self._on_ov_key).pack(side="left", padx=2)
        ttk.Label(row, text="  (click a point in the plot to select it)", style="Sub.TLabel").pack(side="left")
        self.ov_plot = Plot(f); self.ov_plot.pack(fill="both", expand=True)
        self.ov_plot.canvas.mpl_connect("button_press_event", self._on_ov_click)
        self._ov_hit = None
        self.null_plot = Plot(self.tabs["Null scan"]); self.null_plot.pack(fill="both", expand=True)
        self.track_plot = Plot(self.tabs["Tracking"]); self.track_plot.pack(fill="both", expand=True)
        self.trace_plot = Plot(self.tabs["Traces"]); self.trace_plot.pack(fill="both", expand=True)
        self.fit_plot = Plot(self.tabs["Fit"]); self.fit_plot.pack(fill="both", expand=True)
        ef = self.tabs["ER(t)"]
        row = ttk.Frame(ef); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="angles within +-").pack(side="left")
        self.er_half = tk.DoubleVar(value=4.0)
        ttk.Entry(row, textvariable=self.er_half, width=5).pack(side="left")
        ttk.Label(row, text="deg of crossed; boxcar").pack(side="left")
        self.er_box = tk.DoubleVar(value=20.0)
        ttk.Entry(row, textvariable=self.er_box, width=6).pack(side="left")
        ttk.Label(row, text="us").pack(side="left")
        ttk.Button(row, text="recompute", command=self._refresh_tab).pack(side="left", padx=8)
        self.er_plot = Plot(ef); self.er_plot.pack(fill="both", expand=True)
        df = self.tabs["Details"]
        row = ttk.Frame(df); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="open folder", command=lambda: self.cur and os.startfile(self.cur[1])).pack(side="left")
        ttk.Button(row, text="open manifest", command=self._open_manifest).pack(side="left", padx=4)
        self.details = tk.Text(df, wrap="none", font=("Consolas", 9))
        ys = ttk.Scrollbar(df, orient="vertical", command=self.details.yview)
        self.details.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y"); self.details.pack(fill="both", expand=True)

    def _build_point_table(self):
        f = self.point_frame = ttk.Frame(self.sel_area)
        cols = [("i", "pt", 34), ("x1", "X1", 48), ("x2", "X2", 48), ("hold", "hold ms", 58), ("er", "ER", 62),
                ("imin", "Imin mV", 64), ("imax", "Imax V", 58), ("null", "crossed deg", 78), ("creep", "drift mdeg/ms", 88),
                ("after", "after 1 ms mdeg", 96), ("tau", "tau ms", 52), ("vdiv", "mV/div", 52), ("ratio", "win 1st/2nd", 74),
                ("flags", "flags", 220)]
        self.ptable = ttk.Treeview(f, columns=[c for c, _, _ in cols], show="headings", height=6, selectmode="browse")
        for c, h, w in cols:
            self.ptable.heading(c, text=h)
            self.ptable.column(c, width=w, anchor="w" if c == "flags" else "center", stretch=c == "flags")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.ptable.yview)
        self.ptable.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.ptable.pack(fill="both", expand=True)
        self.ptable.bind("<<TreeviewSelect>>", self._on_point)

    def _build_angle_picker(self):
        f = self.angle_frame = ttk.Frame(self.sel_area)
        lf = ttk.Frame(f); lf.pack(side="left", fill="y")
        ttk.Label(lf, text="analyzer angles shown").pack(anchor="w")
        box = ttk.Frame(lf); box.pack(fill="y", expand=True)
        self.angle_list = tk.Listbox(box, selectmode="extended", height=7, width=12, exportselection=False)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.angle_list.yview)
        self.angle_list.configure(yscrollcommand=sb.set)
        self.angle_list.pack(side="left", fill="y"); sb.pack(side="left", fill="y")
        self.angle_list.bind("<<ListboxSelect>>", lambda _e: self._refresh_tab())
        bf = ttk.Frame(f); bf.pack(side="left", fill="y", padx=4)
        ttk.Button(bf, text="all", width=12, command=lambda: self._select_angles("all")).pack(pady=1)
        ttk.Button(bf, text="none", width=12, command=lambda: self._select_angles("none")).pack(pady=1)
        self.near_btn = ttk.Button(bf, text="near crossed", width=12, command=lambda: self._select_angles("near"))
        self.near_btn.pack(pady=1)
        ttk.Checkbutton(bf, text="raw (no dark\nsubtraction)", variable=self.raw, command=self._on_raw).pack(pady=6, anchor="w")
        df = ttk.Frame(f); df.pack(side="left", fill="both", expand=True)
        ttk.Label(df, text="direct extinction ratios (both intensities measured; pick one to mark its time and angle)").pack(anchor="w")
        cols = [("kind", "kind", 64), ("seg", "segment", 64), ("t", "t ms", 60), ("theta", "angle", 56), ("rot", "rotation", 64),
                ("imin", "Imin mV", 64), ("sig", "+- mV", 52), ("imax", "Imax V", 56), ("er", "ER", 64), ("note", "note", 260)]
        self.dtable = ttk.Treeview(df, columns=[c for c, _, _ in cols], show="headings", height=6, selectmode="browse")
        for c, h, w in cols:
            self.dtable.heading(c, text=h)
            self.dtable.column(c, width=w, anchor="w" if c == "note" else "center", stretch=c == "note")
        sb = ttk.Scrollbar(df, orient="vertical", command=self.dtable.yview)
        self.dtable.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.dtable.pack(fill="both", expand=True)
        self.dtable.bind("<<TreeviewSelect>>", self._on_direct)

    # ---------------------------------------------------------------- run list
    def _runs(self):
        out = []
        for f in os.listdir(self.outdir):
            p = os.path.join(self.outdir, f)
            if os.path.isdir(p):
                k = run_kind(p)
                if k:
                    try:
                        mt = os.path.getmtime(manifest_path(p, k))
                    except OSError:
                        mt = 0.0
                    out.append((f, k, mt))
        return out

    def _fill_tree(self):
        if not hasattr(self, "_all_runs"):
            self._all_runs = self._runs()
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        pat = self.filt.get().strip().lower()
        groups = {}
        for f, k, mt in self._all_runs:
            inf = self.info.get(f) or {}
            hay = (f + " " + inf.get("label", "") + " " + inf.get("note", "") + " " + inf.get("type", "")).lower()
            if pat and not all(w in hay for w in pat.split()):
                continue
            groups.setdefault(series_of(f), []).append((f, k, mt))
        newest = self.newest.get()
        order = sorted(groups, key=(lambda g: -max(m for _f, _k, m in groups[g])) if newest else str.lower)
        for g in order:
            items = sorted(groups[g], key=lambda it: it[0].lower())
            if len(items) == 1:
                self._insert_run("", items[0])
                continue
            node = self.tree.insert("", "end", iid="grp:" + g, text=f"{g}  ({len(items)})", open=bool(pat))
            for it in items:
                self._insert_run(node, it)
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0]); self.tree.see(sel[0])

    def _insert_run(self, parent, item):
        f, k, _mt = item
        inf = self.info.get(f) or {}
        self.tree.insert(parent, "end", iid=f, text=f, values=(inf.get("type", "scan" if k == "scan" else "bias"),
                                                               inf.get("n", ""), inf.get("date", "")))

    def _describe_all(self):
        """Worker: every run's description, posted back for the tree."""
        for f, k, _mt in sorted(self._all_runs, key=lambda it: -it[2]):
            try:
                self._q.put(("info", f, describe(os.path.join(self.outdir, f), k)))
            except Exception as exc:
                self._q.put(("info", f, {"type": "?", "n": "", "date": "", "label": "", "note": "", "lines": [f"cannot read: {exc}"]}))

    # ---------------------------------------------------------------- loading
    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "info":
                    _, f, inf = msg
                    self.info[f] = inf
                    if self.tree.exists(f):
                        self.tree.item(f, values=(inf["type"], inf["n"], inf["date"]))
                    sel = self.tree.selection()
                    if sel and sel[0] == f:
                        self._show_info(f)
                elif msg[0] == "loaded":
                    self._on_loaded(*msg[1:])
        except queue.Empty:
            pass
        if self._job and self._loading:
            self.status.set(f"loading {self._loading} ... {time.time() - self._t_load:.0f} s")
        self.after(100, self._poll)

    _loading = None

    def _show_info(self, name):
        inf = self.info.get(name)
        self.run_info.configure(state="normal")
        self.run_info.delete("1.0", "end")
        self.run_info.insert("1.0", name + "\n\n" + ("\n".join(inf["lines"]) if inf else "(reading ...)"))
        self.run_info.configure(state="disabled")

    def _on_select(self, _e=None):
        sel = self.tree.selection()
        if not sel:
            return
        if sel[0].startswith("grp:"):
            kids = self.tree.get_children(sel[0])
            self.run_info.configure(state="normal"); self.run_info.delete("1.0", "end")
            self.run_info.insert("1.0", f"{sel[0][4:]}: {len(kids)} runs\n\n" + "\n".join(
                f"{k}: {(self.info.get(k) or {}).get('label', '')}" for k in kids))
            self.run_info.configure(state="disabled")
            return
        self._show_info(sel[0])
        # load after the selection settles, so arrowing through the list does not queue loads
        self._pending = sel[0]
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(250, self._start_load)

    def _start_load(self):
        self._after = None
        name = self._pending
        folder = os.path.join(self.outdir, name)
        kind = run_kind(folder)
        raw = bool(self.raw.get()) if kind == "scan" else False
        key = (folder, raw)
        if key in self.loaded:
            self._show(name, kind, folder, self.loaded[key])
            return
        self._job += 1
        job = self._job
        self._loading, self._t_load = name, time.time()
        self.status.set(f"loading {name} ...")
        cache = self.cache.setdefault(folder, {})
        threading.Thread(target=self._load_worker, args=(job, name, kind, folder, raw, cache), daemon=True).start()

    def _load_worker(self, job, name, kind, folder, raw, cache):
        """Worker thread: numbers only, no Tk."""
        try:
            if kind == "bias":
                man = biasmod.load(folder)
                ratios = {p["i"]: window_ratio(folder, man, p) for p in man["points"]}
                data = {"man": man, "ratio": ratios}
            else:
                d = an.load_scan(folder, self.sg.load_capture, cache=cache)
                d.subtract_dark = not raw
                try:
                    pol = an.polarization(d)
                except ValueError as exc:
                    # one or two analyzer angles (the slew sweeps): traces only
                    th, I, _sem, steps = an.scan_matrix(d, "scan")
                    pol = {"theta": th, "I": I, "steps": steps, "rotation": None, "no_fit": str(exc)}
                direct, direct_err = [], ""
                if not pol.get("no_fit"):
                    try:
                        direct = an.direct_er(d, pol, gains=pol.get("angle_gain"))
                    except Exception as exc:
                        direct_err = str(exc)
                data = {"d": d, "pol": pol, "direct": direct, "direct_err": direct_err, "er": None, "er_key": None}
            self._q.put(("loaded", job, name, kind, folder, raw, data, None))
        except Exception as exc:
            traceback.print_exc()
            self._q.put(("loaded", job, name, kind, folder, raw, None, str(exc) or type(exc).__name__))

    def _on_loaded(self, job, name, kind, folder, raw, data, err):
        if data is not None:
            self.loaded[(folder, raw)] = data
        if job != self._job:
            return              # the user has moved on; kept in the cache
        self._loading = None
        if err:
            self.status.set(f"{name}: {err}")
            return
        self._show(name, kind, folder, data, dt=time.time() - self._t_load)

    def _show(self, name, kind, folder, data, dt=None):
        new_run = not self.cur or self.cur[1] != folder
        self.cur = (kind, folder, data)
        inf = self.info.get(name) or {}
        self.head.set(name)
        self.subhead.set(f"{inf.get('type', kind)}: {inf.get('label', '')}")
        for f in (self.point_frame, self.angle_frame):
            f.pack_forget()
        if kind == "bias":
            self.point_frame.pack(fill="both", expand=True)
            if new_run:
                self.sel_point = data["man"]["points"][0]["i"] if data["man"]["points"] else 0
                self.ov_key.set(self._ov_user)
            self._fill_points(data)
            note = "" if data["man"]["points"] else " - no points yet"
        else:
            self.angle_frame.pack(fill="both", expand=True)
            if new_run:
                self.mark = None
            self._fill_angles(data, keep=not new_run)
            self._fill_direct(data)
            nf = data["pol"].get("no_fit")
            self.near_btn.state(["disabled"] if nf else ["!disabled"])
            note = f" - {nf}: traces only, no fit" if nf else ""
        self.status.set(f"{name}{note}" + (f"  (loaded in {dt:.1f} s)" if dt and dt > 1 else ""))
        # only the tabs that apply; stay on the same tab when it still applies
        want = TABS[kind]
        cur_tab = self._tab()
        for t, f in self.tabs.items():
            if t in want:
                self.nb.add(f)
            else:
                self.nb.hide(f)
        if cur_tab not in want:
            self.nb.select(self.tabs[want[0]])
        self._refresh_tab()

    def _tab(self):
        try:
            return self.nb.tab(self.nb.select(), "text")
        except tk.TclError:
            return None

    def _refresh_tab(self):
        if not self.cur:
            return
        tab = self._tab()
        kind = self.cur[0]
        if tab not in TABS[kind]:
            return
        # what this tab would show; skip the redraw when it is already on screen
        key = (self.cur[1], self.raw.get() if kind == "scan" else None,
               self.sel_point if kind == "bias" else (tuple(self.angle_list.curselection()), self.mark),
               self.ov_key.get() if tab == "Overview" else None,
               (self.er_half.get(), self.er_box.get()) if tab == "ER(t)" else None)
        if self.drawn.get(tab) == key:
            return
        self.drawn[tab] = key
        draw = {"Overview": self._draw_overview, "Null scan": self._draw_null, "Tracking": self._draw_track,
                "Traces": self._draw_traces, "Fit": self._draw_fit, "ER(t)": self._draw_er,
                "Details": self._details}[tab]
        try:
            draw()
        except Exception as exc:
            self.drawn.pop(tab, None)
            self.status.set(f"{tab}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    def _open_manifest(self):
        if self.cur:
            p = manifest_path(self.cur[1], self.cur[0])
            if p:
                os.startfile(p)

    def _name(self, tab, extra=""):
        return f"{os.path.basename(self.cur[1])}_{tab}{extra}"

    # ---------------------------------------------------------------- bias runs
    def _fill_points(self, data):
        man = data["man"]
        self.ptable.delete(*self.ptable.get_children())
        hold_plan = (man.get("plan") or {}).get("hold_ms")
        for p in man["points"]:
            tk_ = p.get("track") or {}
            flags = []
            if p.get("er") is None and p.get("er_lower"):
                flags.append("ER is a lower bound")
            if p.get("track_only"):
                flags.append(f"track only (null from {p.get('null_from', '?')})")
            if p.get("scan") and not p.get("null_converged", True):
                flags.append("null not converged")
            ratio = data["ratio"].get(p["i"])
            if ratio and ratio > 1.05:
                flags.append("still recovering in the window")
            er = p.get("er") or p.get("er_lower")
            self.ptable.insert("", "end", iid=str(p["i"]), values=(
                p["i"], f"{p['x1']:g}", f"{p['x2']:g}", fnum(p.get("hold_ms") or hold_plan, "g"),
                (">" if p.get("er") is None and er else "") + fnum(er, ".0f"),
                fnum(p["imin"] * 1e3 if p.get("imin") is not None else None, ".3f"),
                fnum(p.get("imax"), ".3f"), fnum(p.get("theta_n"), ".3f"),
                fnum(tk_.get("hold_slope_mdeg_ms"), "+.1f"), fnum(tk_.get("after_1ms_mdeg"), "+.0f"),
                fnum(tk_.get("tau_ms"), ".0f"), fnum((p.get("scan") or {}).get("vdiv", np.nan) * 1e3, "g"),
                fnum(ratio, ".2f"), "; ".join(flags)))
        if self.ptable.exists(str(self.sel_point)):
            self.ptable.selection_set(str(self.sel_point)); self.ptable.see(str(self.sel_point))

    def _on_point(self, _e=None):
        sel = self.ptable.selection()
        if sel and int(sel[0]) != self.sel_point:
            self.sel_point = int(sel[0])
            self._refresh_tab()

    def _point(self):
        man = self.cur[2]["man"]
        return next(p for p in man["points"] if p["i"] == self.sel_point)

    def _qval(self, p, key):
        """(value, sigma, is_lower_bound) of an Overview quantity at a point."""
        tk_ = p.get("track") or {}
        if key == "er":
            if p.get("er"):
                return p["er"], p.get("sig_er") or np.nan, False
            return (p.get("er_lower") or np.nan), np.nan, bool(p.get("er_lower"))
        if key == "imin":
            return ((p["imin"] * 1e3) if p.get("imin") is not None else np.nan,
                    (p["sig_imin"] * 1e3) if p.get("sig_imin") is not None else np.nan, False)
        if key == "null":
            return (p.get("theta_n") if p.get("theta_n") is not None else np.nan), (p.get("sig_theta_n") or np.nan), False
        k = {"creep": "hold_slope_mdeg_ms", "after": "after_1ms_mdeg", "tau": "tau_ms"}[key]
        v = tk_.get(k)
        return (np.nan if v is None else v), np.nan, False

    def _draw_overview(self):
        man = self.cur[2]["man"]
        pts = man["points"]
        key = self.ov_key.get()
        lab, axlab = QUANT[key]
        fig = self.ov_plot.clear(self.cur[1], self._name("overview", "_" + key))
        self._ov_hit = None
        if not pts:
            message(fig, "no points in this run yet"); self.ov_plot.draw(); return
        vals = [self._qval(p, key) for p in pts]
        if not any(np.isfinite(a) for a, _s, _l in vals):
            # e.g. ER on a track-only run: show the first quantity it has, and say so
            alt = next((k for k in QUANT if any(np.isfinite(self._qval(p, k)[0]) for p in pts)), None)
            if alt is None:
                message(fig, f"no values in {man['name']} yet")
                self.ov_plot.draw(); return
            why = "track-only points, no null scan" if all(p.get("track_only") for p in pts) else "none measured"
            self.status.set(f"{man['name']}: no {lab} values ({why}) - showing {QUANT[alt][0]}")
            self.ov_key.set(alt)
            key = alt; lab, axlab = QUANT[key]
            self.drawn["Overview"] = None
            vals = [self._qval(p, key) for p in pts]
        v = np.array([a for a, _s, _l in vals], float); s = np.array([b for _a, b, _l in vals], float)
        low = np.array([c for _a, _s, c in vals], bool)
        x1 = np.array([p["x1"] for p in pts], float); x2 = np.array([p["x2"] for p in pts], float)
        ids = [p["i"] for p in pts]
        xs, ys = sorted(set(x1)), sorted(set(x2))
        logv = key == "er" and np.any(np.isfinite(v) & (v > 0))
        norm = matplotlib.colors.LogNorm() if logv else None
        fmt = (lambda c, lo: (">" if lo else "") + f"{c:.0f}") if key in ("er", "after", "tau") else (lambda c, lo: f"{c:.2f}")
        ax = fig.add_subplot(111)
        if len(xs) > 1 and len(ys) > 1:
            if len(pts) == len(xs) * len(ys):
                g = np.full((len(ys), len(xs)), np.nan)
                for a, b, c in zip(x1, x2, v):
                    g[ys.index(b), xs.index(a)] = c
                hx = 0.5 * float(np.min(np.diff(xs))); hy = 0.5 * float(np.min(np.diff(ys)))
                im = ax.imshow(g, origin="lower", aspect="auto", cmap="viridis", norm=norm,
                               extent=(xs[0] - hx, xs[-1] + hx, ys[0] - hy, ys[-1] + hy))
            else:
                im = ax.scatter(x1, x2, c=v, s=420, marker="s", cmap="viridis", norm=norm, edgecolors="k", linewidths=0.4)
                ax.set_aspect("equal", adjustable="datalim")
            fig.colorbar(im, ax=ax, label=axlab)
            for a, b, c, lo in zip(x1, x2, v, low):
                if np.isfinite(c):
                    ax.text(a, b, fmt(c, lo), ha="center", va="center", fontsize=7, color="w")
            ax.set(xlabel="X1 rotation (deg)", ylabel="X2 rotation (deg)")
            xy = np.column_stack([x1, x2])
            sel = xy[ids.index(self.sel_point)]
            ax.plot(*sel, "o", ms=22, mfc="none", mec="r", mew=2)
            ax.set_title(f"{lab} per point, {man['name']} (red ring: point {self.sel_point}"
                         + ("; '>': lower bound)" if low.any() else ")"))
        else:
            if len(xs) > 1:
                x, xlab = x1, "X1 rotation (deg)"
            elif len(ys) > 1:
                x, xlab = x2, "X2 rotation (deg)"
            else:
                x, xlab = np.arange(len(pts), dtype=float), "point"
                ax.set_xticks(x); ax.set_xticklabels([f"{i}\n{a:g}/{b:g}" for i, a, b in zip(ids, x1, x2)], fontsize=7)
            m = ~low & np.isfinite(v)
            ax.errorbar(x[m], v[m], np.where(np.isfinite(s[m]), s[m], 0), fmt="o", ms=5, capsize=2, color="C0", label=lab)
            if low.any():
                ax.plot(x[low], v[low], "o", ms=5, mfc="none", mec="C0", label="lower bound")
                for a, c in zip(x[low], v[low]):
                    ax.plot([a, a], [c, c * 3 if logv else c + 0.1 * np.nanmax(np.abs(v))], ":", color="C0")
            if logv:
                ax.set_yscale("log")
            ax.set(xlabel=xlab, ylabel=axlab, title=f"{lab} per point, {man['name']} (red ring: point {self.sel_point})")
            xy = np.column_stack([x, v])
            k = ids.index(self.sel_point)
            if np.isfinite(v[k]):
                ax.plot([x[k]], [v[k]], "o", ms=14, mfc="none", mec="r", mew=2)
            ax.grid(alpha=0.3, which="both")
            legend(ax)
        self._ov_hit = (ax, xy, ids)
        self.ov_plot.draw()

    def _on_ov_key(self):
        self._ov_user = self.ov_key.get()
        self._refresh_tab()

    def _on_ov_click(self, ev):
        if not self._ov_hit or ev.inaxes is not self._ov_hit[0] or self.ov_plot.toolbar.mode:
            return
        ax, xy, ids = self._ov_hit
        ok = np.all(np.isfinite(xy), axis=1)
        if not ok.any():
            return
        px = ax.transData.transform(xy[ok])
        dist = np.hypot(px[:, 0] - ev.x, px[:, 1] - ev.y)
        k = int(np.argmin(dist))
        if dist[k] < 40:
            i = np.array(ids)[ok][k]
            self.ptable.selection_set(str(i)); self.ptable.see(str(i))

    def _point_npz(self, p):
        try:
            return np.load(os.path.join(self.cur[1], f"point_{p['i']:02d}.npz"))
        except OSError:
            return None

    def _draw_null(self):
        man = self.cur[2]["man"]
        p = self._point()
        npz = self._point_npz(p)
        fig = self.null_plot.clear(self.cur[1], self._name("null_scan", f"_p{p['i']:02d}"))
        w = man["window_s"]
        sc = p.get("scan")
        coarse = [k for k in (npz.files if npz is not None else []) if k.startswith("coarse_") or k == "bright"]
        if not sc:
            if not coarse:
                message(fig, f"point {p['i']} (X1 {p['x1']:g} / X2 {p['x2']:g}) has no null scan: it is track-only, "
                             f"its crossed angle {fnum(p.get('theta_n'), '.3f')} deg came from {p.get('null_from', 'elsewhere')}.\n"
                             "See the Tracking tab.")
                self.null_plot.draw(); return
            ax2 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(221 if coarse else 211)
            ax3 = fig.add_subplot(212)
            ax2 = fig.add_subplot(222) if coarse else None
            th = np.array(sc["theta"]); I = np.array(sc["I"]) * 1e3; sem = np.array(sc["sem"]) * 1e3
            cols = matplotlib.cm.viridis(np.linspace(0, 0.95, max(len(th), 2)))
            ax1.errorbar(th, I, sem, fmt="none", ecolor="gray", capsize=2)
            ax1.scatter(th, I, c=cols[:len(th)], s=28, zorder=3, label=f"window means, {sc['vdiv']*1e3:g} mV/div")
            f = p.get("fit") or {}
            if f.get("k"):
                tt = np.linspace(th.min(), th.max(), 200)
                ax1.plot(tt, (f["imin"] + f["k"] * np.sin(np.deg2rad(tt - f["theta_n"])) ** 2) * 1e3, "-", lw=1, color="C1",
                         label=f"Imin + k sin^2: Imin {f['imin']*1e3:.3f} +- {f['sig_imin']*1e3:.3f} mV\n"
                               f"at {f['theta_n']:.3f} deg, rms {f['rms']*1e3:.3f} mV")
            dk = dark_for(man, sc["vdiv"])
            ax1.axhline(0, color="k", lw=0.5)
            er = (f"ER {p['er']:.0f}" + (f" +- {p['sig_er']:.0f}" if p.get("sig_er") else "")) if p.get("er") else \
                 (f"ER > {p['er_lower']:.0f}" if p.get("er_lower") else "no ER")
            ax1.set(title=f"Null scan, point {p['i']}: {er}", xlabel="analyzer (deg)",
                    ylabel="I (mV, minus the dark)")
            if dk:
                ax1.text(0.02, 0.03, f"dark {dk[0]*1e3:+.3f} +- {dk[1]*1e3:.3f} mV subtracted", fontsize=7, transform=ax1.transAxes)
            ax1.legend(fontsize=7, loc="upper center"); ax1.grid(alpha=0.3)
            # the raw traces behind each window mean, same colours as the points
            t = npz["t"] * 1e3
            for a, c in zip(th, cols):
                k = f"null_{a:.3f}"
                if k in npz.files:
                    plot_thin(ax3, t, npz[k] * 1e3, color=c, lw=0.7, label=f"{a:.2f} deg")
            ax3.axvspan(w[0] * 1e3, w[1] * 1e3, color="orange", alpha=0.15, label="ER window")
            if dk:
                ax3.axhline(dk[0] * 1e3, color="k", lw=0.8, ls="--", label=f"dark {dk[0]*1e3:+.3f} mV")
            ax3.set(title="Raw PD traces at the null-scan angles (window mean minus the dark = the points above)",
                    xlabel="t (ms)", ylabel="PD (mV, raw)")
            legend(ax3, ncol=2); ax3.grid(alpha=0.3)
        if ax2 is not None:
            t = npz["t"] * 1e3
            for k in coarse:
                plot_thin(ax2, t, npz[k], lw=0.7, label=k.replace("coarse_", "analyzer ").replace("bright", "bright angle"))
            ax2.axvspan(w[0] * 1e3, w[1] * 1e3, color="orange", alpha=0.15)
            ax2.set(title=f"Coarse angles, {man['coarse'][0]:g} V/div: Imax {fnum(p.get('imax'), '.3f')} V",
                    xlabel="t (ms)", ylabel="PD (V)")
            ax2.legend(fontsize=7, loc="best"); ax2.grid(alpha=0.3)
        self.null_plot.draw()

    def _draw_track(self):
        man = self.cur[2]["man"]
        p = self._point()
        npz = self._point_npz(p)
        fig = self.track_plot.clear(self.cur[1], self._name("tracking", f"_p{p['i']:02d}"))
        if npz is None or "track_t" not in npz.files:
            message(fig, f"point {p['i']} was not tracked (no slope-pair traces)")
            self.track_plot.draw(); return
        w = man["window_s"]
        mons = [k for k in npz.files if k.startswith("mon_")]
        n = 4 if mons else 3
        axs = [fig.add_subplot(n, 1, k + 1) for k in range(n)]
        t = npz["t"] * 1e3
        names = {"slope_hold_plus": "hold pair, null + 45", "slope_hold_minus": "hold pair, null - 45",
                 "slope_rest_plus": "rest pair, null + 45", "slope_rest_minus": "rest pair, null - 45"}
        for k, lab in names.items():
            if k in npz.files:
                plot_thin(axs[0], t, npz[k], lw=0.7, label=lab)
        axs[0].axvspan(w[0] * 1e3, w[1] * 1e3, color="orange", alpha=0.15, label="ER window")
        axs[0].set(title=f"Slope-pair traces, point {p['i']} (X1 {p['x1']:g} / X2 {p['x2']:g})", ylabel="PD (V)")
        tt = npz["track_t"] * 1e3
        sense = p.get("sense", -1.0)
        rot = npz["track_mon_rot"]
        # A pair reads the light only within 45 deg of its own null; its 'valid'
        # flag (|ratio| < 0.95) also passes 90 deg away, where the ratio is ~0
        # again. Each pair is therefore shown only where the monitors put the
        # light within 40 deg of that pair's null: the hold pair's null is the
        # rotation in the ER window, the rest pair's the rotation at the end.
        w_ms = (tt >= w[0] * 1e3) & (tt <= w[1] * 1e3)
        rot_hold = float(np.mean(rot[w_ms])) if w_ms.any() else 0.0
        rot_rest = float(np.median(rot[-max(len(rot) // 20, 1):]))
        v = npz["track_valid"].astype(bool) & (np.abs(rot - rot_hold) < 40)
        lm = np.where(v, npz["track_lm"] * 1e3, np.nan)
        plot_thin(axs[1], tt, rot, lw=0.9, color="C1", label="monitors (summed)")
        # light in the monitors' frame: dpsi = lm + sense x rotation
        plot_thin(axs[1], tt, np.where(v, rot + sense * npz["track_lm"], np.nan), lw=0.7, color="C0", label="light, hold pair")
        plot_thin(axs[2], tt, lm, lw=0.7, color="C0", label="hold pair")
        series = [lm]
        if "track_lm_rest" in npz.files:
            vr = npz["track_valid_rest"].astype(bool) & (np.abs(rot - rot_rest) < 40)
            plot_thin(axs[1], tt, np.where(vr, rot + sense * npz["track_lm_rest"], np.nan), lw=0.7, color="C2", label="light, rest pair")
            lr = np.where(vr, npz["track_lm_rest"] * 1e3, np.nan)
            plot_thin(axs[2], tt, lr, lw=0.7, color="C2", label="rest pair")
            series.append(lr)
        axs[1].set(title="Rotation (each light pair shown within 40 deg of its null, by the monitors)", ylabel="rotation (deg)")
        robust_ylim(axs[2], *series, min_span=50)
        tk_ = p.get("track") or {}
        axs[2].set(ylabel="light - monitors (mdeg)",
                   title=f"Light minus monitors: drift in the hold {fnum(tk_.get('hold_slope_mdeg_ms'), '+.1f')} mdeg/ms, "
                         f"1 ms after the fall {fnum(tk_.get('after_1ms_mdeg'), '+.0f')} mdeg,\n"
                         f"extreme {fnum(tk_.get('after_extreme_mdeg'), '+.0f')} mdeg at {fnum(tk_.get('after_extreme_ms'), '.0f')} ms "
                         f"after the fall, relaxation tau {fnum(tk_.get('tau_ms'), '.0f')} ms")
        if mons:
            tm = npz["t"] * 1e3
            for k in mons:
                plot_thin(axs[3], tm, npz[k], lw=0.7, label=k[4:])
            axs[3].set(title="Monitor and command traces", ylabel="V")
        for ax in axs:
            ax.set_xlabel("t (ms)"); ax.grid(alpha=0.3); legend(ax)
        self.track_plot.draw()

    def _details(self):
        kind, folder, data = self.cur
        man = data["man"] if kind == "bias" else data["d"].manifest
        name = os.path.basename(folder)
        inf = self.info.get(name) or {}
        lines = [name, "=" * len(name), ""] + inf.get("lines", []) + [""]
        if kind == "scan":
            pol = data["pol"]
            if pol.get("no_fit"):
                lines += [f"no fit: {pol['no_fit']}", ""]
            else:
                lines += [f"dark subtracted {pol['dark']*1e3:+.3f} mV; per-angle gains "
                          f"{'fitted' if pol.get('angle_gain') is not None else 'none'} {pol.get('gain_note', '')}",
                          f"rest polarization angle {pol['psi_rest']:.3f} deg; crossed analyzer {(pol['psi_rest'] + 90) % 180:.3f} deg", ""]
        else:
            lines += ["darks (V/div@offset: level, sem):"] + [f"  {k}: {v[0]*1e3:+.3f} +- {v[1]*1e3:.3f} mV"
                                                               for k, v in (man.get("dark") or {}).items()] + [""]
            for c in man.get("limit_checks") or []:
                lines.append("limit check: " + str(c))
            lines.append("")

        def short(v):
            s = json.dumps(v)
            return s if len(s) <= 160 else f"{s[:150]} ... ({len(s)} chars)"
        lines.append("plan:")
        lines += [f"  {k:18s} {short(v)}" for k, v in sorted((man.get("plan") or {}).items())]
        if kind == "scan" and man.get("drive"):
            lines += ["", "drive:"] + [f"  {k:18s} {short(v)}" for k, v in man["drive"].items()]
        if man.get("provenance"):
            lines += ["", "provenance:"] + [f"  {k:18s} {short(v)}" for k, v in man["provenance"].items()]
        lines += ["", "(the full manifest: 'open manifest' above)"]
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))

    # ---------------------------------------------------------------- ramp scans
    def _fill_angles(self, data, keep=False):
        old = set(self.angle_list.get(i) for i in self.angle_list.curselection()) if keep else None
        self.angle_list.delete(0, "end")
        self._angles = list(data["pol"]["theta"])
        for a in self._angles:
            self.angle_list.insert("end", f"{a + 0.0:7.2f} deg")
        for k in range(len(self._angles)):
            if old is None or self.angle_list.get(k) in old:
                self.angle_list.select_set(k)

    def _fill_direct(self, data):
        self.dtable.delete(*self.dtable.get_children())
        for k, p in enumerate(data["direct"]):
            notes = []
            if p["lower"]:
                notes.append("lower bound" + (f" ({p['bound_from']})" if p.get("bound_from") else ""))
            if p["kind"] == "static":
                notes.append(f"angle {p['off_deg']:+.2f} deg from crossed (Imax sin^2 = {p['imin_from_offset_mV']:.3f} mV)")
                if p.get("offset_limited"):
                    notes.append("offset-limited")
            self.dtable.insert("", "end", iid=str(k), values=(
                p["kind"], p["seg"], f"{p['t_ms']:.2f}", f"{p['theta']:.1f}", f"{p['rotation']:+.1f}", f"{p['imin_mV']:.3f}",
                f"{p['sig_mV']:.3f}", f"{p['imax_V']:.3f}", (">" if p["lower"] else "") + f"{p['er']:.0f}", "; ".join(notes)))
        if data.get("direct_err"):
            self.dtable.insert("", "end", values=("", "", "", "", "", "", "", "", "", f"direct ER failed: {data['direct_err']}"))
        elif data["pol"].get("no_fit"):
            self.dtable.insert("", "end", values=("", "", "", "", "", "", "", "", "", "no fit, so no direct ER (needs >= 3 angles)"))

    def _on_direct(self, _e=None):
        sel = self.dtable.selection()
        if not sel or not self.cur or self.cur[0] != "scan":
            return
        try:
            p = self.cur[2]["direct"][int(sel[0])]
        except (ValueError, IndexError):
            return
        self.mark = (p["t_ms"], p["theta"])
        th = np.asarray(self._angles)
        k = int(np.argmin(np.abs((th - p["theta"] + 90) % 180 - 90)))
        self.angle_list.select_clear(0, "end"); self.angle_list.select_set(k); self.angle_list.see(k)
        if self._tab() not in ("Traces", "ER(t)"):
            self.nb.select(self.tabs["Traces"])
        self._refresh_tab()

    def _on_raw(self):
        if self.cur and self.cur[0] == "scan":
            self._pending = os.path.basename(self.cur[1])
            self._start_load()

    def _select_angles(self, how):
        if not self.cur or self.cur[0] != "scan":
            return
        self.angle_list.select_clear(0, "end")
        if how == "all":
            self.angle_list.select_set(0, "end")
        elif how == "near":
            pol = self.cur[2]["pol"]
            if pol.get("no_fit"):
                return
            crossed = (pol["psi_rest"] + 90) % 180
            for k, a in enumerate(self._angles):
                if abs((a - crossed + 90) % 180 - 90) <= 8:
                    self.angle_list.select_set(k)
        self._refresh_tab()

    def _mark(self, *axs):
        if self.mark:
            for ax in axs:
                ax.axvline(self.mark[0], color="r", lw=0.8, ls="--")

    def _draw_traces(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        fig = self.trace_plot.clear(self.cur[1], self._name("traces"))
        t = d.t * 1e3
        has_cmd = any(r in d.roles for r in ("CmdX1", "CmdX2"))
        n = 3 if has_cmd else 2
        gs = fig.add_gridspec(n, 1, height_ratios=[2] + [1] * (n - 1))
        axs = [fig.add_subplot(gs[k]) for k in range(n)]
        for ax in axs[1:]:
            ax.sharex(axs[0])
        sel = list(self.angle_list.curselection())
        th = np.asarray(pol["theta"], float); I = pol["I"]
        # up to 8 angles: a legend; more: colour = analyzer angle mod 180 on a cyclic map, with a colour bar
        few = len(sel) <= 8
        cmap = matplotlib.cm.hsv
        cols = matplotlib.cm.tab10(np.arange(len(sel)) % 10) if few else cmap((th[sel] % 180) / 180)
        for c, k in zip(cols, sel):
            plot_thin(axs[0], t, I[k] * 1e3, color=c, lw=0.6, label=f"{th[k]:.1f} deg" if few else "_")
        axs[0].set(ylabel="PD (mV, minus dark, drift-corrected)" if d.subtract_dark else "PD (mV, raw)",
                   title=f"PD trace per analyzer angle, {d.name} ({len(sel)} of {len(th)} angles)")
        if not few:
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 180))
            cax = axs[0].inset_axes([1.01, 0.0, 0.015, 1.0])      # outside, so the panels stay aligned
            fig.colorbar(sm, cax=cax, label="analyzer angle mod 180 (deg)")
        mons, mon_sum = mon_rotation(d, pol["steps"])
        for r, m in mons.items():
            plot_thin(axs[1], t, m, lw=0.8, label=f"{r}")
        if len(mons) > 1:
            plot_thin(axs[1], t, mon_sum, lw=0.8, color="C1", ls="--", label="monitors summed")
        if pol.get("rotation") is not None:
            plot_thin(axs[1], t, light_sign(pol, mon_sum) * pol["rotation"], "k", lw=0.7, label="light (harmonic fit)")
        axs[1].set(ylabel="rotation (deg)", title="Rotation from the monitors and from the light")
        if has_cmd:
            for r in ("CmdX1", "CmdX2"):
                if r in d.roles:
                    m = np.mean([s["v"][r] for s in pol["steps"]], axis=0)
                    plot_thin(axs[2], t, m, lw=0.8, label=r)
            axs[2].set(ylabel="AWG (V)", title="Command")
        self._mark(*axs)
        for ax in axs:
            ax.set_xlabel("t (ms)"); ax.grid(alpha=0.3); legend(ax)
        self.trace_plot.draw()

    def _no_fit(self, plot, tab):
        d, pol = self.cur[2]["d"], self.cur[2]["pol"]
        fig = plot.clear(self.cur[1], self._name(tab))
        message(fig, f"{d.name}: {pol['no_fit']}.\nNo harmonic fit is possible, so this tab is empty; the Traces tab shows the data.")
        plot.draw()

    def _draw_fit(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        if pol.get("no_fit"):
            return self._no_fit(self.fit_plot, "fit")
        fig = self.fit_plot.clear(self.cur[1], self._name("fit"))
        t = d.t * 1e3
        axs = [fig.add_subplot(3, 2, k + 1) for k in range(6)]
        for ax in axs[1:]:
            ax.sharex(axs[0])
        _mons, mon = mon_rotation(d, pol["steps"])
        light = light_sign(pol, mon) * pol["rotation"]
        plot_thin(axs[0], t, mon, lw=0.9, color="C1", label="monitors")
        plot_thin(axs[0], t, light, lw=0.7, color="C0", label="light")
        axs[0].set(ylabel="rotation (deg)", title=f"Harmonic fit over {len(pol['theta'])} angles, {d.name}")
        lm = (light - mon) * 1e3
        lm -= np.median(lm[d.t < -0.1e-3]) if np.any(d.t < -0.1e-3) else 0.0
        plot_thin(axs[2], t, lm, lw=0.6, color="C3")
        robust_ylim(axs[2], lm, min_span=50)
        axs[2].set(ylabel="light - monitors (mdeg)", title="Light minus monitors, zeroed before t = 0")
        plot_thin(axs[4], t, pol["sig_psi"] * 1e3, lw=0.6, color="C4")
        axs[4].set(ylabel="mdeg", yscale="log", title="Fit uncertainty of the light's rotation")
        plot_thin(axs[1], t, pol["imax"], lw=0.6, color="C0")
        axs[1].set(ylabel="Imax (V)", title="Imax")
        plot_thin(axs[3], t, pol["imin"] * 1e3, lw=0.6, color="C2")
        robust_ylim(axs[3], pol["imin"] * 1e3)
        axs[3].set(ylabel="Imin (mV)", title="Imin")
        plot_thin(axs[5], t, pol["rms"] * 1e3, lw=0.6, color="C5")
        axs[5].set(ylabel="mV", yscale="log", title="Fit residual rms over the angles")
        self._mark(*axs)
        for ax in axs:
            ax.set_xlabel("t (ms)"); ax.grid(alpha=0.3, which="both"); legend(ax)
        self.fit_plot.draw()

    def _draw_er(self):
        data = self.cur[2]
        d, pol = data["d"], data["pol"]
        if pol.get("no_fit"):
            return self._no_fit(self.er_plot, "er_t")
        half, box = float(self.er_half.get()), float(self.er_box.get())
        if data.get("er") is None or data.get("er_key") != (half, box):
            data["er"] = an.er_vs_time(d, pol, half_deg=half, box_us=box, stride=1, gains=pol.get("angle_gain"))
            data["er_key"] = (half, box)
        er = data["er"]
        fig = self.er_plot.clear(self.cur[1], self._name("er_t"))
        t = d.t * 1e3
        axs = [fig.add_subplot(3, 1, k + 1) for k in range(3)]
        for ax in axs[1:]:
            ax.sharex(axs[0])
        ax1, ax2, ax3 = axs
        if len(er["t"]):
            st = max(1, len(er["t"]) // 6000)
            er = {k: (v[::st] if isinstance(v, np.ndarray) else v) for k, v in er.items()}
            te = er["t"] * 1e3
            par = er["method"] == 1
            lo = er["er_lower"]
            ax1.plot(te[par & ~lo], er["er"][par & ~lo], ".", ms=2, color="C0", label=f"parabola through >= 3 angles within +-{half:g} deg")
            ax1.plot(te[~par & ~lo], er["er"][~par & ~lo], ".", ms=2, color="C1", label="nearest angle, offset-corrected")
            ax1.plot(te[lo], er["er"][lo], ".", ms=2, color="0.6", label="lower bound (Imin < 2 sigma)")
            ax2.plot(te, er["imin"] * 1e3, ".", ms=2, color="C0", label="Imin")
            ax2.plot(te, 2 * er["sig_imin"] * 1e3, "-", lw=0.6, color="gray", label="2 sigma")
        for p in data["direct"]:
            if p["kind"] == "static":
                ax1.plot([p["t_ms"]], [p["er"]], "o" if not p["lower"] else "^", ms=8, mfc="r" if not p["lower"] else "none", mec="r",
                         label=f"direct, {p['seg']}: {'>' if p['lower'] else ''}{p['er']:.0f} at {p['theta']:.1f} deg ({p['off_deg']:+.2f} off)")
        vdiv = an._pd_vdiv(d, "scan") or 1.0
        dk = (d.manifest.get("borrowed") or {}).get("background") or {}
        sem_dark = float(dk.get("sem") or 0.0)
        if sem_dark > 0 and len(er["t"]):
            ax1.axhline(float(np.nanmedian(er["imax"])) / sem_dark, color="gray", lw=0.8, ls="--",
                        label=f"Imax / dark SEM ({sem_dark*1e3:.2f} mV at {vdiv:g} V/div)")
            ax2.axhline(sem_dark * 1e3, color="gray", lw=0.8, ls="--", label="dark SEM")
        ax2.axhline(vdiv * 8 / 256 * 1e3, color="gray", lw=0.5, ls=":", label=f"one 8-bit code at {vdiv:g} V/div")
        ax1.set(yscale="log", ylabel="ER", title=f"Extinction ratio vs time, angles near crossed, {d.name}")
        ax2.set(ylabel="Imin (mV)", yscale="symlog", title="Imin and its 2-sigma noise")
        used = sorted({a for _n, angs in er["sets"] for a in angs})
        th = pol["theta"]; I = pol["I"]
        for a in used:
            k = int(np.argmin(np.abs(th - a)))
            plot_thin(ax3, t, I[k] * 1e3, lw=0.6, label=f"{a:.1f} deg")
        if len(er["t"]):
            ax3.plot(t, np.interp(d.t, er["t"], er["imin"], left=np.nan, right=np.nan) * 1e3, "k", lw=0.8, label="Imin(t)")
        ax3.set(ylabel="PD (mV)", yscale="symlog", title="Traces of the angles used")
        self._mark(*axs)
        for ax in axs:
            ax.set_xlabel("t (ms)"); ax.grid(alpha=0.3, which="both"); legend(ax)
        self.er_plot.draw()


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else cfgmod.load()["outdir"]
    app = Viewer(outdir)
    app.mainloop()


if __name__ == "__main__":
    main()

"""Running a scan: analyzer angles, captures, and the manifest that ties them
together.

A scan is a folder <outdir>/<name>/ holding
    <name>_scan.json             the manifest - plan, wiring, every step
    <name>_a045.00_001.npz/.txt  one file per dither block, Scope Grab's format
    <name>_ref003_001.npz/.txt   returns to the reference angle (drift)
    <name>_dark_001.npz/.txt     beam blocked
    <name>_n1_a137.25_001...     null-refine captures for window 1
The capture names split as Scope Grab expects (prefix_NNN), so its Compare box
opens any angle: KEY:RUNS with KEY = <name>_a045.00.

The manifest is rewritten after every step, so a scan that stops - cancelled,
a lost trigger, a crash - can be resumed where it left off.
"""
import datetime
import json
import os
import random
import re
import time

from .config import replace_retrying
from .hw import Cancelled

FORMAT = "rampol-scan/1"


def safe_name(text):
    """A scan name usable as a filename and as a Scope Grab compare key: no
    spaces, nothing Windows refuses."""
    text = re.sub(r'[<>:"/\\|?*]', "", str(text).strip())
    return re.sub(r"\s+", "-", text) or "scan"


def angle_list(start, stop, step):
    """start, start+step, ... up to stop inclusive (to 1e-6 deg)."""
    if step <= 0:
        raise ValueError("angle step must be positive")
    n = int((stop - start) / step + 1e-6) + 1
    return [round(start + i * step, 6) for i in range(max(n, 0))]


def ordered(angles, order, seed=0):
    """Angles in measuring order. 'bidirectional' interleaves the list from
    both ends; 'shuffled' is a seeded random order. Either keeps a slow drift
    from lining up with angle, which a forward sweep would turn into a fake
    1-theta term."""
    a = list(angles)
    if order == "bidirectional":
        out = []
        while a:
            out.append(a.pop(0))
            if a:
                out.append(a.pop())
        return out
    if order == "shuffled":
        random.Random(seed).shuffle(a)
    return a


def build_steps(angles, ref_every=0, ref_angle=45.0):
    """The step list: a reference capture first, after every `ref_every`
    angles, and last (when ref_every > 0); the scan angles between."""
    steps = []
    refs = 0

    def ref():
        nonlocal refs
        steps.append({"kind": "ref", "target": float(ref_angle), "ref": refs})
        refs += 1

    if ref_every > 0:
        ref()
    for i, a in enumerate(angles):
        steps.append({"kind": "scan", "target": float(a)})
        if ref_every > 0 and ((i + 1) % ref_every == 0 or i == len(angles) - 1):
            ref()
    return steps


def stem(name, step):
    kind = step["kind"]
    if kind == "dark":
        return f"{name}_dark"
    if kind == "ref":
        return f"{name}_ref{step['ref']:03d}"
    a = step["target"] % 360.0
    if kind == "null":
        return f"{name}_n{step['window'] + 1}_a{a:06.2f}"
    return f"{name}_a{a:06.2f}"


def column_name(sg, ch, name):
    """The column header Scope Grab's panel would write for this channel."""
    n = sg.safe_column(name or "")
    return f"CH{ch}_{n}_V" if n else f"CH{ch}_V"


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class ScanRun:
    """One scan folder and the hardware to fill it.

    `link` is a hw.ScopeLink, `rot` a hw.Rotator (None for a dark-only run),
    `sg` the loaded scope_grab module (for write_capture). `channels` is
    {ch: (role, name)}. Callbacks: log(text), progress(done, total, text).
    `clock` gives the time recorded with each step (the simulator passes its
    virtual clock so drift correction sees the time it modelled)."""

    def __init__(self, outdir, name, sg, link, rot, channels, log=print,
                 progress=None, cancelled=None, clock=time.time, fmt=".npz"):
        self.name = safe_name(name)
        self.folder = os.path.join(outdir, self.name)
        self.path = os.path.join(self.folder, f"{self.name}_scan.json")
        self.sg, self.link, self.rot = sg, link, rot
        self.channels = dict(channels)
        self.log = log
        self.progress = progress or (lambda *a: None)
        self.cancelled = cancelled or (lambda: False)
        self.clock = clock
        self.fmt = fmt
        self.manifest = None

    # -- manifest ---------------------------------------------------------
    def exists(self):
        return os.path.exists(self.path)

    def load(self):
        with open(self.path, encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        return self.manifest

    def save(self):
        os.makedirs(self.folder, exist_ok=True)
        self.manifest["updated"] = now()
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=1)
        replace_retrying(tmp, self.path)

    def new(self, plan, steps, extra=None):
        """Start a manifest. Refuses to overwrite a scan already on disk."""
        if self.exists():
            raise FileExistsError(f"{self.path} exists - resume it or pick a new name")
        self.manifest = {
            "format": FORMAT, "name": self.name, "created": now(),
            "plan": plan,
            "channels": {str(ch): {"role": r, "name": n}
                         for ch, (r, n) in self.channels.items()},
            "scope": {"idn": getattr(self.link.scope, "idn", ""),
                      "addr": getattr(self.link.scope, "addr", ""),
                      "model": self.link.prof.key},
            "rotator": self._rotator_info(),
            "steps": [dict(s, status="todo") for s in steps],
        }
        self.manifest.update(extra or {})
        self.save()
        return self.manifest

    def _rotator_info(self):
        if self.rot is None:
            return {}
        d = self.rot.dev
        return {"port": str(getattr(d, "port", "")),
                "serial": str(getattr(d, "serial_no", "")),
                "zero_deg": float(self.rot.zero)}

    def add_steps(self, steps):
        self.manifest["steps"] += [dict(s, status="todo") for s in steps]
        self.save()

    # -- running ----------------------------------------------------------
    def run(self, kinds=None):
        """Measure every step not yet done (optionally only of `kinds`).
        Returns the number of steps completed in this call. Raises Cancelled
        if stopped, after saving."""
        m = self.manifest
        plan = m["plan"]
        todo = [s for s in m["steps"] if s["status"] != "done"
                and (kinds is None or s["kind"] in kinds)]
        total = len(todo)
        done = 0
        started = time.time()
        for s in todo:
            if self.cancelled():
                raise Cancelled()
            eta = ""
            if done:
                left = (time.time() - started) / done * (total - done)
                eta = f", ~{left / 60:.1f} min left"
            what = ("dark (beam blocked)" if s["kind"] == "dark"
                    else f"{s['kind']} {s['target']:.2f} deg")
            self.progress(done, total, f"{what} ({done + 1}/{total}{eta})")
            self.measure(s, plan)
            done += 1
            self.save()
        self.progress(done, total, "done")
        return done

    def measure(self, s, plan):
        sg, link = self.sg, self.link
        chans = list(self.channels)
        names = {ch: n for ch, (_, n) in self.channels.items()}
        s["t_start"] = now()
        s["clock"] = float(self.clock())
        if s["kind"] != "dark":
            s["landed"] = float(self.rot.approach(
                s["target"], backoff=plan.get("backoff_deg", 3.0)))
            s["mount"] = float(self.rot.dev.position() + self.rot.zero) % 360.0
        override = s.get("pd_scale")
        saved = None
        if override:
            ch = int(override["ch"])
            saved = link.channel_state([ch])
            link.set_channel(ch, override["vdiv"], override["offset"])
        try:
            settings = link.scope.read_settings()
            # the scales the step ran at, before any dither moved the offsets
            s["scales"] = {str(ch): [_num(settings.get(link.prof.ch_scale.format(ch=ch))),
                                     _num(settings.get(link.prof.ch_offset.format(ch=ch)))]
                           for ch in chans}
            base = os.path.join(self.folder, stem(self.name, s))
            files, hits = [], []
            mode = plan.get("mode", "average")
            blocks = int(s.get("blocks", plan.get("blocks", 4)))
            shots = int(s.get("shots", plan.get("shots", 64)))
            if mode == "single":
                blocks, shots = shots, shots
            os.makedirs(self.folder, exist_ok=True)

            def on_block(k, recs, h):
                # the dither moved the offsets: re-read just those for the sidecar
                for ch in chans:
                    root = link.prof.ch_offset.format(ch=ch)
                    try:
                        settings[root] = link.scope.get(root)
                    except Exception:
                        pass
                if h is not None:
                    settings[link.prof.wave_count] = str(h)
                cols = {column_name(sg, ch, names[ch]): r for ch, r in recs.items()}
                label = (f"{s['kind']} {s.get('target', 0):.3f} deg, block "
                         f"{k + 1}/{blocks}")
                meta = link.scope.metadata(chans, settings, names, label)
                b = f"{base}_{k + 1:03d}"
                path = sg.write_capture(b, self.fmt, cols, meta)
                files.append(os.path.basename(path))
                hits.append(h)

            link.acquire_blocks(chans, mode, blocks, shots,
                                dither_codes=int(plan.get("dither_codes", 3)),
                                points=plan.get("points"),
                                wait_s=float(plan.get("wait_s", 10.0)),
                                cancelled=self.cancelled, on_block=on_block)
        finally:
            if saved:
                for ch, (sc, off) in saved.items():
                    link.set_channel(ch, sc, off)
        s["files"], s["hits"] = files, hits
        s["clock_end"] = float(self.clock())
        s["t_end"] = now()
        s["status"] = "done"

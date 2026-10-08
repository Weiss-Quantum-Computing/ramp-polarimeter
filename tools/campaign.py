"""A measurement campaign without the window: the instruments of the config
(scope, ELL14, 4063B) driven through the same classes the panel uses, through
a list of stages, logging to <outdir>/campaign/<stamp>.log.

    python tools/campaign.py plan.json [--yes] [--simulate] [--only grid1]

plan.json:
    {"stages": [
      {"stage": "grid", "name": "ER_grid_5deg", "x1": "0:90:5", "x2": "0:90:5",
       "how": "grid", "shots": 4, "darks_from": "ER_grid_15deg"},
      {"stage": "ramps", "name": "edges_15deg", "x1": "0:90:15", "x2": "0:90:15",
       "how": "grid", "scan": {"shots": 4, "step": 22.5},
       "awg": {"scope_before_ms": 0.5, "scope_after_ms": 0.5},
       "darks_from": "ER_grid_15deg"}
    ]}

A 'compensate' stage cancels the slow drift at one held point by iterating
a correction on the plateau from the tracked light-minus-monitors (x1, x2,
iterations, beta, done_mdeg, tail_ms). A 'transients' stage tracks the
azimuth through and after holds of a list of lengths (hold_ms_list), the
nulls taken from a finished grid (nulls_from), no null scan.

A 'grid' stage is a Fixed rotations run (rampol.bias.BiasRun): any plan key
of bias.PLAN may be given; `darks_from` reuses another run's darks (no beam
block), else the run asks for one in this terminal. A stage whose folder
exists with its darks on record is resumed. A 'ramps' stage is the AWG
sequence of ramp scans (one per X1 / X2 pair, the Ramp scan settings of the
config under `scan`, the AWG tab's ramp under `awg`), interleaved per
analyzer angle with the hold-null angles, darks borrowed from `darks_from`
(the grid run's coarse dark) or the latest on disk.

Rules kept from the window: never-float (the AWG ends parked, outputs ON at
idle), EOM-ILC's limit check on every waveform, outputs taken over only when
what they play is this program's. No dry run is asked for: say --yes to skip
the 'outputs ON' question too. The beam-block prompts are always asked.

Stop: Ctrl-C. A grid resumes under the same name; a ramps stage resumes
too (steps already measured are skipped).
"""
import argparse
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from rampol import analysis as an, awg as awgmod, bias as biasmod, calib, config as cfgmod  # noqa: E402
from rampol import hw, plan as planmod, provenance, scan as scanmod  # noqa: E402
from rampol import __version__  # noqa: E402


class Log:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, text):
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {text}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def ask_factory(log, yes, ask_dir=None):
    """The questions a run asks. With `ask_dir` (no terminal to type in) the
    question is written to <ask_dir>/ASK.txt and the answer read from
    <ask_dir>/ANSWER.txt (first word y / n) when that file appears."""
    def ask(title, text):
        log(f"? {title}: {text}")
        hands = "block" in text.lower() or "unblock" in text.lower() or "cover" in text.lower()
        if yes and not hands:
            log("  (--yes) yes")
            return True
        if ask_dir:
            q, a = os.path.join(ask_dir, "ASK.txt"), os.path.join(ask_dir, "ANSWER.txt")
            if os.path.exists(a):
                os.remove(a)
            with open(q, "w", encoding="utf-8") as fh:
                fh.write(f"{title}: {text}\nWrite y or n into ANSWER.txt in this folder.\n")
            log(f"  waiting for {a} (y / n)")
            while not os.path.exists(a):
                time.sleep(2)
            time.sleep(0.5)
            with open(a, encoding="utf-8") as fh:
                ans = (fh.read().strip().split() or [""])[0].lower()
            os.remove(a)
            os.remove(q)
            log(f"  answered {ans or 'nothing'}")
            return ans in ("y", "yes")
        try:
            ans = input(f"{title}: {text}\n  [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        log(f"  answered {ans or 'nothing'}")
        return ans in ("y", "yes")
    return ask


def connect(cfg, log, simulate=False):
    """(sg, link, rot, sess, eom, bench) - the bench or the simulator."""
    sg = hw.load_scope_grab(cfg["scope_grab_path"])
    roles = {ch: r for ch, (r, _n) in cfgmod.channel_roles(cfg).items()}
    bench = None
    if simulate:
        from rampol import sim
        bench = sim.Bench(pd_noise=0.2e-3, drift=0.0, er_rest=5000.0, er_mid=300.0,
                          legs_ms=(0.0,))
        scope, ell, bench = sim.make(sg, roles=roles, bench=bench,
                                     zero_offset_deg=float(cfg["ell_zero_deg"]))
        scope.noise_per_div = 0.012
        link = hw.ScopeLink(scope, log=log)
        rot = hw.Rotator(ell, log=log)
        sess = awgmod.Session(sim.FakeAWG(bench), None, log=log, never_float=True,
                              require_dry_run=False)
        eom = None
        log("SIMULATED bench")
    else:
        prof = sg.scope_profiles.get_profile(cfg["scope_model"])
        scope = hw.share_rm(sg.Scope(prof), sg.pyvisa)
        scope.connect(cfg["scope_addr"] or None)
        link = hw.ScopeLink(scope, log=log)
        log(f"Scope: {scope.idn.strip()} at {scope.addr}")
        from rampol.ell14 import ELL14
        dev = ELL14(cfg["ell_port"] or None, address=cfg["ell_address"],
                    zero_offset_deg=float(cfg["ell_zero_deg"]))
        rot = hw.Rotator(dev, log=log)
        info = dev.info()
        log(f"Analyzer: ELL{info['type']} S/N {info['serial']} on {dev.port}, at "
            f"{dev.position():.3f} deg")
        eom = hw.load_eomilc(cfg["eomilc_path"])
        mod = hw.load_module(cfg["awg_path"], "bk4063b")
        import ilc_bench as ib
        ib._AWGMOD = mod
        awg = mod.BK4063B(connect=False, resource_manager=hw.shared_rm(mod.pyvisa))
        log(f"AWG: {awg.connect(cfg.get('awg_addr') or None).strip()} on {awg.resource_name}")
        sess = awgmod.Session(awg, ib, log=log, never_float=True, require_dry_run=False)
        found = sess.take_over()
        if found:
            log("  outputs found ON with this program's waveform: taken over")
    return sg, link, rot, sess, eom, bench


def awg_idle(cfg):
    a = cfg["awg"]
    auto = awgmod.idle_from_states([cfg["ilc"].get("x1"), cfg["ilc"].get("x2")])
    out = {}
    for name, key in (("EO1", "idle1"), ("EO2", "idle2")):
        txt = str(a.get(key, "")).strip()
        out[name] = float(txt) if txt else auto[name]
    return out


# ----------------------------------------------------------------- stages
def stage_grid(st, cfg, parts, log, ask, prov):
    sg, link, rot, sess, eom, bench = parts
    name = scanmod.safe_name(st["name"])
    roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(cfg).items()}
    plan = dict(cfg["bias"])
    plan.pop("name", None)
    for k, v in st.items():
        if k not in ("stage", "name"):
            plan[k] = v
    plan["idle"] = awg_idle(cfg)
    plan["end"] = "park"
    plan.setdefault("sense", float(cfg["awg"].get("seq_sense", -1.0) or -1.0))
    folder = os.path.join(cfg["outdir"], name)
    resume = os.path.isfile(os.path.join(folder, "bias.json"))
    if resume:
        with open(os.path.join(folder, "bias.json"), encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("finished"):
            log(f"Stage {name}: already finished ({old['finished']}) - skipped")
            return
        log(f"Stage {name}: resuming")
    if bench is not None:
        def ask_sim(title, text):
            if "Block the beam" in text:
                bench._imax_saved, bench.imax = bench.imax, 0.0
            elif "Unblock" in text:
                bench.imax = bench._imax_saved
            log(f"? {title}: {text} -> yes (simulated)")
            return True
        ask = ask_sim
    run = biasmod.BiasRun(cfg["outdir"], name, link, rot, sess.awg, roles, plan=plan, log=log,
                          ask=ask, progress=lambda d, t, s: log(f"  [{d}/{t}] {s}"),
                          eomilc=eom, ilc_bench=sess.ib, provenance=prov, session=sess,
                          resume=resume)
    pts = run.run()
    man = biasmod.load(folder)
    from rampol import lablog
    lablog.upsert(cfg["outdir"], lablog.bias_row(man))
    bounds = [p for p in pts if not p.get("er") and p.get("er_lower")]
    log(f"Stage {name} done: {len(pts)} points"
        + (f", {len(bounds)} lower bounds: " + ", ".join(
            f"X1 {p['x1']:g}/X2 {p['x2']:g} > {p['er_lower']:.0f}" for p in bounds[:12])
           if bounds else ""))


def _drive_info(w, a):
    out = {"source": "campaign AWG (rampol)", "label": w.label,
           "names": list(awgmod.names(w)), "record_ms": w.period * 1e3,
           "dt_us": w.dt * 1e6, "idle_V": w.idle(), "dry_run_passed": False,
           "rotation_deg": w.rotation,
           "hold_ms": None if not w.hold else [w.hold[0] * 1e3, w.hold[1] * 1e3],
           "ends_deg": {"X1": w.ends["EO1"], "X2": w.ends["EO2"]},
           "ramp": {k: a.get(k) for k in ("rotation", "split", "edge", "lead_ms", "rise_ms",
                                          "hold_ms", "fall_ms", "tail_ms")}}
    if getattr(w, "files", None):
        out["files"] = dict(w.files)
        out.pop("ramp", None)
    return out


def _borrow(cfg, sg, link, roles, runs, lender, log):
    """Darks for the ramp scans: the grid run's coarse dark when its V/div and
    offset match the PD's now, else the latest on disk (find_offsets)."""
    pd = roles["PD"]
    vdiv, off = link.channel_state([pd])[pd]
    got = {}
    if lender:
        lp = os.path.join(cfg["outdir"], lender, "bias.json")
        try:
            with open(lp, encoding="utf-8") as fh:
                lm = json.load(fh)
            key = biasmod._key((vdiv, off))
            d = (lm.get("dark") or {}).get(key)
            if d:
                got["background"] = {"level": float(d[0]), "sem": float(d[1]),
                                     "n": int((lm.get("plan") or {}).get("shots", 0)),
                                     "vdiv": float(vdiv), "offset": float(off),
                                     "source": lender, "measured": lm.get("created", "")}
                log(f"  background: the grid run {lender}'s dark at {vdiv:g} V/div, offset "
                    f"{off:+.4g} V: {d[0]*1e3:+.2f} mV")
            else:
                log(f"  {lender} has no dark at {vdiv:g} V/div, offset {off:+.4g} V "
                    f"(it has {', '.join(lm.get('dark', {}))}) - the latest on disk instead")
        except (OSError, ValueError) as exc:
            log(f"  darks from {lender} not read ({exc})")
    if "background" not in got:
        found = an.find_offsets(cfg["outdir"], vdiv, off, sg.load_capture, exclude=runs[0].folder)
        for kind in ("background", "dark"):
            f = next((x for x in found if x["kind"] == kind), None)
            if f:
                got[kind] = {k: f[k] for k in ("level", "sem", "n", "vdiv", "offset", "source",
                                                "measured")}
                log(f"  {kind}: reusing {f['level']*1e3:+.2f} mV from {f['source']} ({f['measured']})")
    st = an.find_stray(cfg["outdir"], sg.load_capture, vdiv, exclude=runs[0].folder)
    if st:
        got["stray"] = st
    for r in runs:
        r.manifest.setdefault("borrowed", {}).update(
            {k: (dict(v) if isinstance(v, dict) else v) for k, v in got.items()})
        r.save()


def stage_ramps(st, cfg, parts, log, ask, prov):
    sg, link, rot, sess, eom, bench = parts
    base = scanmod.safe_name(st["name"])
    a = dict(cfg["awg"], **(st.get("awg") or {}))
    sc = dict(cfg["scan"], **(st.get("scan") or {}))
    idle = awg_idle(cfg)
    ends = [(float(x), float(y)) for x, y in
            awgmod.parse_ends(st["x1"], st.get("x2", "0"), st.get("how", "pairs"))]
    # rise_ms_list: the same end points with every edge length in the list
    # (fall = rise), named ..._r<ms> - the slew sweep of 8 Oct 2026
    rises = [float(x) for x in (st.get("rise_ms_list") or [])]
    waves, pairs, names = [], [], []
    for r in rises or [None]:
        ar = dict(a) if r is None else dict(a, rise_ms=r, fall_ms=r)
        for e1, e2 in ends:
            waves.append(awgmod.ramp_hold(0.0, ar, idle=idle, ends={"EO1": e1, "EO2": e2}))
            pairs.append((e1, e2))
            names.append(scanmod.safe_name(f"{base}_X1_{e1:g}_X2_{e2:g}"
                                           + ("" if r is None else f"_r{r:g}".replace(".", "p"))))
    for w in waves:
        found = awgmod.check(w, eom, trig_hz=float(a.get("trig_hz") or 0) or None)
        bad = [m for lv, m in found if lv == "FAIL"]
        if bad and st.get("allow_fail"):
            log(f"  ! {w.label}: " + "; ".join(bad) + " - PLAYED ANYWAY (allow_fail: the Trek "
                "limits itself; the slew sweep asks for exactly this)")
        elif bad:
            raise RuntimeError(f"{w.label}: " + "; ".join(bad))
    return _ramp_sequence(st, cfg, parts, log, prov, base, waves, pairs, names, sc, a)


def _read_target(path):
    """(time_us, volts) of an ILC target / drive CSV: '#' comments, a
    'time_us,voltage_V' header, then the samples."""
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            f = line.strip().split(",")
            if line.startswith("#") or len(f) < 2:
                continue
            try:
                rows.append((float(f[0]), float(f[1])))
            except ValueError:
                continue
    a = np.array(rows, float)
    return a[:, 0], a[:, 1]


def stage_targets(st, cfg, parts, log, ask, prov):
    """The suite's ILC target pairs (target_<stem>X1.csv / X2.csv in `dir`:
    HV volts on a 2 us grid, each channel scaled to its V90, EO zero not
    applied) played as drives - HV V / 1000 = monitor V, / the chain's gain
    = AWG V, no learned correction - one ramp scan each, interleaved, the
    hold-null angles from each pair's peak rotation. `stems`: a list, or
    "all" for every pair in the folder."""
    sg, link, rot, sess, eom, bench = parts
    base = scanmod.safe_name(st["name"])
    d = st["dir"]
    a = dict(cfg["awg"], **(st.get("awg") or {}))
    sc = dict(cfg["scan"], **(st.get("scan") or {}))
    stems = st.get("stems") or "all"
    scale = float(st.get("scale", 1.0))
    if stems == "all":
        stems = sorted({f[len("target_"):-len("X1.csv")] for f in os.listdir(d)
                        if f.startswith("target_") and f.endswith("X1.csv")})
    waves, ends, names = [], [], []
    for stem in stems:
        t, u, files = {}, {}, {}
        for n, suf in (("EO1", "X1"), ("EO2", "X2")):
            files[n] = os.path.join(d, f"target_{stem}{suf}.csv")
            tt, v = _read_target(files[n])
            t[n], u[n] = tt, scale * v / 1000.0 / biasmod.CHAN[n]["gain"]
        if len(t["EO1"]) != len(t["EO2"]) or np.abs(t["EO1"] - t["EO2"]).max() > 1e-6:
            raise RuntimeError(f"{stem}: X1 and X2 targets are not on one time grid")
        dt = float(t["EO1"][1] - t["EO1"][0]) * 1e-6
        e = {n: 90.0 * float(u[n].max()) * biasmod.CHAN[n]["gain"] / biasmod.CHAN[n]["v90"]
             for n in u}
        w = awgmod.Wave(t["EO1"] * 1e-6, u, dt,
                        f"target {stem}" + (f" x {scale:g}" if scale != 1.0 else ""),
                        rotation=e["EO1"] + e["EO2"], source="files", files=files)
        w.ends = e
        found = awgmod.check(w, eom, trig_hz=float(a.get("trig_hz") or 0) or None)
        bad = [m for lv, m in found if lv == "FAIL"]
        if bad:
            raise RuntimeError(f"{w.label}: " + "; ".join(bad))
        log(f"  {stem}: {w.n} points at {dt*1e6:g} us = {w.period*1e3:.3f} ms, peaks X1 "
            f"{e['EO1']:.1f} / X2 {e['EO2']:.1f} deg, AWG {u['EO1'].max():.2f} / {u['EO2'].max():.2f} V"
            + (f" (target x {scale:g})" if scale != 1.0 else ""))
        waves.append(w)
        ends.append((e["EO1"], e["EO2"]))
        names.append(scanmod.safe_name(f"{base}_{stem}"))
    return _ramp_sequence(st, cfg, parts, log, prov, base, waves, ends, names, sc, a)


def _ramp_sequence(st, cfg, parts, log, prov, base, waves, ends, names, sc, a):
    """One ramp scan per wave, interleaved per analyzer angle, the AWG loaded
    live between them (stage_ramps and stage_targets)."""
    sg, link, rot, sess, eom, bench = parts
    channels = cfgmod.channel_roles(cfg)
    roles = {r: ch for ch, (r, _n) in channels.items()}
    chroles = {ch: r for ch, (r, _n) in channels.items()}
    # the scope as a ramp scan takes it: the preset, then the span around the record
    pre = cfgmod.all_presets(cfg).get(cfg.get("preset"))
    if pre:
        writes = link.preset_writes(pre, chroles)
        bad, errs = link.apply_checked(writes)
        log(f"  scope set from the preset '{cfg.get('preset')}' ({len(writes)} settings"
            + (", every one read back)" if not bad and not errs else ")"))
        for root, (want, got) in bad.items():
            log(f"  ! {root}: wrote {want}, scope reads {got}")
    lender = st.get("darks_from")
    if lender:
        # the PD at the grid run's coarse setting, so its dark applies exactly
        # (the scope's offset error moves with V/div and offset)
        try:
            with open(os.path.join(cfg["outdir"], lender, "bias.json"), encoding="utf-8") as fh:
                lc = json.load(fh).get("coarse")
            pd = roles["PD"]
            now = link.channel_state([pd])[pd]
            if lc and abs(float(lc[0]) - now[0]) < 1e-9 and abs(float(lc[1]) - now[1]) > 1e-4:
                link.set_channel(pd, float(lc[0]), float(lc[1]))
                log(f"  PD offset {now[1]:+.4g} -> {float(lc[1]):+.4g} V, the grid run "
                    f"{lender}'s, so its dark applies")
        except (OSError, ValueError, TypeError) as exc:
            log(f"  {lender}'s coarse setting not read ({exc})")
    longest = max(waves, key=lambda w: w.period)
    div, pos = awgmod.timebase_for(longest, float(a.get("scope_before_ms", 0.2) or 0),
                                   float(a.get("scope_after_ms", 0.0) or 0))
    scope = link.scope
    scope.put(":TIMebase:REFerence", "LEFT")
    scope.put(":TIMebase:SCALe", f"{div:.6g}")
    scope.put(":TIMebase:POSition", f"{pos:.6g}")
    log(f"  scope timebase {div*1e3:g} ms/div from {(pos - div)*1e3:+.2f} ms to "
        f"{(pos + 9 * div)*1e3:+.2f} ms (record {longest.period*1e3:.3f} ms)")
    rate = max(awgmod.peak_rate(w) for w in waves)
    dt_us, per = planmod.sampling(10 * div, int(sc.get("points") or 0), rate)
    log(f"  sampling: {dt_us:.2f} us, {per:.2f} deg per sample at {rate:.0f} deg/ms"
        + (" - WARN: edges sampling-limited" if per > planmod.SAMPLING_WARN_DEG else ""))
    angles = scanmod.ordered(scanmod.angle_list(sc["start"], sc["stop"], sc["step"]), sc["order"])
    grid = scanmod.angle_list(sc["start"], sc["stop"], sc["step"])
    crossed = float(a.get("seq_crossed_deg", 0.0) or 0)
    sense = float(a.get("seq_sense", -1.0) or -1)
    extras = [scanmod.hold_angles(crossed, e1 + e2, float(a.get("seq_null_half_deg", 3.0) or 0),
                                  int(a.get("seq_null_points", 3) or 1), grid, sense=sense)
              if a.get("seq_null_angles", True) else [] for e1, e2 in ends]
    runs = []
    here = {}
    plan = dict(sc, preset=cfg.get("preset"), software=f"rampol {__version__} campaign",
                dark_mode="reuse latest", bg_mode="reuse latest")
    for k, (n, w, (e1, e2), x) in enumerate(zip(names, waves, ends, extras)):
        r = scanmod.ScanRun(cfg["outdir"], n, sg, link, rot, channels, log=log)
        r.here = here
        if r.exists():
            r.load()
            log(f"  {n}: on disk, {sum(1 for s in r.manifest['steps'] if s['status'] == 'done')} "
                f"steps done - resumed")
        else:
            steps = scanmod.build_steps(angles, int(sc["ref_every"]), sc["ref_angle"], extra=x)
            pl = dict(plan, series={"base": base, "index": k, "of": len(ends),
                                    "ends_deg": {"X1": e1, "X2": e2},
                                    "order": "interleaved (per angle)", "members": names},
                      hold_angles=list(x))
            r.new(pl, steps, extra={"zero_deg": float(cfg["ell_zero_deg"]), "provenance": prov,
                                    "drive": _drive_info(w, a)})
        runs.append(r)
    _borrow(cfg, sg, link, roles, runs, st.get("darks_from"), log)
    steps = [[s for s in r.manifest["steps"] if s["kind"] in ("scan", "ref")] for r in runs]
    order = planmod.interleave(steps, "interleaved")
    todo = [(k, i) for k, i in order if steps[k][i].get("status") != "done"]
    log(f"Stage {base}: {len(runs)} ramp scans, {len(todo)} steps to measure")
    settle = float(a.get("seq_settle_s", 1.0) or 0)
    cur, t0 = None, time.time()
    try:
        for n_done, (k, i) in enumerate(todo):
            if cur != k:
                if sess.wave is None or awgmod.names(sess.wave) != awgmod.names(waves[k]):
                    sess.load(waves[k], keep_on=True)
                if not all(sess.outputs().values()):
                    sess.on()
                cur = k
                if settle > 0:
                    time.sleep(settle)
            s = steps[k][i]
            left = ""
            if n_done:
                left = f", ~{(time.time() - t0) / n_done * (len(todo) - n_done) / 60:.0f} min left"
            log(f"  [{n_done + 1}/{len(todo)}{left}] {runs[k].name}: {s['kind']} {s['target']:.2f} deg")
            runs[k].run_step(s)
    finally:
        sess.end("park")
    log(f"Stage {base} done")


def _bias_run(cfg, parts, log, ask, prov, name, plan, resume=False):
    sg, link, rot, sess, eom, bench = parts
    roles = {r: ch for ch, (r, _n) in cfgmod.channel_roles(cfg).items()}
    if bench is not None:
        def ask_sim(title, text):
            if "Block the beam" in text:
                bench._imax_saved, bench.imax = bench.imax, 0.0
            elif "Unblock" in text:
                bench.imax = bench._imax_saved
            return True
        ask = ask_sim
    run = biasmod.BiasRun(cfg["outdir"], name, link, rot, sess.awg, roles, plan=plan, log=log,
                          ask=ask, progress=lambda d, t, s: log(f"  [{d}/{t}] {s}"),
                          eomilc=eom, ilc_bench=sess.ib, provenance=prov, session=sess,
                          resume=resume)
    return run.run()


def stage_compensate(st, cfg, parts, log, ask, prov):
    """Cancel the slow drift at one held point by pre-distorting the drive:
    each iteration measures the point (Fixed rotations, the azimuth tracked
    at the hold's null and at the rest null) and adds -beta x the tracked
    light-minus-monitors (deg -> AWG volts of the held crystal) to the
    plateau: through the hold (its creep, referenced to the hold's first
    ms) and through the tail after the fall (the after-fall offset and its
    relaxation, referenced to the lead). The edges are left alone. The tail
    of the plateau is long (tail_ms) so the correction has room after the
    fall. Stops when the error's rms is under `done_mdeg` or after
    `iterations`. Records every iteration's point and the correction (CSV,
    JSON) under <outdir>/<name>/."""
    sg, link, rot, sess, eom, bench = parts
    base = scanmod.safe_name(st["name"])
    e1, e2 = float(st.get("x1", 0) or 0), float(st.get("x2", 0) or 0)
    chan = "EO1" if e1 or not e2 else "EO2"
    plan = dict(biasmod.PLAN, **cfg["bias"])
    plan.pop("name", None)
    skip = ("stage", "name", "iterations", "beta", "done_mdeg", "smooth_ms")
    plan.update({k: v for k, v in st.items() if k not in skip})
    plan.update(x1=f"{e1:g}", x2=f"{e2:g}", how="pairs", idle=awg_idle(cfg), end="park",
                predict_null=False)
    plan.setdefault("tail_ms", 150.0)
    plan.setdefault("dt_us", 20.0)
    plan.setdefault("track_ms", 20.0)
    plan.setdefault("sense", float(cfg["awg"].get("seq_sense", -1.0) or -1.0))
    iters = int(st.get("iterations", 4))
    beta = float(st.get("beta", 0.7))
    done_mdeg = float(st.get("done_mdeg", 10.0))
    smooth_ms = float(st.get("smooth_ms", 1.0))
    cal = calib.get(cfg)
    dpv = calib.deg_per_awg_v(cal, chan)
    folder = os.path.join(cfg["outdir"], base)
    os.makedirs(folder, exist_ok=True)
    hist_path = os.path.join(folder, "compensation.json")
    hist = {"name": base, "x1": e1, "x2": e2, "channel": chan, "deg_per_V": dpv, "beta": beta,
            "iterations": []}
    if os.path.isfile(hist_path):
        with open(hist_path, encoding="utf-8") as fh:
            hist = json.load(fh)
        if hist.get("finished"):
            log(f"Stage {base}: already finished - skipped")
            return
    corr = hist.get("correction")          # {t_s, u_V} of the last iteration, or None
    t_hold0 = (plan["lead_ms"] + plan["rise_ms"]) * 1e-3
    t_hold1 = t_hold0 + plan["hold_ms"] * 1e-3
    t_fall = t_hold1 + plan["rise_ms"] * 1e-3
    for k in range(len(hist["iterations"]), iters):
        it_plan = dict(plan, correction={chan: corr} if corr else None)
        name = f"{base}_it{k}"
        log(f"Compensation {base}, iteration {k}: "
            + ("no correction yet" if corr is None else
               f"correction peak {np.max(np.abs(corr['u_V'])) * 1e3:.1f} mV"))
        pts = _bias_run(cfg, parts, log, ask, prov, name, it_plan,
                        resume=os.path.isfile(os.path.join(cfg["outdir"], name, "bias.json")))
        tr = biasmod.load_tracks(os.path.join(cfg["outdir"], name)).get(0)
        if tr is None:
            raise RuntimeError(f"{name}: no tracked azimuth on record")
        t = tr["t"]
        # The error is the LIGHT against the commanded rotation - not against
        # the monitors: a corrected drive moves the monitors with the light,
        # so light minus monitors cannot change (iteration 1 of 8 Oct 2026:
        # creep -17.6 -> -16.2 mdeg/ms). The light's rotation in the
        # monitors' sense is sense x the null's displacement (dpsi). Hold:
        # the creep from the hold pair, referenced to the hold's first ms
        # after the rise (the command is flat there); tail: the rest pair
        # (when there is one) referenced to the lead, where the command is 0.
        sense = float(pts[-1].get("sense", plan.get("sense", -1.0)))
        err = np.zeros(len(t))
        hold = (t >= t_hold0 + 0.5e-3) & (t <= t_hold1 - 0.1e-3)
        ref = (t >= t_hold0 + 0.5e-3) & (t <= t_hold0 + 1.5e-3)
        err[hold] = sense * (tr["dpsi"][hold] - tr["dpsi"][ref].mean())
        t_end = t_fall + float(plan["tail_ms"]) * 1e-3 + float(plan.get("track_ms") or 0) * 1e-3
        tail = (t >= t_fall + 1.0e-3) & (t <= t_end - 0.5e-3)
        if "dpsi_rest" in tr:
            v = tr.get("valid_rest", np.ones(len(t), bool))
            err[tail & v] = sense * tr["dpsi_rest"][tail & v]
        elif abs(e1 + e2) <= 25.0:
            err[tail] = sense * (tr["dpsi"][tail] - tr["dpsi"][t < t_hold0 - 0.2e-3].mean())
        # the light's own creep and tail, for the record
        pf = np.polyfit(t[hold] * 1e3, err[hold] * 1e3, 1) if hold.sum() > 5 else (np.nan, np.nan)
        light_creep = float(pf[0])
        light_after = float(np.interp(t_fall + 1.0e-3, t, err) * 1e3)
        # smooth over smooth_ms (the per-sample azimuth is ~10 mdeg of noise)
        dt = float(np.median(np.diff(t)))
        n = max(1, int(round(smooth_ms * 1e-3 / dt)))
        if n > 1:
            box = np.ones(n) / n
            err = np.convolve(np.r_[np.full(n, err[0]), err, np.full(n, err[-1])], box,
                              mode="same")[n:-n]
        err[~(hold | tail)] = 0.0
        rms = float(np.sqrt(np.mean(err[hold | tail] ** 2))) * 1e3
        pt = pts[-1]
        tk = pt.get("track") or {}
        rec = {"iteration": k, "run": name, "err_rms_mdeg": rms,
               "light_creep_mdeg_ms": light_creep, "light_after_1ms_mdeg": light_after,
               "creep_mdeg_ms": tk.get("hold_slope_mdeg_ms"),
               "after_1ms_mdeg": tk.get("after_1ms_mdeg"),
               "after_extreme_mdeg": tk.get("after_extreme_mdeg"), "tau_ms": tk.get("tau_ms"),
               "er": pt.get("er") or pt.get("er_lower"), "imin_mV": pt["imin"] * 1e3,
               "null_deg": pt["theta_n"], "phi_mon_deg": pt.get("phi_mon")}
        hist["iterations"].append(rec)
        log(f"  iteration {k}: light - command rms {rms:.1f} mdeg over hold + tail; the "
            f"light's creep {light_creep:+.1f} mdeg/ms, 1 ms after the fall {light_after:+.0f} "
            f"mdeg (light - monitors: creep {tk.get('hold_slope_mdeg_ms', float('nan')):+.1f}, "
            f"after-fall {tk.get('after_1ms_mdeg', float('nan')):+.0f}), ER {rec['er']:.0f}")
        # the update: the error in deg -> volts on the driven channel, on the
        # scope's time grid (the plateau interpolates it onto its own)
        u_prev = np.zeros(len(t)) if corr is None else np.interp(
            t, corr["t_s"], corr["u_V"], left=0.0, right=0.0)
        u_new = u_prev - beta * err / dpv
        corr = {"t_s": t.tolist(), "u_V": u_new.tolist()}
        hist["correction"] = corr
        with open(os.path.join(folder, f"correction_it{k + 1}.csv"), "w", encoding="utf-8") as fh:
            fh.write(f"# {base}: correction on {chan} after iteration {k} (AWG volts added to "
                     f"the plateau), {dpv:.4f} deg per V\n")
            fh.write("time_s,u_V\n")
            for a, b in zip(t, u_new):
                fh.write(f"{a:.6e},{b:.6e}\n")
        with open(hist_path, "w", encoding="utf-8") as fh:
            json.dump(hist, fh, indent=1)
        if rms < done_mdeg:
            log(f"  error under {done_mdeg:g} mdeg: done")
            break
    hist["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(hist, fh, indent=1)
    log(f"Stage {base} done: " + "; ".join(f"it{r['iteration']} {r['err_rms_mdeg']:.0f} mdeg"
                                           for r in hist["iterations"]))


def stage_transients(st, cfg, parts, log, ask, prov):
    """The azimuth against time at held points for a list of hold lengths
    (hold_ms_list, e.g. 0.5 .. 200 ms), tracked only: no null scan, the
    hold's null taken from the run `nulls_from` (a finished grid) for each
    (x1, x2), the two slope pairs read at the coarse V/div (no overdrive, so
    holds under a millisecond are fine) through the hold and after the fall
    for as long as the trigger period allows (period_ms, default 270). One
    Fixed rotations run per hold length, named <name>_h<hold>."""
    hold_list = [float(x) for x in st["hold_ms_list"]]
    period = float(st.get("period_ms", 270.0))
    nulls = []
    if st.get("nulls_from"):
        with open(os.path.join(cfg["outdir"], st["nulls_from"], "bias.json"), encoding="utf-8") as fh:
            nulls = [[q["x1"], q["x2"], q["theta_n"]] for q in json.load(fh).get("points", [])
                     if q.get("theta_n") is not None]
        log(f"  nulls from {st['nulls_from']}: {len(nulls)} points")
    for h in hold_list:
        tail = float(st.get("tail_ms", 0.5))
        rec = float(st.get("lead_ms", 0.5)) + 2 * float(st.get("rise_ms", 1.0)) + h + tail
        track_ms = max(5.0, min(float(st.get("track_ms", 200.0)), period - rec - 10.0))
        plan = dict(biasmod.PLAN, **cfg["bias"])
        plan.pop("name", None)
        skip = ("stage", "name", "hold_ms_list", "period_ms", "nulls_from")
        plan.update({k: v for k, v in st.items() if k not in skip})
        plan.update(hold_ms=h, settle_ms=0.0, tail_ms=tail, track_ms=track_ms, track=True,
                    track_only=True, idle=awg_idle(cfg), end="park", nulls=nulls,
                    sense=float(cfg["awg"].get("seq_sense", -1.0) or -1.0))
        plan.setdefault("dt_us", 10.0 if h >= 50 else 2.0)
        name = f"{scanmod.safe_name(st['name'])}_h{h:g}".replace(".", "p")
        folder = os.path.join(cfg["outdir"], name)
        if os.path.isfile(os.path.join(folder, "bias.json")):
            with open(os.path.join(folder, "bias.json"), encoding="utf-8") as fh:
                if json.load(fh).get("finished"):
                    log(f"Stage {name}: already finished - skipped")
                    continue
        log(f"Transients {name}: hold {h:g} ms, tracked {track_ms:g} ms after the record")
        _bias_run(cfg, parts, log, ask, prov, name, plan,
                  resume=os.path.isfile(os.path.join(folder, "bias.json")))
    log(f"Stage {st['name']} done: holds " + ", ".join(f"{h:g}" for h in hold_list) + " ms")


def stage_frf(st, cfg, parts, log, ask, prov):
    """The polarization's transfer function against the monitors and the
    command: a multisine (modulation: chan, f_hz, deg, start_ms, stop_ms)
    on one channel through the hold at each held point (x1 / x2 as a grid
    run), tracked only at the hold's slope pair, the traces saved at full
    rate (save_decimate 1). The hold's null comes from `nulls_from`.
    Offline: frf_readout.py takes light / monitors and monitors / command
    per tone."""
    nulls = []
    if st.get("nulls_from"):
        with open(os.path.join(cfg["outdir"], st["nulls_from"], "bias.json"), encoding="utf-8") as fh:
            nulls = [[q["x1"], q["x2"], q["theta_n"]] for q in json.load(fh).get("points", [])
                     if q.get("theta_n") is not None]
        log(f"  nulls from {st['nulls_from']}: {len(nulls)} points")
    plan = dict(biasmod.PLAN, **cfg["bias"])
    plan.pop("name", None)
    skip = ("stage", "name", "nulls_from")
    plan.update({k: v for k, v in st.items() if k not in skip})
    plan.update(settle_ms=0.0, track=True, track_only=True, idle=awg_idle(cfg), end="park",
                nulls=nulls, save_decimate=1,
                sense=float(cfg["awg"].get("seq_sense", -1.0) or -1.0))
    plan.setdefault("track_ms", 5.0)
    f, amp, _ph = biasmod.multisine(plan)
    m = plan.get("modulation") or {}
    chan = m.get("chan", "EO1")
    t, u = biasmod.plateau(0.0, plan)
    um = biasmod.modulation_on(t, plan, chan)
    hv_per_v = 1000.0 * biasmod.CHAN[chan]["gain"]
    slew = float(np.max(np.abs(np.diff(um)))) / (plan["dt_us"]) * hv_per_v      # V/us at the Trek
    log(f"FRF {st['name']}: {len(f)} tones {f.min():.0f}-{f.max():.0f} Hz on {chan}, "
        f"{amp.min():.2f}-{amp.max():.2f} deg each, peak {np.max(np.abs(um)):.3f} V at the AWG "
        f"({np.max(np.abs(um)) * hv_per_v:.0f} V HV), peak slew {slew:.2f} V/us "
        f"({slew * 0.2:.2f} mA into 200 pF)")
    name = scanmod.safe_name(st["name"])
    folder = os.path.join(cfg["outdir"], name)
    if os.path.isfile(os.path.join(folder, "bias.json")):
        with open(os.path.join(folder, "bias.json"), encoding="utf-8") as fh:
            if json.load(fh).get("finished"):
                log(f"Stage {name}: already finished - skipped")
                return
    _bias_run(cfg, parts, log, ask, prov, name, plan,
              resume=os.path.isfile(os.path.join(folder, "bias.json")))
    log(f"Stage {name} done")


STAGES = {"grid": stage_grid, "ramps": stage_ramps, "targets": stage_targets,
          "compensate": stage_compensate, "transients": stage_transients, "frf": stage_frf}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--yes", action="store_true", help="answer yes to everything but a beam block")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--only", help="run only the stage(s) named (comma-separated)")
    ap.add_argument("--config", help="a config.json other than the panel's")
    ap.add_argument("--ask-file", action="store_true",
                    help="no terminal: questions through <outdir>/campaign/ASK.txt / ANSWER.txt")
    args = ap.parse_args(argv)
    with open(args.plan, encoding="utf-8") as fh:
        plan = json.load(fh)
    if args.config:
        cfgmod.CONFIG_PATH = args.config
    cfg = cfgmod.load()
    calib.apply(calib.get(cfg), cfg)
    os.makedirs(os.path.join(cfg["outdir"], "campaign"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = Log(os.path.join(cfg["outdir"], "campaign", f"{stamp}.log"))
    log(f"Campaign {os.path.basename(args.plan)}: {len(plan['stages'])} stages; outdir {cfg['outdir']}")
    ask = ask_factory(log, args.yes,
                      os.path.join(cfg["outdir"], "campaign") if args.ask_file else None)
    parts = connect(cfg, log, simulate=args.simulate)
    prov = provenance.collect(cfg)
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    sess = parts[3]
    try:
        for st in plan["stages"]:
            if only and st.get("name") not in only:
                continue
            fn = STAGES.get(st.get("stage"))
            if fn is None:
                raise ValueError(f"unknown stage kind {st.get('stage')!r}")
            log(f"=== stage {st.get('stage')} {st.get('name')}")
            fn(st, cfg, parts, log, ask, prov)
    except KeyboardInterrupt:
        log("Stopped (Ctrl-C). Start again with the same plan to resume.")
        return 1
    finally:
        try:
            sess.end("park")
        except Exception as exc:
            log(f"  AWG park at the end: {exc}")
        log("AWG parked (outputs ON at idle)")
    log("Campaign done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

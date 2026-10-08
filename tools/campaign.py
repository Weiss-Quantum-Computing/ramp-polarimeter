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
    waves = [awgmod.ramp_hold(0.0, a, idle=idle, ends={"EO1": e1, "EO2": e2}) for e1, e2 in ends]
    for w in waves:
        found = awgmod.check(w, eom, trig_hz=float(a.get("trig_hz") or 0) or None)
        bad = [m for lv, m in found if lv == "FAIL"]
        if bad:
            raise RuntimeError(f"{w.label}: " + "; ".join(bad))
    names = [scanmod.safe_name(f"{base}_X1_{e1:g}_X2_{e2:g}") for e1, e2 in ends]
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


STAGES = {"grid": stage_grid, "ramps": stage_ramps}


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

"""The window, driven against the simulator: connect, dark, scan, load, every
tab drawn, null refine. Needs a desktop session for Tk; no instrument.

The config is sandboxed BEFORE the window exists - building the App against
the real %APPDATA% config would rewrite it on close.
The window is built off screen (+6000+200) and never raised, so it cannot
cover a panel in use on this PC.

    python tests/test_gui.py
"""
import os
import sys
import tempfile
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import config as cfgmod  # noqa: E402

SANDBOX = tempfile.mkdtemp(prefix="rampol-gui-")
cfgmod.CONFIG_PATH = os.path.join(SANDBOX, "config.json")

cfgmod.DEFAULTS["autoconnect"] = False      # never reach for the real scope from a test
from rampol import gui  # noqa: E402
from rampol import analysis as an  # noqa: E402

# the dark prompts answer themselves
gui.messagebox.askokcancel = lambda *a, **k: True
gui.messagebox.showinfo = lambda *a, **k: None
gui.messagebox.askyesno = lambda *a, **k: True
gui.messagebox.showwarning = lambda *a, **k: None
gui.messagebox.showerror = lambda *a, **k: None
# Export brief and Lab log open Explorer / Excel: not from a test
os.startfile = lambda *a, **k: None

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def settle(root, app, timeout=120):
    t0 = time.time()
    root.update()
    while (app.busy or not app.calls.empty()) and time.time() - t0 < timeout:
        root.update()
        time.sleep(0.01)
    # the pump runs every 80 ms: give it time to flush the log and callbacks
    t1 = time.time()
    while time.time() - t1 < 0.4 or app.busy or not app.calls.empty():
        root.update()
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            break


def main():
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)
    root.geometry("+6000+200")
    check("config is the sandbox", cfgmod.CONFIG_PATH.startswith(SANDBOX))
    app.simulate.set(True)
    app.outdir.set(os.path.join(SANDBOX, "data"))
    for k, v in {"start": "0", "stop": "330", "step": "30", "ref_every": "4",
                 "shots": "64", "blocks": "4", "rep_s": "0.27"}.items():
        app.sv[k].set(v)
    app.mode.set("average")
    app.scan_name.set("gui test")
    # the simulated bench has no chain offsets for an idle trim to cancel
    app.av["idle1"].set("0")
    app.av["idle2"].set("0")
    print("\nconnect (as on open)")
    app.auto_connect()
    settle(root, app)
    app.bench.rng = __import__("numpy").random.default_rng(5)
    app._sim_parts[0].realtime = 0.0
    check("scope connected (simulated)", app.link is not None,
          app.scope_status.cget("text"))
    check("analyzer connected (simulated)", app.rot is not None,
          app.ell_status.cget("text"))
    app.do_connect_awg()
    settle(root, app)
    check("AWG Connect in the hardware pane: session up, outputs untouched",
          app.awg_sess is not None and not app.awg_sess.owned
          and "4063B" in app.awg_hw.cget("text"), app.awg_hw.cget("text"))
    app.goto_var.set("123.4")
    app.do_goto()
    settle(root, app)
    pos = float(app.pos_label.cget("text").split()[1])
    check("go to lands within 0.05 deg", abs(pos - 123.4) < 0.05, f"{pos:.3f}")
    app.pos_label._copy()
    check("the position can be copied", root.clipboard_get() == app.pos_label.cget("text"),
          root.clipboard_get())
    app.step_var.set("0.25")
    for _ in range(4):
        app.do_step(+1)
        settle(root, app)
    pos = float(app.pos_label.cget("text").split()[1])
    check("four +0.25 steps count from the target, not the landings",
          abs(app.target - 124.4) < 1e-9 and abs(pos - 124.4) < 0.05,
          f"target {app.target:.4f}, read {pos:.4f}")
    app.do_jog(-10)
    settle(root, app)
    check("jog -10 from the target", abs(app.target - 114.4) < 1e-9)
    app.step_var.set("0.001")
    app.do_step(+1)
    settle(root, app)
    check("a step below one pulse is refused", abs(app.target - 114.4) < 1e-9)

    print("\nscope settings window")
    app.open_scope_settings()
    settle(root, app)
    check("window read the scope", app.set_vars[":TIMebase:SCALe"].get() != "",
          app.set_vars[":TIMebase:SCALe"].get())
    check("record span shown", "record:" in app.span_label.cget("text"),
          app.span_label.cget("text"))
    app.set_vars[":TIMebase:SCALe"].set("2.0E-03")
    app.do_apply_scope()
    settle(root, app)
    check("an edited field is written to the scope",
          app.link.scope.inst.state[":TIMebase:SCALe"] == "2.0E-03")
    check("span follows the setting", "-4.000 to +16.000 ms" in app.span_label.cget("text"),
          app.span_label.cget("text"))
    app.do_save_preset("test preset")
    check("preset saved and selectable", "test preset" in app.preset_box["values"]
          and app.cfg["user_presets"]["test preset"]["scope"][":TIMebase:SCALe"] == "2.0E-03")
    app.preset.set("Spin echo")
    app.preset_picked()
    app.seq["spacing_ms"].set("30")
    app.sequence_changed()
    sc = gui.cfgmod.all_presets(app.cfg)["Spin echo"]["scope"]
    t0_, t1_ = gui.cfgmod.record_span(sc[":TIMebase:SCALe"], sc[":TIMebase:POSition"], "LEFT")
    check("spin echo: the record follows the sequence fields (30 ms legs)",
          abs(t0_ + 12e-3) < 1e-9 and t1_ >= (30 + 9.5 + 12) * 1e-3,
          f"{t0_*1e3:.1f}..{t1_*1e3:.1f} ms; {app.seq_lbl.cget('text')}")
    app.seq["spacing_ms"].set("16.667")
    app.sequence_changed()
    check("picking a preset fills the shot settings",
          app.mode.get() == "single" and app.sv["rep_s"].get() == "10.0")
    app.mode.set("average")
    for k, v in {"shots": "64", "rep_s": "0.27"}.items():
        app.sv[k].set(v)
    app.link.scope.inst.state[":TIMebase:SCALe"] = "1.5E-03"
    app.set_win.destroy()

    print("\npreset applied with read-back, and the check before a scan")
    app.link.scope.put(":TIMebase:SCALe", "1.0E-03")
    app.link.scope.put(":CHANnel1:SCALe", "5.0")
    app.preset.set("AWG bench ramp")
    app.do_apply_preset()
    settle(root, app)
    log = app.logbox.get("1.0", "end")
    check("preset written and confirmed", "every one read back as written" in log)
    check("hand-changed V/div put back by role",
          float(app.link.scope.inst.state[":CHANnel1:SCALe"]) == 1.0)
    check("settings check logged", "Settings check:" in log)
    app.do_check_scope()
    settle(root, app)
    check("scope check with a test shot ran", "Scope check:" in app.logbox.get("1.0", "end")
          and app.last_check and any("ramp activity" in m for _, m in app.last_check),
          app.last_check and [m for lv, m in app.last_check if lv != "INFO"][:3])
    app.mode.set("average")
    for k, v in {"shots": "64", "rep_s": "0.27", "wait_s": "10"}.items():
        app.sv[k].set(v)

    print("\nscan")
    live_seen = []
    real_live_done = app._live_done

    def spy(res):
        if res is not None:
            live_seen.append(res["n_done"])
        real_live_done(res)
    app._live_done = spy
    app.do_start_scan()
    settle(root, app, timeout=300)
    check("plots updated while the scan ran", len(live_seen) >= 3
          and min(live_seen) < max(live_seen), live_seen)
    check("scan loaded after it ran", app.result is not None)
    if app.result is None:
        print(app.logbox.get("1.0", "end")[-3000:])
        return report()
    pol = app.result["pol"]
    check("12 angles fitted", pol["n_angles"] == 12, pol["n_angles"])
    check("the pre-run check is kept in the manifest",
          bool(app.result["d"].manifest.get("precheck")))
    prov = app.result["d"].manifest.get("provenance") or {}
    check("the manifest records the software versions",
          (prov.get("software", {}).get("ramp-polarimeter") or {}).get("commit"),
          gui.provenance.short(prov))
    from rampol import lablog
    rows = lablog.read(app.outdir.get())
    check("the scan has its row in the lab log",
          any(r["kind"] == "ramp scan" and r["name"] == "gui_test" for r in rows),
          [(r["kind"], r["name"]) for r in rows])
    check("direct ER computed with the load", len(app.result["direct"]) >= 10,
          len(app.result["direct"]))
    check("rest azimuth recovered", abs(pol["psi_rest"] - 23.7) < 0.1,
          f"{pol['psi_rest']:.3f}")
    check("dip ER points found", len(app.result["dips"]) >= 20, len(app.result["dips"]))
    out = os.path.join(SANDBOX, "figs")
    os.makedirs(out)
    for frame, (fig, _draw) in app.plot_tabs.items():
        app.nb.select(frame)
        root.update()
        if fig is not None:
            name = app.nb.tab(frame, "text")
            fig.savefig(os.path.join(out, f"{name}.png"))
    nfig = sum(1 for f, _d in app.plot_tabs.values() if f is not None)
    check("every figure tab drew", len(os.listdir(out)) == nfig, sorted(os.listdir(out)))
    caps = [sf._supxlabel.get_text() for sf in app.fig_diag.subfigs
            if getattr(sf, "_supxlabel", None) is not None]
    check("Diagnostics: a line under each of the 4 panels saying what it shows",
          len(caps) == 4 and all(len(c_) > 80 for c_ in caps), [c_[:40] for c_ in caps])
    check("table has rows", len(app.tv.get_children()) > 5, len(app.tv.get_children()))

    print("\ncursor and compare")
    app.cursor_var.set("5.2")
    app.set_cursor_text()
    root.update()
    check("cursor set", abs(app.cursor_t - 5.2e-3) < 1e-9)

    print("\nnull refine")
    app.rv["windows"].set("auto")
    app.rv["offsets"].set("-3,-1.5,0,1.5,3")
    app.rv["pd_vdiv"].set("0.02")
    app.do_run_refine()
    settle(root, app, timeout=300)
    ref = app.result["refine"]
    if len(ref) != 3:
        print(app.logbox.get("1.0", "end")[-2500:])
    check("three static windows refined", len(ref) == 3, [r.get("label") for r in ref])
    for r in ref:
        check(f"  {r['label']}: ER {'>' if r.get('er_lower') else ''}{r.get('er', 0):.0f}"
              f" (model light+analyzer 3333)", "er" in r and 1500 < r["er"] < 8000)
    app.nb.select(app.fig_ext._frame)
    root.update()
    app.fig_ext.savefig(os.path.join(out, "Extinction_refined.png"))
    # a refine that stopped part-way (7 Oct: a scope read error at step 3):
    # pressing Measure again finishes it instead of adding a second set
    rr = app._new_run(app.result["d"].name)
    rr.load()
    nulls = [s for s in rr.manifest["steps"] if s["kind"] == "null"]
    for s in nulls[-2:]:
        s["status"] = "todo"
    rr.save()
    n_before = len(rr.manifest["steps"])
    gui.messagebox.askyesnocancel = lambda *a_, **k_: True
    app.do_load_shown()
    settle(root, app, timeout=120)
    app.do_run_refine()
    settle(root, app, timeout=300)
    rr.load()
    check("a stopped refine: Measure finishes its steps, no second set added",
          len(rr.manifest["steps"]) == n_before
          and all(s["status"] == "done" for s in rr.manifest["steps"] if s["kind"] == "null"),
          (n_before, len(rr.manifest["steps"])))

    print("\nthe Light tab: the DS345 gating the light")
    app.do_connect_ds345()
    settle(root, app)
    check("DS345 Connect in the hardware pane (simulated)",
          app.ds_sess is not None and "simulated" in app.ds_hw.cget("text"), app.ds_hw.cget("text"))
    app.do_ds_from_scan("hold")
    root.update()
    lv = app.light_view
    check("'hold only' from the shown scan: light off, on inside the hold",
          lv is not None and app.ds_idle.get() == "off" and lv["gate"].windows,
          app.dv["windows"].get())
    app.redraw(app.fig_awg)
    check("the gate is drawn on the AWG plot tab",
          app.fig_awg.axes and app.fig_awg.axes[0].get_title().startswith("Light gate"),
          app.fig_awg.axes and app.fig_awg.axes[0].get_title()[:40])
    app.fig_awg.savefig(os.path.join(out, "Light_gate_preview.png"))
    app.do_ds_dry()
    settle(root, app, timeout=200)
    check("dry run of the gate on the simulated scope passed (Hi-Z, gain 1)",
          app.ds_dry is not None and app.ds_dry["ok"],
          app.ds_dry and (app.ds_dry["problems"], app.ds_dry["result"]["gain"]))
    app.rv["offsets"].set("-1.5,0,1.5")
    app.ds_gate_refine.set(True)
    rr.load()
    n0 = len(rr.manifest["steps"])
    app.do_run_refine()
    settle(root, app, timeout=400)
    rr.load()
    new = [s for s in rr.manifest["steps"][n0:] if s["kind"] == "null"]
    kinds = sorted({s.get("gate", {}).get("kind") for s in new})
    check("a gated refine: every null step carries its window's gate, hold and rest kinds",
          new and all(s["status"] == "done" and s.get("gate") for s in new)
          and kinds == ["hold", "rest"], (len(new), kinds))
    check("the DS345 parked afterwards (light on, ungated)",
          app.ds_sess.gate is None and app.ds_sess.parked_v == float(app.dv["on_v"].get()))
    sg_ = app.load_sg()
    hs = [s for s in new if s["gate"]["kind"] == "hold"][0]
    a_ = __import__("numpy").asarray(sg_.load_capture(os.path.join(rr.folder, hs["files"][0]))[1])
    pre = a_[a_[:, 0] < 0, 1]
    check("hold-null step: before the trigger the light is gated off (no bright rest)",
          pre.size and float(pre.mean()) < 0.01, pre.size and float(pre.mean()))
    app.ds_gate_refine.set(False)

    print("\nstopping part-way: what was measured is loaded and drawn")
    app.scan_name.set("stopped part way")
    app.bg_mode.set("none")
    app.dark_mode.set("none")
    app.check_first.set(False)
    app._sim_parts[0].realtime = 0.05        # slow enough to stop mid-scan
    app.nb.select(app.fig_traces._frame)      # draw the live traces as they come
    live_seen.clear()
    app.do_start_scan()
    t0 = time.time()
    root.update()
    while (app.busy or not app.calls.empty()) and time.time() - t0 < 120:
        root.update()
        time.sleep(0.01)
        if len(live_seen) >= 4 and not app.stop_flag.is_set():
            app.do_stop()
    settle(root, app)
    res = app.result
    log = app.logbox.get("1.0", "end")
    check("the stopped scan is loaded", res is not None and res["d"].name == "stopped_part_way"
          and 0 < res["n_done"] < res["n_total"],
          res and (res["d"].name, res["n_done"], res["n_total"]))
    check("the log says where it stopped and how to resume", "Stopped after" in log
          and "resume" in log)
    check("live traces drew from the first steps on", live_seen and min(live_seen) <= 2,
          live_seen)
    app.redraw(app.fig_traces)
    lines = len(app.fig_traces.axes[0].lines) if app.fig_traces.axes else 0
    check("Traces draws the shots even before there is a fit (fewer than 3 angles)",
          res is not None and (res["pol"] is not None or lines > 0),
          f"fit: {res and res['pol'] is not None}, lines drawn: {lines}")
    app.fig_traces.savefig(os.path.join(out, "Traces_stopped.png"))
    # the same, deterministically: a result with no fit yet
    app.result = dict(res, pol=None, dips=[], refine=[], mon=None)
    app.redraw(app.fig_traces)
    check("no fit yet: Traces still draws every shot",
          len(app.fig_traces.axes[0].lines) >= len(res["raw"][0]) > 0,
          len(app.fig_traces.axes[0].lines))
    app.redraw(app.fig_angle)
    check("no fit yet: the fit tabs say they need 3 angles",
          any("needs 3" in t.get_text() for ax in app.fig_angle.axes for t in ax.texts))
    app.result = res

    print("\nbias points: AWG plateaus into the simulated bench")
    tabs = [app.modes.tab(f, "text") for f in app.modes.tabs()]
    check("measurement modes are tabs (find and refine share Analyzer)",
          tabs == ["Ramp scan", "Analyzer", "AWG", "Fixed rotations", "ILC target",
                   "Light"], tabs)
    root.update_idletasks()
    col = app.modes.master
    check("the left column fits a 1080 px screen and keeps its width (Light tab + DS345 row "
          "added 8 Oct 2026: channels two a line)",
          col.winfo_reqheight() <= 962 and col.winfo_reqwidth() <= 520,
          f"{col.winfo_reqwidth()} x {col.winfo_reqheight()} px")
    app.bv["biases"].set("0:90:45")
    app.bv["shots"].set("4")
    app.bv["name"].set("gui-bias")
    app._sim_parts[0].realtime = 0.0
    app.bias_result = None
    app.do_start_bias()
    settle(root, app)
    check("a bias run without a dry run of its plateaus is refused", app.bias_result is None)
    app.do_bias_preview()
    root.update()
    ax_ = app.fig_awg.axes
    check("Fixed rotations Preview: every plateau on the AWG tab, the window shaded",
          len(ax_) >= 2 and ax_[0].get_title().startswith("Fixed rotations plan: 3 points, 0/0, 22.5/22.5, 45/45")
          and len(ax_[0].lines) == 6 and len(ax_[1].patches) == 1,
          (ax_[0].get_title() if ax_ else "", len(ax_[0].lines) if ax_ else 0))
    app.fig_awg.savefig(os.path.join(out, "AWG_fixed_rotations_preview.png"))
    pv = app.plan_view
    npts = int(app.bv["null_points"].get())
    # the null is predicted from the point before (4 angles for the first point
    # only) and the azimuth tracked at 2 slope angles per point (defaults)
    per_pt = npts + 1 + (2 if app.bias_track.get() else 0)
    n_az = 4 if app.bias_predict.get() else 12
    check("... and the plan step by step on the Plan tab: the dark, 4 azimuth angles for the "
          "first point, then per point the null points, the bright angle and the track pair",
          len(pv["steps"]) == 1 + n_az + 3 * per_pt
          and [st["kind"] for st in pv["steps"][1:6]] == ["azimuth"] * 4 + ["null"]
          and [st["kind"] for st in pv["steps"]].count("track") == (6 if app.bias_track.get() else 0)
          and [st["x1"] for st in pv["steps"] if st["kind"] == "bright"]
          == [0.0, 45.0 * float(app.bv["split"].get()), 90.0 * float(app.bv["split"].get())],
          (len(pv["steps"]), pv["title"]))
    app.redraw(app.fig_plan)
    check("the Plan tab draws it with the time estimate",
          "~" in app.fig_plan.axes[0].get_title() and "min" in app.fig_plan.axes[0].get_title(),
          app.fig_plan.axes[0].get_title())
    app.fig_plan.savefig(os.path.join(out, "Plan_fixed_rotations.png"))
    app.do_scan_plan()
    pv = app.plan_view
    n_ang = len([st for st in pv["steps"] if st["kind"] == "scan"])
    check("Ramp scan 'Preview plan': its angles in order, and that the AWG is not "
          "driving from here", n_ang > 0 and "not driving" in pv["note"], pv["note"][:60])
    app.find_light.set(gui.FIND_LIGHT[1])
    root.update()
    check("Analyzer: static light shows the mains field, not the record window",
          app.find_line_row.winfo_manager() == "pack" and not app.find_win_row.winfo_manager())
    app.find_light.set(gui.FIND_LIGHT[0])
    root.update()
    for opener, attr, want in ((app.open_scan_settings, "scan_set_win",
                                ["blocks", "dither_codes", "points", "wait_s", "rep_s",
                                 "backoff_deg", "ref_angle"]),
                               (app.open_find_settings, "find_set_win", [])):
        opener()
        root.update()
        w_ = getattr(app, attr)
        stack, kids = [w_], []
        while stack:
            x_ = stack.pop()
            kids.append(x_)
            stack += x_.winfo_children()
        bound = {str(k_.cget("textvariable")) for k_ in kids
                 if k_.winfo_class() in ("TEntry", "TCombobox")}
        check(f"{attr}: opens, every set-once field in it",
              {str(app.sv[k_]) for k_ in want} <= bound, sorted(want))
        w_.destroy()
    app.do_bias_dry()
    settle(root, app, timeout=300)
    dry = getattr(app, "awg_dry_all", [])
    check("dry run of the plan: every plateau played into the scope and passed",
          len(dry) == 3 and all(r["ok"] for r in dry), [r["problems"][:1] for r in dry])
    check("the dry run is on record (file and lab log)",
          os.path.isdir(os.path.join(app.outdir.get(), "awg_dryrun"))
          and any(r["kind"] == "AWG dry run" for r in lablog.read(app.outdir.get())))
    check("after the dry run the AWG is parked (never-float rule)",
          all(app.bench.awg_on.values()) and app.awg_sess.parked)
    app.do_start_bias()
    settle(root, app, timeout=300)
    br = app.bias_result
    pts = (br or {}).get("points", [])
    check("bias run measured and loaded", len(pts) == 3,
          ", ".join(f"{p['bias']:g}: ER {p.get('er') or 0:.0f}" for p in pts))
    check("ER at rest near the bench's 3333", bool(pts) and bool(pts[0].get("er"))
          and abs(pts[0]["er"] / 3333 - 1) < 0.2, pts and pts[0].get("er"))
    check("AWG parked afterwards: idle waveform, outputs ON (nothing floats)",
          all(app.bench.awg_on.values()) and app.awg_sess.parked
          and __import__("numpy").ptp(app.bench.awg_drive[1][1]) == 0)
    check("the beam is back (dark unblocked)", app.bench.imax > 0)
    # the AWG now plays idle; the experiment's ramps come back when its own
    # drive is put back (on the bench: the ILC panel uploads it)
    app.awg_sess.off(force=True)
    app.awg_sess.forget()
    app.nb.select(app.fig_bias._frame)
    root.update()
    app.fig_bias.savefig(os.path.join(out, "Bias_points.png"))
    check("bias tab drew four panels", len(app.fig_bias.axes) >= 4, len(app.fig_bias.axes))
    check("the bias run is in the lab log and its manifest has provenance",
          any(r["kind"] == "bias points" for r in lablog.read(app.outdir.get()))
          and bool((br or {}).get("provenance")))
    app.nb.select(app.fig_ilc._frame)
    root.update()
    check("ILC tab says what to do with no comparison yet",
          any("Compare shown scan" in t.get_text() for ax in app.fig_ilc.axes for t in ax.texts))

    print("\nnames count up; dark and background: measured, reused, shown up front")
    import numpy as np
    app.scan_name.set("gui test")            # taken (and finished) above
    app.bg_mode.set("reuse latest")
    app.dark_mode.set("measure")
    for k, v in {"start": "0", "stop": "150", "step": "30", "ref_every": "0",
                 "shots": "8", "blocks": "4"}.items():
        app.sv[k].set(v)
    app.bench.ambient = 0.004                # 4 mV of stray light on the PD
    app.check_first.set(False)
    app.do_start_scan()
    settle(root, app, timeout=300)
    check("a used name counts up, spaces as underscores",
          app.scan_name.get() == "gui_test_2", app.scan_name.get())
    res = app.result
    man = res["d"].manifest if res else {}
    check("background reused from the earlier scan",
          man.get("borrowed", {}).get("background", {}).get("source") == "gui_test",
          man.get("borrowed"))
    lv = an.offset_levels(res["d"], 1.0) if res else {}
    check("dark (PD covered) measured in this scan",
          lv.get("dark", {}).get("source") == "this scan",
          lv.get("dark", {}).get("level"))
    txt = app.corr_label.cget("text")
    check("corrections shown up front", "subtract background" in txt and "gui_test" in txt,
          txt[:150])
    app.sub_dark.set(False)
    app.reanalyse()
    settle(root, app)
    check("subtraction can be switched off and says so",
          "NOTHING subtracted" in app.corr_label.cget("text"))
    app.sub_dark.set(True)
    app.reanalyse()
    settle(root, app)

    print("\nstray light read at a fine V/div")
    app.scan_name.set("stray test")
    app.dark_mode.set("measure")
    app.bg_mode.set("measure")
    app.stray_on.set(True)
    app.stray_vdiv.set("5")
    for k, v in {"start": "0", "stop": "90", "step": "30", "ref_every": "0",
                 "shots": "8", "blocks": "4"}.items():
        app.sv[k].set(v)
    scope_ = app._sim_parts[0]
    scope_.offset_err = -0.013               # the bench's: -35 mV at a 2.7 V offset
    app.bench.ambient = 0.004
    app.do_start_scan()
    settle(root, app, timeout=300)
    scope_.offset_err = 0.0
    res = app.result
    st = an.stray_light(res["d"])
    check("stray light: dark and background also read at 5 mV/div, their difference "
          "is the 4 mV on the bench", st is not None and abs(st["vdiv"] - 0.005) < 1e-9
          and abs(st["level"] - 0.004) < 0.3e-3 and st["sem"] < 0.2e-3,
          st and f"{st['level']*1e3:.3f} +- {st['sem']*1e3:.3f} mV")
    sub, info = an.dark_level(res["d"], an._pd_vdiv(res["d"], "scan"))
    off_ = [s_ for s_ in res["d"].steps if s_["kind"] == "dark" and "pd_scale" not in s_][0]
    o_set = an._scale_of(res["d"], off_)[1]
    want = app.bench.dark + 0.004 - 0.013 * o_set
    check("subtracted: the offset at the scan's V/div plus the fine stray light",
          info is not None and info["kind"] == "dark + stray light" and abs(sub - want) < 3e-3,
          info and f"{sub*1e3:.2f} mV vs {want*1e3:.2f}; {app.corr_label.cget('text')[:170]}")
    names_ = sorted(os.listdir(res["d"].folder))
    check("the fine dark / background have files of their own",
          any(n_.startswith("stray_test_dark_5mVdiv_") for n_ in names_)
          and any(n_.startswith("stray_test_dark_0") for n_ in names_), names_[:6])
    app.bench.ambient = 0.0
    app.dark_mode.set("measure")
    app.bg_mode.set("reuse latest")
    # the rest of the test works on gui_test_2
    app.refresh_scan_list(select="gui_test_2")
    app.do_load_shown()
    settle(root, app)

    print("\nextinction with the direct points, residual map, Poincare")
    app.nb.select(app.fig_ext._frame)
    root.update()
    def key():
        leg = app.fig_ext.axes[0].get_legend()
        return [t.get_text() for t in leg.get_texts()] if leg else []
    labels = key()
    check("Extinction: the measured points, dip fits and both legs in the key",
          any(x.startswith("Measured at a crossing") for x in labels)
          and any(x.startswith("Dip fit") for x in labels)
          and any(x.startswith("leg") for x in labels), labels)
    check("error bars drawn", len(app.fig_ext.axes[0].collections) > 5,
          len(app.fig_ext.axes[0].collections))
    app.ext_show["direct"].set(False)
    app.redraw(app.fig_ext)
    check("and the measured points hidden when unticked",
          not any(x.startswith("Measured") for x in key()))
    app.ext_show["direct"].set(True)
    app.show_er_help()
    root.update()
    check("'What are these?' explains every family",
          "DIP FIT" in gui.ER_HELP and app.er_help_win.winfo_exists())
    app.er_help_win.destroy()
    app.nb.select(app.fig_map._frame)
    app.map_show.set(gui.MAP_MODES[2])
    app.redraw(app.fig_map)
    root.update()
    check("Map: residual / standard error with a per-angle rms panel",
          len(app.fig_map.axes) >= 3 and "residual" in app.fig_map.axes[0].get_title().lower(),
          [a.get_title()[:30] for a in app.fig_map.axes])
    app.fig_map.savefig(os.path.join(out, "Map_residual.png"))
    app.map_show.set(gui.MAP_MODES[0])
    app.nb.select(app.fig_poin._frame)
    root.update()
    check("Poincare: a 3-d sphere, the ellipticity and the ellipse",
          any(getattr(a, "name", "") == "3d" for a in app.fig_poin.axes)
          and len(app.fig_poin.axes) >= 3)
    app.fig_poin.savefig(os.path.join(out, "Poincare.png"))

    print("\ncursor moves in place; right-click copies a plot position")
    app.nb.select(app.fig_malus._frame)
    root.update()
    ax0 = app.fig_malus.axes[0]
    app.cursor_var.set("3.0")
    app.set_cursor_text()
    root.update()
    check("Malus follows the cursor without a rebuild",
          app.fig_malus.axes[0] is ax0 and "t = 3.0" in ax0.get_title(), ax0.get_title())

    class Ev:
        button, inaxes, xdata, ydata = 3, ax0, 12.5, 3.25
    app.copy_coords(Ev())
    check("right-click copies x and y", root.clipboard_get() == "12.5\t3.25",
          repr(root.clipboard_get()))

    print("\ncompare scans")
    i = list(app.cmp_lb.get(0, "end")).index("gui_test")
    app.cmp_lb.selection_clear(0, "end")
    app.cmp_lb.selection_set(i)
    app.do_compare_load()
    settle(root, app)
    root.update()
    ax = app.fig_cmp.axes
    names = [ln.get_label() for ln in ax[0].lines] if ax else []
    check("Compare: rotation of both scans, their difference and the ER",
          len(ax) == 3 and "gui_test" in names and "gui_test_2 (shown)" in names, names)
    app.fig_cmp.savefig(os.path.join(out, "Compare.png"))
    app.fig_cmp._ylog.set(False)
    app.redraw(app.fig_cmp)
    check("Compare: the ER panel goes linear", app.fig_cmp.axes[-1].get_yscale() == "linear")
    app.fig_cmp._ylog.set(True)

    print("\ncompared scans on the other tabs; log / linear; the ER table")
    app.nb.select(app.fig_ext._frame)
    app.fig_ext._cmp_on.set(True)
    app.redraw(app.fig_ext)
    leg = app.fig_ext.axes[0].get_legend()
    texts = [t.get_text() for t in leg.get_texts()] if leg else []
    check("Extinction: the compared scan drawn and named in the key",
          any(t == "compared: gui_test" for t in texts), texts[-6:])
    check("Extinction: log by default", app.fig_ext.axes[0].get_yscale() == "log")
    app.fig_ext._ylog.set(False)
    app.redraw(app.fig_ext)
    check("... and linear from 0 when unticked", app.fig_ext.axes[0].get_yscale() == "linear"
          and app.fig_ext.axes[0].get_ylim()[0] == 0)
    app.fig_ext._ylog.set(True)
    app.fig_ext._cmp_on.set(False)
    app.ext_show["lower"].set(False)
    app.redraw(app.fig_ext)
    leg = app.fig_ext.axes[0].get_legend()
    texts = [t.get_text() for t in leg.get_texts()] if leg else []
    check("'lower bounds' unticked: no lower bound drawn",
          not any("lower bound" in t for t in texts), texts)
    app.ext_show["lower"].set(True)
    for f_, what in ((app.fig_malus, "Malus"), (app.fig_angle, "Angle"),
                     (app.fig_poin, "Poincaré"), (app.fig_diag, "Diagnostics")):
        f_._cmp_on.set(True)
        app.redraw(f_)
        labels = [ln.get_label() for a_ in f_.axes for ln in a_.lines]
        check(f"{what}: the compared scan drawn", "gui_test" in labels, labels[:8])
        f_._cmp_on.set(False)
    for f_, what in ((app.fig_traces, "Traces"), (app.fig_malus, "Malus")):
        f_._ylog.set(True)
        app.redraw(f_)
        check(f"{what}: log y is symlog (the dark-subtracted level goes below zero)",
              f_.axes[0].get_yscale() == "symlog")
        f_._ylog.set(False)
        app.redraw(f_)
    app.do_copy_er()
    txt = root.clipboard_get()
    import csv as _csv
    import io as _io
    body = [ln for ln in txt.splitlines() if not ln.startswith("#")]
    rows = list(_csv.DictReader(_io.StringIO("\n".join(body))))
    meth = {r_["method"] for r_ in rows}
    check("Copy CSV: a '#' header, then crossings, dips and the binned ER_fit",
          txt.startswith("# extinction ratio along the ramp - scan gui_test_2")
          and {"crossing", "dip", "er_fit"} <= meth
          and all(r_["er"] for r_ in rows if r_["method"] == "crossing"), sorted(meth))
    cr = [r_ for r_ in rows if r_["method"] == "crossing"]
    check("each crossing row: leg, direction, rotation, sigma or a lower-bound flag",
          all(r_["leg"] in ("1", "2") and r_["direction"] in ("away", "back")
              and r_["rotation_deg"] and (r_["lower_bound"] == "1" or r_["er_sigma_lo"])
              for r_ in cr), cr[0] if cr else None)
    check("direction from the segment: ramp out = away, ramp back = back",
          all((r_["segment"].startswith("up") and r_["direction"] == "away")
              or (r_["segment"].startswith("down") and r_["direction"] == "back")
              for r_ in cr if r_["segment"][:2] in ("up", "do")),
          [(r_["segment"], r_["direction"]) for r_ in cr][:6])
    corr_line = [ln for ln in txt.splitlines() if ln.startswith("# corrections applied:")]
    check("the corrections line gives the stray light with its error",
          bool(corr_line) and ("stray light" not in corr_line[0] or "+-" in corr_line[0]),
          corr_line)
    target = os.path.join(SANDBOX, "er_export.csv")
    gui.filedialog.asksaveasfilename = lambda **k: target
    app.do_export_er()
    check("Export CSV writes the same table", os.path.isfile(target)
          and open(target, encoding="utf-8").read() == app._er_csv())

    print("\nbrief export")
    app.do_export_brief()
    bdir = os.path.join(app.result["d"].folder, "analysis", "brief")
    made = sorted(os.listdir(bdir)) if os.path.isdir(bdir) else []
    check("brief: 8 figures, summary.json and summary.md",
          sum(m.endswith(".png") for m in made) == 8 and "summary.json" in made
          and "summary.md" in made, made)
    import json as _json
    summ = _json.load(open(os.path.join(bdir, "summary.json"), encoding="utf-8"))
    check("summary has the direct ER and the provenance",
          summ.get("direct_er_min") and summ.get("provenance"), summ.get("direct_er_min"))

    print("\nrename / edit a scan")
    from rampol import lablog
    outd = app.outdir.get()
    old_dir = app.result["d"].folder
    app.open_edit_scan()
    root.update()
    check("Rename / edit opens on the shown scan", "gui_test_2" in app.edit_win.title())
    app.edit_win.destroy()
    new = app.edit_scan(old_dir, {"plan.sequence": {"spacing_ms": 133.333, "motion_ms": 9.5,
                                                    "before_ms": 60.0, "after_ms": 125.0},
                                  "notes": "the offset was not applied"},
                        "gui test 2 renamed")
    nd = os.path.join(outd, new)
    files = os.listdir(nd)
    import json as _json2
    man = _json2.load(open(os.path.join(nd, f"{new}_scan.json"), encoding="utf-8"))
    check("renamed with underscores: folder, every capture, the manifest",
          new == "gui_test_2_renamed" and not os.path.exists(old_dir)
          and all(f.startswith(new + "_") for f in files if os.path.isfile(os.path.join(nd, f)))
          and man["name"] == new
          and all(f.startswith(new + "_") for st in man["steps"] for f in st.get("files", [])),
          files[:3])
    briefs = os.listdir(os.path.join(nd, "analysis", "brief"))
    check("exported figures carry the new name",
          any(b.startswith(new + "_") for b in briefs)
          and not any("gui_test_2_" in b and not b.startswith(new) for b in briefs), briefs[:3])
    check("corrections kept with the old values",
          [e["field"] for e in man.get("edits", [])] == ["plan.sequence", "notes", "name"]
          and man["edits"][-1]["from"] == "gui_test_2"
          and man["plan"]["sequence"]["spacing_ms"] == 133.333, man.get("edits"))
    rows = [r["name"] for r in lablog.read(outd) if r["kind"] == "ramp scan"]
    check("the lab log row follows the rename", new in rows and "gui_test_2" not in rows, rows)
    app.refresh_scan_list(select=new)
    app.do_load_shown()
    settle(root, app)
    check("the renamed scan loads and reports its notes",
          app.result["d"].name == new
          and an.scan_summary(app.result)["notes"] == "the offset was not applied")
    # a rename that cannot happen changes nothing
    lock = open(os.path.join(nd, sorted(f for f in files if f.endswith(".npz"))[-1]), "rb")
    try:
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            app.edit_scan(nd, {}, "never")
            undone = False
        except OSError:
            undone = os.path.isdir(nd) and not os.path.exists(os.path.join(outd, "never"))
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        lock.close()
    check("a held file: the rename is refused and everything put back", undone,
          sorted(os.listdir(nd))[:2])
    # the first scan lent its background to gui_test_2: renaming the lender
    # updates the citation
    app.edit_scan(os.path.join(outd, "gui_test"), {}, "gui test lender")
    man = _json2.load(open(os.path.join(nd, f"{new}_scan.json"), encoding="utf-8"))
    check("a scan citing the renamed one follows",
          man["borrowed"]["background"]["source"] == "gui_test_lender", man["borrowed"])
    app.edit_scan(os.path.join(outd, "gui_test_lender"), {}, "gui_test")
    app.refresh_scan_list(select=new)
    app.do_load_shown()
    settle(root, app)

    print("\nshots: single traces and averages")
    app.nb.select(app.fig_shots._frame)
    root.update()
    d = app.result["d"]
    i45 = next(i for i, st in enumerate(d.steps) if st["kind"] == "scan")
    app.shots_lb.selection_clear(0, "end")
    app.shots_lb.selection_set(i45)
    app.sv_shots["partner"].set(True)
    app.redraw(app.fig_shots)
    lines = app.fig_shots.axes[0].lines if app.fig_shots.axes else []
    labels = [ln.get_label() for ln in lines]
    check("a step and its 90-deg partner, shots and averages",
          sum("average" in x for x in labels) == 2 and sum("shots" in x for x in labels) == 2,
          labels[:6])
    app.shots_t0.set("1.0")
    app.shots_t1.set("1.5")
    app.redraw(app.fig_shots)
    xl = app.fig_shots.axes[0].get_xlim()
    check("a time window, read at full resolution", abs(xl[0] - 1.0) < 1e-6 and abs(xl[1] - 1.5) < 1e-6)
    app.fig_shots.savefig(os.path.join(out, "Shots_window.png"))
    app.cursor_t = 2.0e-3
    app.shots_crossed()
    root.update()
    sel = app.shots_lb.curselection()
    check("'crossed at cursor' picks one step and a window", len(sel) == 1
          and app.shots_t0.get() != "")

    print("\nbuild: the angles adding up")
    app.nb.select(app.fig_build._frame)
    root.update()
    app.build_t.set(3.0)
    app._build_move()
    b = app._build
    xs, ys = b["corr"].get_data()
    check("the slider moves the Malus points and the fit", len(xs) == len(app.result["pol"]["theta"])
          and "t = 3.0" in b["txt"].get_text(), b["txt"].get_text().splitlines()[0])
    app.fig_build.savefig(os.path.join(out, "Build.png"))
    app.nb.select(app.fig_corr._frame)
    root.update()
    app.fig_corr.savefig(os.path.join(out, "Corrections.png"))
    check("corrections tab drew four panels", len(app.fig_corr.axes) >= 4)

    print("\nfind the min / max transmission angle")
    t0 = app.result["d"].t[0] * 1e3
    app.find_light.set(gui.FIND_LIGHT[0])
    app.fv["shots"].set("4")
    app.fv["window"].set("-50:-40")
    app.find_zoom.set(False)
    app.do_find_angle()
    settle(root, app)
    check("without the zoom, a window outside the record is refused before anything moves",
          "is not inside the record" in app.logbox.get("1.0", "end"))
    app.find_zoom.set(True)
    texts = []
    real_progress = app._progress
    app._progress = lambda d, n, t: (texts.append(t), real_progress(d, n, t))
    app.link.scope.put(":TIMebase:SCALe", "5.0E-03")   # hand-changed since the preset
    app.fv["window"].set(f"{t0 + 0.2:.2f}:-0.1")
    for kind, truth in (("min", (23.7 + 90) % 180), ("max", 23.7)):
        app.find_kind.set(kind)
        app.do_find_angle()
        settle(root, app, timeout=300)
        fr = getattr(app, "find_result", None) or {}
        err = abs((fr.get("angle", 999) - truth + 90) % 180 - 90)
        check(f"{kind} found at rest", err < 0.1, f"{fr.get('angle')} vs {truth}")
    app.fig_find.savefig(os.path.join(out, "Find_angle.png"))
    check("the timebase was zoomed onto the window while measuring",
          "around the window" in app.logbox.get("1.0", "end"))
    check("progress shows the step and the time left",
          any(t.startswith("Find min: analyzer") and "left" in t for t in texts),
          [t for t in texts if "left" in t][:2])
    pre = gui.cfgmod.all_presets(app.cfg)[app.preset.get()]["scope"]
    check("Find set the scope from the preset first, as a ramp scan",
          float(app.link.scope.get(":TIMebase:SCALe")) == float(pre[":TIMebase:SCALe"])
          and "scope set from the preset" in app.logbox.get("1.0", "end"),
          app.link.scope.get(":TIMebase:SCALe"))

    print("\nstatic light: line trigger, the PD mean, no ramp")
    sc = app.link.scope
    trig0 = sc.get(":TRIGger:EDGE:SOURce")
    app.find_light.set(gui.FIND_LIGHT[1])
    app.fv["step"].set("15")
    app.do_malus_scan()
    settle(root, app, timeout=300)
    fr = app.find_result
    err = abs((fr["angle_max"] - 23.7 + 90) % 180 - 90)
    check("Malus scan of the static light: maximum at the rest azimuth", fr["kind"] == "scan"
          and err < 0.5 and fr.get("static"), f"{fr['angle_max']:.2f}")
    check("the Malus scan shows its progress with the time left",
          any(t.startswith("Malus scan: analyzer") and "left" in t for t in texts))
    check("Malus scan autoranged: points near crossed read at a finer V/div",
          min(fr["vdivs"]) < max(fr["vdivs"]) / 10, sorted(set(fr["vdivs"])))
    check("the background was measured after, at every V/div used, and subtracted",
          fr.get("offsets") and not fr.get("missing")
          and all(v["kind"] == "background" for v in fr["offsets"].values()),
          fr.get("offset_note"))
    check("so the fitted ER is near the bench's 3333 (raw would be nonsense)",
          not fr["er_lower"] and 2000 < fr["er"] < 5000, f"{fr['er']:.0f}")
    app.fig_find.savefig(os.path.join(out, "Malus_scan.png"))
    app.find_kind.set("min")
    app.do_find_angle()
    settle(root, app, timeout=300)
    fr = app.find_result
    err = abs((fr["angle"] - (23.7 + 90)) % 180)
    err = min(err, 180 - err)
    check("static light: crossed found", err < 0.1 and fr.get("static"), f"{fr['angle']:.3f}")
    check("the scope's trigger is put back after", sc.get(":TRIGger:EDGE:SOURce") == trig0,
          sc.get(":TRIGger:EDGE:SOURce"))
    app.find_light.set(gui.FIND_LIGHT[0])

    print("\nEOM calibration")
    app.open_calibration()
    root.update()
    app.cal_vars[("EO1", "gain")].set("0.6")
    app.conv_ch.set("EO1")
    app.conv["deg"].set("45")
    app._cal_convert("deg")
    check("the converter: 45 deg on EO1 -> AWG volts with the typed gain",
          abs(float(app.conv["awg"].get()) - 45 / 90 * 5.1283 / 0.6) < 1e-3,
          app.conv["awg"].get())
    app._cal_apply()
    from rampol import bias as biasmod
    check("applied: waveforms and analysis use it", abs(biasmod.CHAN["EO1"]["gain"] - 0.6) < 1e-12
          and abs(app.cfg["calibration"]["EO1"]["gain"] - 0.6) < 1e-12)
    app._cal_fill(gui.calib.DEFAULT)
    app._cal_apply()
    check("and back to the 1 Sep values", abs(biasmod.CHAN["EO1"]["gain"] - 0.5594) < 1e-12)
    app.cal_win.destroy()

    print("\nAWG mode: dry run, then ramp to a rotation and find the null in the hold")
    app.modes.select(2)
    app.a_choice["source"].set("ILC drives")
    root.update()
    check("AWG tab: choosing ILC drive files shows their fields, not the ramp's",
          app.a_files.winfo_manager() == "pack" and not app.a_ramp.winfo_manager())
    app.a_choice["source"].set("ramp")
    root.update()
    check("... and back", app.a_ramp.winfo_manager() == "pack" and not app.a_files.winfo_manager())
    app.open_awg_settings()
    root.update()
    win_ = app.awg_set_win
    kids = []
    stack = [win_]
    while stack:
        w_ = stack.pop()
        kids.append(w_)
        stack += w_.winfo_children()
    bound = {str(k_.cget("textvariable")) for k_ in kids
             if k_.winfo_class() in ("TEntry", "TCombobox")}
    want = {str(app.av[k_]) for k_ in ("dt_us", "idle1", "idle2", "trig_hz", "scope_before_ms",
                                       "scope_after_ms", "dry_shots", "seq_settle_s")}
    want |= {str(app.a_choice[k_]) for k_ in ("dry_ch1", "dry_ch2")}
    check("Settings...: every set-once AWG field is in the dialog", want <= bound,
          sorted(want - bound))
    app.a_require.set(False)
    gui.messagebox.askyesno = lambda *a_, **k_: True
    app._awg_flags()
    check("the safety line turns red with a rule off",
          "NO dry run" in app.awg_safety.cget("text")
          and str(app.awg_safety.cget("foreground")) == "#c00000", app.awg_safety.cget("text"))
    app.a_require.set(True)
    app._awg_flags()
    win_.destroy()
    # what the AWG can reach (+-10 V; 9.6 V cap here): shown as it is typed
    app.av["rotation"].set("180")
    app.av["split"].set("1")
    root.update()
    t_ = app.a_reach.cget("text")
    check("180 deg all on X1: flagged red, with the X1 shares that would work",
          t_.startswith("PAST THE AWG'S REACH") and "X1 share of 0.4" in t_
          and str(app.a_reach.cget("foreground")) == "#c00000", t_)
    app.av["split"].set("0.5")
    root.update()
    check("... and 90 / 90 is fine", not app.a_reach.cget("text").startswith("PAST"),
          app.a_reach.cget("text"))
    app.av["seq_x1"].set("0:120:60")
    app.av["seq_x2"].set("0:120:60")
    app.a_choice["seq_how"].set("grid")
    root.update()
    e_, w_, f_ = app._seq_waves(app.gather())
    check("a sequence grid leaves out the corners past the AWG's reach",
          len(e_) == 4 and all(max(x, y) <= 60 for x, y in e_)
          and any("left out" in m for _lv, m in f_), (e_, app.awg_seq_lbl.cget("text")))
    app.a_choice["seq_how"].set("pairs")
    app.av["seq_x1"].set("120")
    app.av["seq_x2"].set("0")
    root.update()
    check("... a pairs list says which to change",
          "change them to start" in app.awg_seq_lbl.cget("text"), app.awg_seq_lbl.cget("text"))
    app.bv["split"].set("1.0")
    app.bv["biases"].set("0:120:60")
    root.update()
    check("Fixed rotations: the rotations past what the split reaches, in red",
          app.bias_reach.cget("text").startswith("PAST THE AWG'S REACH: 120"),
          app.bias_reach.cget("text"))
    app.bv["split"].set("0.5")
    app.a_choice["source"].set("ramp")
    app.av["rotation"].set("60")
    app.do_awg_preview()
    root.update()
    check("preview drew the waveform", len(app.fig_awg.axes) >= 2
          and "ramp to 60" in app.fig_awg.axes[0].get_title(), app.fig_awg.axes[0].get_title())
    app.do_awg_park()                # live outputs at idle: nothing floats
    t1 = time.time()
    while not (app.awg_sess and app.awg_sess.parked) and time.time() - t1 < 10:
        root.update()
        time.sleep(0.02)
    settle(root, app)
    app.do_awg_load()
    settle(root, app)
    check("an unverified waveform is refused onto the live (parked) outputs",
          "has not passed a dry run" in app.logbox.get("1.0", "end")
          and app.awg_sess.parked)
    app.do_awg_dry()
    settle(root, app, timeout=300)
    check("dry run of the ramp passed", app.awg_dry["ok"], app.awg_dry["problems"][:2])
    app.nb.select(app.fig_awg._frame)
    root.update()
    check("the AWG tab shows what the scope saw", "Dry run PASSED" in app.fig_awg.axes[0].get_title(),
          app.fig_awg.axes[0].get_title())
    app.fig_awg.savefig(os.path.join(out, "AWG_dry_run.png"))
    app.do_awg_load()
    settle(root, app)
    app.do_awg_on()
    settle(root, app)
    check("outputs on with the ramp (asked first)", all(app.bench.awg_on.values())
          and not app.awg_sess.parked and "ON" in app.awg_lbl.cget("text"),
          app.awg_lbl.cget("text"))
    app.av["shots"].set("4")
    app.do_awg_find("min")
    settle(root, app, timeout=300)
    fr = app.find_result
    err = abs((fr["angle"] - (23.7 + 90 + 60) % 180 + 90) % 180 - 90)
    check("null in the hold of a 60 deg ramp", err < 0.1 and all(app.bench.awg_on.values()),
          f"{fr['angle']:.3f}")

    # a ramp scan while this window's AWG plays: the scan records what drives
    app.scan_name.set("awg driven")
    app.dark_mode.set("none")
    app.bg_mode.set("none")
    app.check_first.set(False)
    for k, v in {"start": "0", "stop": "150", "step": "30", "ref_every": "0",
                 "shots": "4", "blocks": "4"}.items():
        app.sv[k].set(v)
    app.do_start_scan()
    settle(root, app, timeout=300)
    man = app.result["d"].manifest if app.result else {}
    check("a ramp scan with the AWG playing: the manifest says this window's AWG drives",
          (man.get("drive") or {}).get("label", "").startswith("ramp to 60")
          and man["drive"]["dry_run_passed"] and all(app.bench.awg_on.values())
          and app.result["pol"] is not None
          and 55 < float(np.max(np.abs(app.result["pol"]["rotation"]))) < 65,
          (man.get("drive", {}).get("label"),
           app.result and float(np.max(np.abs(app.result["pol"]["rotation"])))))
    check("the AWG-hold Find put the PD's V/div back (the scan was not clipped)",
          man["steps"][0]["scales"][str(app.result["d"].roles["PD"])][0] == 1.0,
          man["steps"][0]["scales"])
    row = [r for r in lablog.read(app.outdir.get()) if r["name"] == "awg_driven"]
    check("... and so does its lab-log row", row and row[0]["ilc"].startswith("AWG (this window)"),
          row and row[0]["ilc"])

    # saving a figure: a name and the scan's saved_figures folder filled in
    asked = {}

    def fake_save(**k):
        asked.update(k)
        return os.path.join(k["initialdir"], k["initialfile"])
    gui.filedialog.asksaveasfilename = fake_save
    app.nb.select(app.fig_ext._frame)
    root.update()
    app.fig_ext._toolbar._buttons["Save"].invoke()
    root.update()
    check("Save (toolbar): the scan's saved_figures folder and a filled-in name",
          asked.get("initialdir") == os.path.join(app.result["d"].folder, "saved_figures")
          and asked.get("initialfile", "").startswith("awg_driven_Extinction_vs_")
          and os.path.isfile(os.path.join(asked["initialdir"], asked["initialfile"])),
          f"{asked.get('initialdir')} | {asked.get('initialfile')}")

    print("\nAWG sequence: X1 / X2 end points, one ramp scan each, with the analyzer")
    app.av["seq_x1"].set("30")
    app.av["seq_x2"].set("0, 30")
    app.a_choice["seq_how"].set("pairs")
    app.a_choice["seq_order"].set(gui.SEQ_ORDERS[0])
    root.update()
    check("the sequence is listed under its fields", app.awg_seq_lbl.cget("text").startswith("2 ramps"),
          app.awg_seq_lbl.cget("text"))
    app.scan_name.set("seq test")
    app.do_seq_start()               # not dry-run yet: asks, then dry-runs all (stubbed yes)
    settle(root, app, timeout=300)
    s_ = app.awg_sess
    _c = app.gather()
    _e, waves_, _f = app._seq_waves(_c)
    check("Start asks for the dry run first; both ramps then pass it",
          all(s_.is_verified(w) for w in waves_), [w.label for w in waves_])
    app.do_seq_start()
    settle(root, app, timeout=600)
    import json as _js
    mans = {}
    for n_ in ("seq_test_X1_30_X2_0", "seq_test_X1_30_X2_30"):
        mp_ = os.path.join(app.outdir.get(), n_, f"{n_}_scan.json")
        mans[n_] = _js.load(open(mp_, encoding="utf-8")) if os.path.isfile(mp_) else {}
    ok_ = all(m and all(x["status"] == "done" for x in m["steps"]) for m in mans.values())
    check("two ramp scans, every step measured, each recording its own end points",
          ok_ and [m["drive"]["ends_deg"] for m in mans.values()] == [{"X1": 30.0, "X2": 0.0},
                                                                     {"X1": 30.0, "X2": 30.0}],
          {k: (m.get("drive") or {}).get("ends_deg") for k, m in mans.items()})
    second = mans["seq_test_X1_30_X2_30"]
    check("interleaved: the analyzer stayed put for the second ramp at each grid angle",
          all(x.get("stayed") for x in second["steps"]
              if x["kind"] == "scan" and not x.get("hold_null")))
    nulls = {n_: [x for x in m["steps"] if x.get("hold_null")] for n_, m in mans.items()}
    check("each member measured its own hold-null angles after the grid (3 across the "
          "null + the bright angle, those not on the grid)",
          all(0 < len(v) <= 4 and all(x["status"] == "done" for x in v) for v in nulls.values())
          and all(m["plan"]["hold_angles"] == [x["target"] for x in nulls[n_]]
                  for n_, m in mans.items())
          and nulls["seq_test_X1_30_X2_30"] != nulls["seq_test_X1_30_X2_0"],
          {k: [x["target"] for x in v] for k, v in nulls.items()})
    check("the AWG is parked at the end, on the ILC's 11 ms record",
          s_.parked and abs(s_.wave.period - 11.002e-3) < 1e-9 and all(app.bench.awg_on.values()))
    rots = {r["d"].name: float(np.max(np.abs(r["pol"]["rotation"]))) for r, _c2 in app._compared()}
    rots.update({app.result["d"].name: float(np.max(np.abs(app.result["pol"]["rotation"])))}
                if app.result else {})
    check("both in the Compare tab afterwards, at their own rotations (30 and 60 deg)",
          len(app.cmp_sel) == 2 and sorted(round(abs(v)) for k, v in rots.items()
                                           if k.startswith("seq_test")) == [30, 60], rots)
    # dark and background reused from an earlier scan: the first scan has
    # nothing measured of its own to lend, only what it borrowed (7 Oct: this
    # raised 'no completed steps with data in this scan yet' before any step)
    app.dark_mode.set("reuse latest")
    app.bg_mode.set("reuse latest")
    app.scan_name.set("seq reuse")
    app.link.scope.put(":TIMebase:SCALe", "0.0005")   # a preset's span, too short
    cmp_seen = []
    _lcd = app._live_cmp_done
    app._live_cmp_done = lambda p, k, r: (cmp_seen.append((os.path.basename(p), r["n_done"])),
                                          _lcd(p, k, r))
    app.do_seq_start()
    settle(root, app, timeout=600)
    app._live_cmp_done = _lcd
    check("while it runs, the scan not shown is re-analysed after its steps and drawn over "
          "the shown one (compare ticked)",
          len(cmp_seen) >= 3 and {n.lower() for n, _ in cmp_seen} == {"seq_reuse_x1_30_x2_30"}
          and cmp_seen[0][1] < cmp_seen[-1][1] and app.fig_ext._cmp_on.get(),
          cmp_seen[:4])
    mans = {}
    for n_ in ("seq_reuse_X1_30_X2_0", "seq_reuse_X1_30_X2_30"):
        mp_ = os.path.join(app.outdir.get(), n_, f"{n_}_scan.json")
        mans[n_] = _js.load(open(mp_, encoding="utf-8")) if os.path.isfile(mp_) else {}
    srcs = [{k: (m.get("borrowed") or {}).get(k, {}).get("source") for k in ("dark", "background")}
            for m in mans.values()]
    check("a sequence reusing the latest dark / background: every step measured, the "
          "second scan borrowing what the first borrowed",
          all(m and all(x["status"] == "done" for x in m["steps"]) for m in mans.values())
          and srcs[0] == srcs[1] and all(srcs[0].values()), srcs)
    _c = app.gather()
    _e, waves_, _f = app._seq_waves(_c)
    span_ = app._scope_span()
    longest_ = max(w.period for w in waves_)
    b_, a_ = float(_c["awg"]["scope_before_ms"]) * 1e-3, float(_c["awg"]["scope_after_ms"]) * 1e-3
    check("the sequence set the scope to the AWG tab's span around the longest record",
          span_ is not None and span_[0] <= -b_ + 1e-9 and span_[1] >= longest_ + a_ - 1e-9,
          (span_, longest_))
    import shutil as _sh
    _o = app.outdir.get()
    for n_ in ("seq_reuse_X1_30_X2_0", "seq_reuse_X1_30_X2_30"):
        _sh.copytree(os.path.join(_o, n_), os.path.join(_o, "cp" + n_))
        for f_ in os.listdir(os.path.join(_o, "cp" + n_)):
            if f_.startswith(n_):
                os.rename(os.path.join(_o, "cp" + n_, f_), os.path.join(_o, "cp" + n_, "cp" + f_))
        mp_ = os.path.join(_o, "cp" + n_, f"cp{n_}_scan.json")
        m_ = _js.load(open(mp_, encoding="utf-8"))
        m_["name"] = "cp" + n_
        m_["plan"]["series"]["members"] = ["cpseq_reuse_X1_30_X2_0", "cpseq_reuse_X1_30_X2_30"]
        _js.dump(m_, open(mp_, "w", encoding="utf-8"))
    gui.scanmod.rename(_o, "cpseq_reuse_X1_30_X2_0", "renamed_X1_30_X2_0", log=lambda *_: None)
    sib = _js.load(open(os.path.join(_o, "cpseq_reuse_X1_30_X2_30",
                                     "cpseq_reuse_X1_30_X2_30_scan.json"), encoding="utf-8"))
    own = _js.load(open(os.path.join(_o, "renamed_X1_30_X2_0", "renamed_X1_30_X2_0_scan.json"),
                        encoding="utf-8"))
    check("renaming a sequence member: its own and its sibling's member lists follow",
          sib["plan"]["series"]["members"] == own["plan"]["series"]["members"]
          == ["renamed_X1_30_X2_0", "cpseq_reuse_X1_30_X2_30"], sib["plan"]["series"]["members"])
    app.dark_mode.set("none")
    app.bg_mode.set("none")
    # nothing loaded, scans compared: the Compare tab and the overlay tabs draw them
    saved_res = app.result
    app.result = None
    app.redraw(app.fig_cmp)
    check("nothing loaded: Compare draws the compared scans, the difference from the first",
          len(app.fig_cmp.axes) == 3, len(app.fig_cmp.axes))
    app.redraw(app.fig_ext)
    t_ = app.fig_ext.axes[0].get_title() if app.fig_ext.axes else ""
    check("nothing loaded: Extinction draws the first compared scan and the other",
          t_.startswith("Extinction ratio along the ramp (seq_") and "with 1 compared" in t_, t_)
    app.result = saved_res
    # the record follows its parts; the scope's span is set apart
    app.av["hold_ms"].set("19")
    app.av["tail_ms"].set("0.5")
    app.do_awg_preview()
    root.update()
    check("a 22 ms ramp previews (the record is the sum of its parts)",
          abs(app.awg_wave.period - 22.002e-3) < 1e-9 and "record 22 ms" in app.a_record.cget("text"),
          app.a_record.cget("text"))
    app.av["scope_before_ms"].set("2")
    app.av["scope_after_ms"].set("5")
    c_ = app.gather()
    div, pos = app._awg_scope_tb(c_, app.awg_wave)
    check("the scope's span: 2 ms before the trigger to 5 ms after the record (2 figures)",
          abs((pos - div) + 2e-3) < 1e-9 and 27.0e-3 <= pos + 9 * div < 28.5e-3,
          f"{(pos-div)*1e3:.2f}..{(pos+9*div)*1e3:.2f} ms")
    app.av["hold_ms"].set("8")
    app.av["scope_before_ms"].set("0.2")
    app.av["scope_after_ms"].set("0")

    def wait_for(cond, limit=5.0):
        t1 = time.time()
        while not cond() and time.time() - t1 < limit:
            root.update()
            time.sleep(0.02)
        # the status line is updated by the pump (every 80 ms)
        t1 = time.time()
        while time.time() - t1 < 0.3:
            root.update()
            time.sleep(0.02)
    app.do_awg_park()
    wait_for(lambda: app.awg_sess.parked)
    check("Park: idle waveform, outputs still ON", app.awg_sess.parked
          and all(app.bench.awg_on.values()))
    app.do_awg_off()                 # asks (stubbed yes) under the never-float rule
    wait_for(lambda: not any(app.bench.awg_on.values()))
    check("Outputs OFF, after asking", not any(app.bench.awg_on.values())
          and "OFF" in app.awg_lbl.cget("text"),
          app.awg_lbl.cget("text") + " | " + app.logbox.get("1.0", "end")[-300:])
    app.do_awg_load()                # the ramp again (dry-run passed), then ON
    settle(root, app)
    app.do_awg_on()
    settle(root, app)
    check("found angles go to the lab log",
          sum(r["kind"].startswith("find") for r in lablog.read(app.outdir.get())) >= 3)
    print(f"\nfigures in {out}")
    bench = app.bench
    live = all(bench.awg_on.values()) and not app.awg_sess.parked
    sess = app.awg_sess
    app.on_close()
    check("closing the window parks the AWG (never-float rule)", live and sess.parked
          and all(bench.awg_on.values()) and __import__("numpy").ptp(bench.awg_drive[2][1]) == 0)
    check("settings saved to the sandbox", os.path.exists(cfgmod.CONFIG_PATH))
    return report()


def report():
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("GUI OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

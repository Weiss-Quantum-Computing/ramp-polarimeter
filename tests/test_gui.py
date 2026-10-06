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

from rampol import gui  # noqa: E402

# the dark prompts answer themselves
gui.messagebox.askokcancel = lambda *a, **k: True
gui.messagebox.showinfo = lambda *a, **k: None
gui.messagebox.askyesno = lambda *a, **k: True

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
    print("\nconnect")
    app.do_connect_scope()
    settle(root, app)
    app.bench.rng = __import__("numpy").random.default_rng(5)
    app._sim_parts[0].realtime = 0.0
    app.do_connect_ell()
    settle(root, app)
    check("scope connected (simulated)", app.link is not None,
          app.scope_status.cget("text"))
    check("analyzer connected (simulated)", app.rot is not None,
          app.ell_status.cget("text"))
    app.goto_var.set("123.4")
    app.do_goto()
    settle(root, app)
    pos = float(app.pos_label.cget("text").split()[1])
    check("go to lands within 0.05 deg", abs(pos - 123.4) < 0.05, f"{pos:.3f}")
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
    app.preset.set("Spin echo 16.7 ms (2 legs)")
    app.preset_picked()
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
    check("every figure tab drew", len(os.listdir(out)) == 8, sorted(os.listdir(out)))
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

    print("\nstopping part-way: what was measured is loaded and drawn")
    app.scan_name.set("stopped part way")
    app.with_dark.set(False)
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
    check("the stopped scan is loaded", res is not None and res["d"].name == "stopped-part-way"
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
    check("measurement modes are tabs", tabs == ["Ramp scan", "Null refine", "Bias points",
                                                "ILC target"], tabs)
    app.bv["biases"].set("0:90:45")
    app.bv["shots"].set("4")
    app.bv["name"].set("gui-bias")
    app._sim_parts[0].realtime = 0.0
    app.do_start_bias()
    settle(root, app, timeout=300)
    br = app.bias_result
    pts = (br or {}).get("points", [])
    check("bias run measured and loaded", len(pts) == 3,
          ", ".join(f"{p['bias']:g}: ER {p.get('er') or 0:.0f}" for p in pts))
    check("ER at rest near the bench's 3333", bool(pts) and bool(pts[0].get("er"))
          and abs(pts[0]["er"] / 3333 - 1) < 0.2, pts and pts[0].get("er"))
    check("AWG outputs off afterwards", not any(app.bench.awg_on.values()))
    check("the beam is back (dark unblocked)", app.bench.imax > 0)
    app.nb.select(app.fig_bias._frame)
    root.update()
    app.fig_bias.savefig(os.path.join(out, "Bias_points.png"))
    check("bias tab drew four panels", len(app.fig_bias.axes) >= 4, len(app.fig_bias.axes))
    app.nb.select(app.fig_ilc._frame)
    root.update()
    check("ILC tab says what to do with no comparison yet",
          any("Compare shown scan" in t.get_text() for ax in app.fig_ilc.axes for t in ax.texts))
    print(f"\nfigures in {out}")
    app.on_close()
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

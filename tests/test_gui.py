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

    print("\nscan")
    app.do_start_scan()
    settle(root, app, timeout=300)
    check("scan loaded after it ran", app.result is not None)
    if app.result is None:
        print(app.logbox.get("1.0", "end")[-3000:])
        return report()
    pol = app.result["pol"]
    check("12 angles fitted", pol["n_angles"] == 12, pol["n_angles"])
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
    check("every figure tab drew", len(os.listdir(out)) == 6, sorted(os.listdir(out)))
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

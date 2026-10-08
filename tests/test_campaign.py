"""The headless campaign runner on the simulated bench: a grid stage, a ramps
stage borrowing its darks, both resumable.

    python tests/test_campaign.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import analysis as an, bias, config, hw  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def main():
    print("\ncampaign.py on the simulated bench")
    tmp = tempfile.mkdtemp(prefix="rampol-camp-")
    cfg_path = os.path.join(tmp, "config.json")
    cfg = json.loads(json.dumps(config.DEFAULTS))
    cfg["outdir"] = os.path.join(tmp, "out")
    cfg["simulate"] = True
    cfg["scan"].update(step=45.0, shots=2, points=4000, stop=170.0)
    cfg["awg"].update(scope_before_ms=0.5, scope_after_ms=0.5, hold_ms=4.0, seq_settle_s=0.0)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    # two synthetic ILC target pairs in the suite's format (HV volts, 2 us grid)
    tdir = os.path.join(tmp, "targets")
    os.makedirs(tdir)
    import numpy as _np
    tt = _np.arange(0, 6000, 2.0)
    prof = _np.clip((tt - 500) / 1000, 0, 1)
    prof = 0.5 - 0.5 * _np.cos(_np.pi * prof)
    prof[tt > 3500] = _np.clip(0.5 + 0.5 * _np.cos(_np.pi * _np.clip((tt[tt > 3500] - 3500) / 1000, 0, 1)), 0, 1)
    for stem, (p1, p2) in (("T1", (5128.3, 0.0)), ("T2", (2564.15, 5137.4))):
        for suf, pk in (("X1", p1), ("X2", p2)):
            with open(os.path.join(tdir, f"target_{stem}{suf}.csv"), "w", encoding="utf-8") as fh:
                fh.write(f"# {stem} {suf}: test target\ntime_us,voltage_V\n")
                for x, y in zip(tt, prof * pk):
                    fh.write(f"{x:.6f},{y:.6f}\n")
    plan = {"stages": [
        {"stage": "grid", "name": "g", "x1": "0, 40", "x2": "0", "how": "pairs", "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "track": True, "track_ms": 20.0},
        {"stage": "ramps", "name": "r", "x1": "0, 40", "x2": "0", "how": "pairs",
         "darks_from": "g"},
        {"stage": "grid", "name": "g2", "x1": "20", "x2": "20", "how": "pairs", "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "darks_from": "g", "track": False},
        {"stage": "compensate", "name": "c", "x1": 40, "x2": 0, "iterations": 2, "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "darks_from": "g", "tail_ms": 30.0,
         "dt_us": 20.0, "track_ms": 10.0},
        {"stage": "transients", "name": "t", "x1": "40, 0", "x2": "0, 40", "how": "pairs",
         "hold_ms_list": [0.5, 5.0], "nulls_from": "g", "darks_from": "g", "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "period_ms": 100.0},
        {"stage": "grid", "name": "neg", "x1": "-30", "x2": "0", "how": "pairs", "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "darks_from": "g", "track": False,
         "allow_negative": True},
        {"stage": "targets", "name": "suite", "dir": tdir, "stems": "all", "darks_from": "g",
         "awg": {"scope_before_ms": 0.5, "scope_after_ms": 2.0}}]}
    plan_path = os.path.join(tmp, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh)
    cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "tools", "campaign.py"),
           plan_path, "--yes", "--simulate", "--config", cfg_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    check("the campaign ran to the end", r.returncode == 0 and "Campaign done" in out,
          out[-1500:] if r.returncode else "")
    g = bias.load(os.path.join(cfg["outdir"], "g"))
    check("grid stage: 2 points, tracked, finished",
          len(g["points"]) == 2 and g.get("finished") and all(p.get("track") for p in g["points"]),
          len(g["points"]))
    g2 = bias.load(os.path.join(cfg["outdir"], "g2"))
    check("second grid stage took its darks from the first (no beam block)",
          g2.get("darks_from", {}).get("run") == "g" and g2["dark"] == g["dark"]
          and len(g2["points"]) == 1 and g2["points"][0].get("er"),
          g2.get("darks_from"))
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    names = ["r_X1_0_X2_0", "r_X1_40_X2_0"]
    mans = {}
    for n in names:
        p = os.path.join(cfg["outdir"], n, f"{n}_scan.json")
        mans[n] = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    check("ramps stage: one scan per pair, every step done, hold-null angles on the 40 deg one",
          all(m and all(s["status"] == "done" for s in m["steps"]) for m in mans.values())
          and mans["r_X1_40_X2_0"]["plan"]["hold_angles"]
          and all(x.get("stayed") for x in mans["r_X1_40_X2_0"]["steps"]
                  if x["kind"] == "scan" and not x.get("hold_null")),
          {n: len(m.get("steps", [])) for n, m in mans.items()})
    check("the ramp scans borrowed the grid run's coarse dark as their background",
          all((m.get("borrowed") or {}).get("background", {}).get("source") == "g"
              for m in mans.values()),
          {n: (m.get("borrowed") or {}).get("background", {}).get("source") for n, m in mans.items()})
    d = an.load_scan(os.path.join(cfg["outdir"], "r_X1_40_X2_0"), sg.load_capture)
    pol = an.polarization(d, angle_gain=False)
    segs = [s["kind"] for s in an.segments(pol["t"], pol["rotation"])]
    import numpy as np
    reach = float(np.max(np.abs(pol["rotation"])))
    check("the 40 deg ramp scan analyses: rest / up / hold / down / after, hold near 40 deg",
          segs == ["rest", "up", "hold", "down", "after"] and abs(reach - 40) < 2, (segs, reach))
    cfold = os.path.join(cfg["outdir"], "c")
    hist = json.load(open(os.path.join(cfold, "compensation.json"), encoding="utf-8"))
    its = hist.get("iterations", [])
    c1 = bias.load(os.path.join(cfg["outdir"], "c_it1"))
    corr = c1["plan"].get("correction", {}).get("EO1")
    check("compensate stage: 2 iterations, the second played with the first's correction, "
          "the correction on file, the drift-free bench's error small",
          len(its) == 2 and hist.get("finished") and corr and len(corr["u_V"]) > 100
          and os.path.isfile(os.path.join(cfold, "correction_it2.csv"))
          and all(r["err_rms_mdeg"] < 100 for r in its),
          [(r["iteration"], round(r["err_rms_mdeg"], 1)) for r in its])
    for h, nm in ((0.5, "t_h0p5"), (5.0, "t_h5")):
        th = bias.load(os.path.join(cfg["outdir"], nm))
        tk = [q.get("track") or {} for q in th["points"]]
        trk = bias.load_tracks(os.path.join(cfg["outdir"], nm))
        check(f"transients at a {h:g} ms hold: 2 track-only points, the null from the grid "
              f"where it has the point and from 4 angles where not, creep and after-fall "
              f"numbers, traces on disk",
              len(th["points"]) == 2 and all(q.get("track_only") for q in th["points"])
              and [q["null_from"] for q in th["points"]] == ["plan", "4 angles"]
              and all("hold_slope_mdeg_ms" in x and "after_1ms_mdeg" in x for x in tk)
              and len(trk) == 2 and th["points"][0]["hold_ms"] == h,
              [(q["x1"], q["x2"], q["null_from"], round(x.get("after_1ms_mdeg", 0))) for q, x in zip(th["points"], tk)])
    neg = bias.load(os.path.join(cfg["outdir"], "neg"))
    check("a negative bias is driven when the plan allows it, and read",
          len(neg["points"]) == 1 and neg["points"][0]["x1"] == -30.0 and neg["points"][0].get("er")
          and abs(neg["points"][0]["phi_mon"] + 30) < 1.0,
          (neg["points"][0].get("phi_mon"), neg["points"][0].get("er")))
    mans = {}
    for n in ("suite_T1", "suite_T2"):
        p = os.path.join(cfg["outdir"], n, f"{n}_scan.json")
        mans[n] = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    check("targets stage: a ramp scan per suite pair, every step done, the drive on record "
          "with its files and peak rotations (T1: X1 90, T2: 45 + 90)",
          all(m and all(x["status"] == "done" for x in m["steps"]) for m in mans.values())
          and abs(mans["suite_T1"]["drive"]["ends_deg"]["X1"] - 90) < 0.1
          and abs(mans["suite_T2"]["drive"]["ends_deg"]["X1"] - 45) < 0.1
          and abs(mans["suite_T2"]["drive"]["ends_deg"]["X2"] - 90) < 0.1
          and "files" in mans["suite_T1"]["drive"],
          {n: (m.get("drive") or {}).get("ends_deg") for n, m in mans.items()})
    d2 = an.load_scan(os.path.join(cfg["outdir"], "suite_T2"), sg.load_capture)
    p2 = an.polarization(d2, angle_gain=False)
    check("the T2 target turned the simulated light by its 135 deg",
          abs(float(np.max(np.abs(p2["rotation"]))) - 135) < 3, float(np.max(np.abs(p2["rotation"]))))
    # a second run of the same plan: everything finished, nothing measured again
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out2 = r2.stdout + r2.stderr
    check("run again: finished stages skipped, ramp scans resumed with 0 steps to measure",
          r2.returncode == 0 and out2.count("already finished") == 6 and out2.count("0 steps to measure") == 2,
          out2[-800:] if not (r2.returncode == 0 and "0 steps to measure" in out2) else "")
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Campaign OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

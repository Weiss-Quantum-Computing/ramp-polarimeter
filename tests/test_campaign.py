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
    plan = {"stages": [
        {"stage": "grid", "name": "g", "x1": "0, 40", "x2": "0", "how": "pairs", "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "track": True, "track_ms": 20.0},
        {"stage": "ramps", "name": "r", "x1": "0, 40", "x2": "0", "how": "pairs",
         "darks_from": "g"},
        {"stage": "grid", "name": "g2", "x1": "20", "x2": "20", "how": "pairs", "shots": 2,
         "points": 4000, "upload_settle_s": 0.0, "darks_from": "g", "track": False}]}
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
    # a second run of the same plan: everything finished, nothing measured again
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out2 = r2.stdout + r2.stderr
    check("run again: finished stages skipped, ramp scans resumed with 0 steps to measure",
          r2.returncode == 0 and out2.count("already finished") == 2 and "0 steps to measure" in out2,
          out2[-800:] if not (r2.returncode == 0 and "0 steps to measure" in out2) else "")
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Campaign OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""A running lab log: one row per scan, bias run and found angle in
<outdir>/lab_log.csv, so trends across days are one file to open.

A row is keyed by (kind, name): analysing a scan again updates its row
instead of adding another, so the log holds each measurement once, with the
numbers of its latest full analysis. Columns are fixed (COLUMNS); a value a
kind does not have is left blank. The file is rewritten through a temporary
file, and a log that Excel holds open is reported, not lost: the row waits
for the next write.
"""
import csv
import datetime
import os

from .config import replace_retrying

NAME = "lab_log.csv"
COLUMNS = ["kind", "name", "measured", "logged", "status", "angles", "shots",
           "pd_vdiv", "subtracted", "subtracted_mV", "shots_dropped",
           "rotation_min_deg", "rotation_max_deg", "psi_rest_deg",
           "direct_er_min", "direct_er_min_at", "er_fit_rest", "er_fit_holds",
           "drift_resid", "fit_resid_mV", "result", "software", "ilc", "folder",
           "notes"]


def path(outdir):
    return os.path.join(outdir, NAME)


def read(outdir):
    p = path(outdir)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def upsert(outdir, row):
    """Add `row` (dict of COLUMNS) or replace the row with its kind and name.
    Returns the log's path. Raises OSError if the file cannot be written
    (open in Excel)."""
    rows = read(outdir)
    row = {k: _fmt(row.get(k, "")) for k in COLUMNS}
    row["logged"] = datetime.datetime.now().isoformat(timespec="seconds")
    for i, r in enumerate(rows):
        if r.get("kind") == row["kind"] and r.get("name") == row["name"]:
            # keep a note typed into the log by hand
            if r.get("notes") and not row["notes"]:
                row["notes"] = r["notes"]
            rows[i] = row
            break
    else:
        rows.append(row)
    rows.sort(key=lambda r: (r.get("measured") or "", r.get("name") or ""))
    os.makedirs(outdir, exist_ok=True)
    p = path(outdir)
    tmp = p + ".part"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    replace_retrying(tmp, p)
    return p


def rename(outdir, old, new, folder=None):
    """A renamed scan: its row takes the new name (and folder), and the old
    name is replaced where other rows cite it (a borrowed background).
    Returns the number of rows changed. Raises OSError if the log cannot be
    written (open in Excel)."""
    import re
    rows = read(outdir)
    pat = re.compile(r"(?<![\w.-])" + re.escape(old) + r"(?![\w-])")
    n = 0
    for r in rows:
        before = dict(r)
        if r.get("kind") == "ramp scan" and r.get("name") == old:
            r["name"] = new
            if folder:
                r["folder"] = folder
        for k in ("result", "subtracted", "direct_er_min_at", "notes"):
            if r.get(k):
                r[k] = pat.sub(new, r[k])
        n += r != before
    if n:
        p = path(outdir)
        tmp = p + ".part"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        replace_retrying(tmp, p)
    return n


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def scan_row(summary):
    """A ramp scan's row from analysis.scan_summary."""
    from .provenance import short
    s = summary
    segs = s.get("segments") or []
    rest = [x["er_fit_median"] for x in segs if x["kind"] == "rest"]
    holds = [x["er_fit_median"] for x in segs if x["kind"].startswith("hold")]
    lo = s.get("direct_er_min") or {}
    prov = s.get("provenance") or {}
    return {
        "kind": "ramp scan", "name": s["name"], "measured": s.get("created", ""),
        "status": "complete" if s.get("complete") else
        f"partial {s.get('steps_done')}/{s.get('steps_total')}",
        "angles": s.get("n_angles"), "shots": s.get("shots_per_angle"),
        "pd_vdiv": s.get("pd_vdiv"), "subtracted": s.get("subtracted") or "",
        "subtracted_mV": s.get("subtracted_mV"),
        "shots_dropped": (f"{s['shots_dropped']}/{s['shots_total']}"
                          if "shots_dropped" in s else ""),
        "rotation_min_deg": s.get("rotation_min_deg"),
        "rotation_max_deg": s.get("rotation_max_deg"),
        "psi_rest_deg": s.get("psi_rest_deg"),
        "direct_er_min": lo.get("er"),
        "direct_er_min_at": (f"{lo['kind']} {lo['seg']}, {lo['t_ms']:.3f} ms, rotation "
                             f"{lo['rotation']:.1f} deg, analyzer {lo['theta']:.2f}"
                             if lo else ""),
        "er_fit_rest": rest[0] if rest else None,
        "er_fit_holds": "; ".join(f"{x:.0f}" for x in holds),
        "drift_resid": s.get("drift_resid"),
        "fit_resid_mV": s.get("fit_residual_mV_median"),
        "result": s.get("corrections", ""),
        "software": short({"software": prov.get("software")}) if prov else "",
        # what played into the Treks: this window's AWG when the scan says so
        "ilc": (f"AWG (this window): {s['drive'].get('label', '')}" if s.get("drive")
                else short({"ilc_state_files": prov.get("ilc_state_files")}) if prov else ""),
        "folder": s.get("folder", ""),
        "notes": s.get("notes", ""),
    }


def bias_row(man):
    """A bias run's row from its manifest (rampol.bias.load)."""
    from .provenance import short
    pts = man.get("points", [])
    ers = [(p["er"], p["bias"]) for p in pts if p.get("er")]
    lo = min(ers) if ers else None
    tf = man.get("transfer") or {}
    prov = man.get("provenance") or {}
    return {
        "kind": "bias points", "name": man.get("name", ""),
        "measured": man.get("created", ""),
        "status": "complete" if man.get("finished") and len(pts) == len(man.get("biases", []))
        else f"{len(pts)}/{len(man.get('biases', []))} points",
        "shots": (man.get("plan") or {}).get("shots"),
        "direct_er_min": lo[0] if lo else None,
        "direct_er_min_at": f"bias {lo[1]:g} deg" if lo else "",
        "result": (f"light/monitors gain {tf['gain']:.4f}, {tf['rms_resid']*1e3:.0f} mdeg "
                   f"rms left" if tf.get("gain") is not None else ""),
        "software": short({"software": prov.get("software")}) if prov else "",
        "ilc": "",
        "folder": man.get("folder", ""),
    }


def find_row(out, when=None):
    """A Find angle / Malus scan result's row (bias.find_extremum,
    bias.malus_scan)."""
    when = when or datetime.datetime.now().isoformat(timespec="seconds")
    w = out.get("window")
    if out.get("static"):
        where = "static light, LINE trigger, one line period"
    elif out.get("bias") is not None:
        where = f"AWG hold at {out['bias']:g} deg"
    else:
        where = "whole record" if not w else f"{w[0]*1e3:.2f}..{w[1]*1e3:.2f} ms"
    if out["kind"] == "scan":
        kind = "malus scan"
        res = (f"maximum at analyzer {out['angle_max']:.2f} deg, minimum at "
               f"{out['angle_min']:.2f} deg ({where}); Imax {out['imax']:.4f} V, Imin "
               f"{out['imin']*1e3:.3f} +- {out.get('sig_imin', 0)*1e3:.3f} mV, ER "
               f"{'>' if out.get('er_lower') else ''}{out.get('er', 0):.0f}; "
               f"{out.get('offset_note', '')}")
    else:
        kind = f"find {out['kind']}"
        res = (f"{out['kind']} transmission at analyzer {out['angle']:.3f} +- "
               f"{out['sig']*1e3:.0f} mdeg ({where}), level {out['level']*1e3:.3f} mV "
               f"at {out['vdiv']*1e3:g} mV/div; {out.get('offset_note', 'raw')}")
    return {"kind": kind, "name": f"{kind.replace(' ', '_')}_{when.replace(':', '')}",
            "measured": when, "status": "done", "result": res}


def dry_row(reports, label, when):
    """An AWG dry run's row: what was played, passed or not, and the numbers."""
    when_s = when.isoformat(timespec="seconds")
    ok = all(r.get("ok") for r in reports)
    last = reports[-1] if reports else {}
    nums = []
    both = (last.get("steps") or {}).get("both") or {}
    for n, r in both.items():
        nums.append(f"{n} gain {r.get('gain', float('nan')):.4f}, delay "
                    f"{r.get('delay_us', float('nan')):.1f} us")
    res = ("PASSED" if ok else "FAILED: " + "; ".join(last.get("problems", [])[:3]))
    return {"kind": "AWG dry run", "name": f"dryrun_{when_s.replace(':', '')}",
            "measured": when_s, "status": "passed" if ok else "failed",
            "result": f"{label}: {len(reports)} waveform(s) {res}" + (
                f" ({'; '.join(nums)})" if nums else "")}

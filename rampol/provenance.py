"""What produced a measurement: the software versions and the ILC drive, as
found when it started. Written into every scan and bias-run manifest
("provenance"), so a scan can be tied to the code and the drive behind it
later.

Software: the git commit of ramp-polarimeter, EOM-ILC and Scope Grab, and
whether each had uncommitted changes. `git` is used when it is on PATH; else
the commit is read from the .git folder itself (no dirty flag then).

ILC: the state files the ILC target tab points at (drive_*.state.npz) - name,
channel, iteration, file time, the target's fingerprint (the same fields as
eomilc.corrections.target_fingerprint) and a hash of the drive u. These are
the files CONFIGURED here: if the bench plays something else, the hash will
not match the drive the ILC panel shows for it.
"""
import datetime
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_dir(path):
    p = os.path.abspath(path)
    if os.path.isfile(p):
        p = os.path.dirname(p)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return p
        up = os.path.dirname(p)
        if up == p:
            return None
        p = up


def _read_head(repo):
    """(commit, branch) from .git without the git program."""
    g = os.path.join(repo, ".git")
    if os.path.isfile(g):                     # a worktree: "gitdir: <path>"
        with open(g, encoding="utf-8") as fh:
            g = fh.read().split(":", 1)[1].strip()
    with open(os.path.join(g, "HEAD"), encoding="utf-8") as fh:
        head = fh.read().strip()
    if not head.startswith("ref:"):
        return head, None
    ref = head.split(":", 1)[1].strip()
    p = os.path.join(g, *ref.split("/"))
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return fh.read().strip(), ref.rsplit("/", 1)[-1]
    packed = os.path.join(g, "packed-refs")
    if os.path.exists(packed):
        with open(packed, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0], ref.rsplit("/", 1)[-1]
    return None, ref.rsplit("/", 1)[-1]


def git_info(path):
    """{repo, commit (12 chars), branch, dirty (True/False/None = unknown)}
    for the repository holding `path`, or {repo: None} outside one."""
    repo = _git_dir(path)
    if repo is None:
        return {"repo": None}
    out = {"repo": repo, "commit": None, "branch": None, "dirty": None}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        def git(*args):
            return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                                  text=True, timeout=10, creationflags=flags)
        r = git("rev-parse", "HEAD")
        if r.returncode == 0:
            out["commit"] = r.stdout.strip()[:12]
            b = git("rev-parse", "--abbrev-ref", "HEAD")
            out["branch"] = b.stdout.strip() if b.returncode == 0 else None
            st = git("status", "--porcelain", "--untracked-files=no")
            out["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        commit, branch = _read_head(repo)
        out["commit"] = commit[:12] if commit else None
        out["branch"] = branch
    except OSError:
        pass
    return out


def software(cfg):
    from . import __version__
    out = {"rampol": __version__,
           "python": platform.python_version(), "numpy": np.__version__,
           "ramp-polarimeter": git_info(HERE)}
    for key, path in (("EOM-ILC", cfg.get("eomilc_path")),
                      ("Scope Grab", cfg.get("scope_grab_path"))):
        if path and os.path.exists(path):
            out[key] = git_info(path)
        else:
            out[key] = {"repo": None, "missing": str(path)}
    return out


def ilc_state(path):
    """The ILC state file's identity: what drive and target it holds."""
    if not path:
        return None
    out = {"path": os.path.abspath(path)}
    try:
        st = os.stat(path)
        out["modified"] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds")
        z = np.load(path, allow_pickle=True)
        for k in ("name", "channel"):
            if k in z:
                out[k] = str(z[k])
        if "iteration" in z:
            out["iteration"] = int(z["iteration"])
        if "target" in z:
            v = np.asarray(z["target"], float)
            dt = float(z["dt"]) if "dt" in z else float(np.median(np.diff(z["t"])))
            out.update(target_n=int(len(v)), target_dt_us=round(dt * 1e6, 6),
                       target_peak_V=round(float(np.max(np.abs(v))), 4),
                       target_rms_V=round(float(np.sqrt(np.mean(v * v))), 4))
        if "u" in z:
            u = np.ascontiguousarray(np.asarray(z["u"], np.float64))
            out["drive_sha1"] = hashlib.sha1(u.tobytes()).hexdigest()[:12]
    except Exception as exc:                  # a missing or foreign file: say so
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def collect(cfg):
    """Everything above, for a manifest."""
    ilc = cfg.get("ilc") or {}
    return {"recorded": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "software": software(cfg),
            "ilc_state_files": {k: ilc_state(ilc.get(k)) for k in ("x1", "x2")
                                if ilc.get(k)},
            "argv": sys.argv[:1]}


def short(prov):
    """One line: 'rampol 0.x @ab12cd34ef56*, EOM-ILC @..., ILC X1 it 15'."""
    if not prov:
        return ""
    sw = prov.get("software", {})
    parts = []
    for key in ("ramp-polarimeter", "EOM-ILC", "Scope Grab"):
        g = sw.get(key) or {}
        if g.get("commit"):
            parts.append(f"{key} {g['commit'][:8]}{'*' if g.get('dirty') else ''}")
    for k, s in (prov.get("ilc_state_files") or {}).items():
        if s and "iteration" in s:
            parts.append(f"ILC {s.get('name', k)} it {s['iteration']} "
                         f"drive {s.get('drive_sha1', '?')[:8]}")
    return ", ".join(parts)

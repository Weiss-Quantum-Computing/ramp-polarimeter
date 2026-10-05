"""The two instruments as the scan sees them.

The scope is Scope Grab's own `Scope`, loaded from scope_grab.py by file path
the way EOM-ILC loads it, so a capture here is the same NPZ + .txt the panel
writes and opens in its Compare box. The rotator is the ELL14 driver with one
addition: every angle is approached from the same side.
"""
import importlib.util
import os
import sys
import time


def load_scope_grab(path):
    """Import scope_grab.py from `path` (it puts its own folder on sys.path
    for scope_profiles). Cached in sys.modules under 'scope_grab'."""
    mod = sys.modules.get("scope_grab")
    if mod is not None and os.path.normcase(os.path.abspath(
            getattr(mod, "__file__", ""))) == os.path.normcase(os.path.abspath(path)):
        return mod
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scope_grab.py not found at {path}")
    spec = importlib.util.spec_from_file_location("scope_grab", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scope_grab"] = mod
    spec.loader.exec_module(mod)
    for name in ("write_capture", "setting_roots"):
        if not hasattr(mod, name):
            raise ImportError(
                f"{path} predates the headless API ({name} missing) - point "
                f"scope_grab_path at scope-grab-multi at dcf2c0b or later")
    return mod


class Cancelled(Exception):
    pass


class ScopeLink:
    """Acquisition on top of a scope_grab.Scope (or the simulator's stand-in,
    which has the same methods).

    A *block* is one stored capture: either a scope-built average of
    `count` triggers (:DIGitize on the MSO-X, read out at the 7680 points an
    averaged record has), or one single HRES shot read out at `points`. Blocks
    within an angle step the channel offsets across the dither, so the
    converter's per-code error pattern averages out across them instead of
    surviving every block identically."""

    def __init__(self, scope, log=print):
        self.scope = scope
        self.log = log

    @property
    def prof(self):
        return self.scope.prof

    def set_acquisition(self, mode, count):
        p = self.prof
        if mode == "average":
            self.scope.put(p.acq_type, p.avg_name)   # AVERage / AVERages
            self.scope.put(p.acq_count, str(int(count)))
        else:
            self.scope.put(p.acq_type, "HRESolution")

    def channel_state(self, chans):
        """{ch: (V/div, offset)} as the scope reports them now."""
        p = self.prof
        return {ch: (float(self.scope.get(p.ch_scale.format(ch=ch))),
                     float(self.scope.get(p.ch_offset.format(ch=ch))))
                for ch in chans}

    def set_channel(self, ch, scale=None, offset=None):
        p = self.prof
        if offset is not None:
            # park the offset first: a large offset can be illegal at the new
            # scale (MSO-X: about +-2 V below 0.5 V/div, +-20 V at or above)
            self.scope.put(p.ch_offset.format(ch=ch), "0")
        if scale is not None:
            self.scope.put(p.ch_scale.format(ch=ch), f"{scale:.6g}")
        if offset is not None:
            self.scope.put(p.ch_offset.format(ch=ch), f"{offset:.6g}")

    def apply(self, settings):
        """Write a preset's {scpi root: value}. The profile's write_first roots
        go first (an average count is ignored unless the type is AVERage)."""
        first = getattr(self.prof, "write_first", ())
        for scpi in sorted(settings, key=lambda s: s not in first):
            self.scope.put(scpi, settings[scpi])

    def acquire_blocks(self, chans, mode, blocks, shots, dither_codes=0,
                       points=None, wait_s=10.0, cancelled=None, on_block=None):
        """Acquire `blocks` captures of `chans`. For mode 'average' each block
        averages shots // blocks triggers on the scope; for 'single' each block
        is one shot and `blocks` is the shot count. on_block(k, recs, hits)
        receives each block as it arrives ({ch: Record}). The offsets are
        always put back, whatever happens."""
        cancelled = cancelled or (lambda: False)
        per = max(1, shots // blocks) if mode == "average" else 1
        plan = {}
        if dither_codes and blocks > 1:
            plan = self.scope.dither_plan(chans, dither_codes)
        try:
            if mode == "average":
                self.set_acquisition("average", per)
            for k in range(blocks):
                if cancelled():
                    raise Cancelled()
                if plan:
                    for ch, exc in self.scope.dither_step(plan, k, blocks).items():
                        self.log(f"  dither: CH{ch} offset refused ({exc})")
                if mode == "average":
                    hits = self.scope.accumulate(per, wait_s=wait_s,
                                                 cancelled=cancelled,
                                                 channels=tuple(chans))
                    if hits is None:
                        raise Cancelled()
                    if hits == 0:
                        raise RuntimeError(f"no trigger within {wait_s:g} s")
                    pmode, pts = self.scope.transfer_plan(True, None)
                else:
                    got = self.scope.single(wait_s=wait_s, cancelled=cancelled)
                    if got is None:
                        raise Cancelled()
                    if got is False:
                        raise RuntimeError(f"no trigger within {wait_s:g} s")
                    hits = 1
                    pmode, pts = self.scope.transfer_plan(False, points)
                recs = {ch: self.scope.record(ch, points_mode=pmode, points=pts)
                        for ch in chans}
                if on_block:
                    on_block(k, recs, hits)
        finally:
            if plan:
                for ch, exc in self.scope.restore_offsets(plan).items():
                    self.log(f"  dither: could not restore CH{ch} offset ({exc})")
            try:
                self.scope.run()
            except Exception:
                pass


class Rotator:
    """The ELL14 (or the simulator's) with a fixed approach direction.

    The mount's absolute move picks its own direction, so a target reached
    from above and from below can land on opposite sides of any backlash.
    `approach` goes to target - backoff first and finishes with a positive
    relative move, then reads the position back."""

    def __init__(self, dev, log=print):
        self.dev = dev
        self.log = log

    @property
    def zero(self):
        return self.dev.zero_offset

    @zero.setter
    def zero(self, value):
        self.dev.zero_offset = float(value) % 360.0

    def position(self):
        return self.dev.position()

    def home(self):
        return self.dev.home()

    def approach(self, angle, backoff=3.0, tol=0.05, settle=0.1):
        """Land on `angle` (analyzer frame) from below. Returns the
        read-back angle, unwrapped to the branch nearest `angle`."""
        if backoff > 0:
            self.dev.goto(angle - backoff, tol=max(tol, 0.2), settle=0.0)
            self.dev.move_by(backoff)
        else:
            self.dev.move_to(angle)
        time.sleep(settle)
        got = self.dev.position()
        err = (got - angle + 180.0) % 360.0 - 180.0
        if abs(err) > tol:
            # one correction from below, then accept and report what it is
            self.dev.goto(angle - backoff, tol=max(tol, 0.2), settle=0.0)
            self.dev.move_by(backoff)
            time.sleep(settle)
            got = self.dev.position()
            err = (got - angle + 180.0) % 360.0 - 180.0
            if abs(err) > tol:
                self.log(f"  analyzer landed {err:+.3f} deg from {angle:.3f}")
        return angle + err

    def close(self):
        self.dev.close()

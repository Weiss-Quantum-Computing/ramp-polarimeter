"""The pre-run scope check and the checked preset apply, against the
simulator: each thing that should stop a scan does, and a setting the scope
quietly refuses is reported. No hardware.

    python tests/test_checks.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import checks, config, hw, sim  # noqa: E402

FAILS = []
ROLES = {1: "CmdX1", 2: "PD", 3: "MonX1", 4: "MonX2"}
PRESET = config.PRESETS["Spin echo 16.7 ms (2 legs)"]
PLAN = {**config.DEFAULTS["scan"], **PRESET["scan"], "dither_codes": 3, "points": 20000}


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def bench_scope(sg, **kw):
    bench = sim.Bench(seed=4, legs_ms=(0.0, 16.667), ramp_up_ms=4.5, hold_ms=0.5, **kw)
    scope, ell, bench = sim.make(sg, roles=ROLES, bench=bench)
    link = hw.ScopeLink(scope, log=lambda *_: None)
    link.apply(link.preset_writes(PRESET, ROLES))
    return scope, link, bench


def levels(found, text):
    return [lv for lv, msg in found if text in msg]


def run(link, plan=PLAN):
    st = link.scope.read_settings()
    found = checks.settings_checks(st, link.prof, ROLES, plan)
    t, tr, st2 = link.test_shot(list(ROLES), plan["mode"], plan["points"])
    return found + checks.shot_checks(t, tr, st2, link.prof, ROLES, plan)


def main():
    sg = hw.load_scope_grab(config.DEFAULTS["scope_grab_path"])
    print("\npreset applied and read back")
    scope, link, bench = bench_scope(sg)
    scope.put(":TIMebase:SCALe", "1.0E-03")
    scope.put(":CHANnel2:SCALe", "5.0")
    scope.put(":TRIGger:SWEep", "AUTO")
    bad, errs = link.apply_checked(link.preset_writes(PRESET, ROLES))
    check("a hand-changed scope is put back, every setting confirmed", not bad and not errs,
          bad or errs)
    found = run(link)
    check("the preset's own setup passes", checks.summary(found) in ("OK", "INFO"),
          [f for f in found if f[0] in ("WARN", "FAIL")])
    check("ramps found inside the record", levels(found, "ramp activity").count("OK") == 3)

    print("\na setting the scope quietly refuses")
    real_write = scope.inst.write

    def refusing(text):
        if text.startswith(":TIMebase:POSition"):
            return                       # accepted on the wire, ignored by the scope
        real_write(text)
    scope.inst.write = refusing
    scope.inst.state[":TIMebase:POSition"] = "0.0E+00"
    bad, _ = link.apply_checked(link.preset_writes(PRESET, ROLES))
    check("it is reported with what was wanted and what the scope reads",
          ":TIMebase:POSition" in bad and bad[":TIMebase:POSition"][0] == "-7.0E-03", bad)
    scope.inst.write = real_write
    link.apply(link.preset_writes(PRESET, ROLES))

    print("\nwhat should stop a scan")
    scope.put(":TRIGger:SWEep", "AUTO")
    check("trigger sweep AUTO fails", "FAIL" in levels(run(link), "AUTO"))
    scope.put(":TRIGger:SWEep", "NORMal")
    scope.put(":CHANnel3:DISPlay", "0")
    check("a recorded channel switched off fails", "FAIL" in levels(
        checks.settings_checks(scope.read_settings(), link.prof, ROLES, PLAN), "switched off"))
    scope.put(":CHANnel3:DISPlay", "1")
    check("trigger wait not longer than the repetition fails", "FAIL" in levels(
        checks.settings_checks(scope.read_settings(), link.prof, ROLES,
                               dict(PLAN, wait_s=10.0)), "trigger wait"))
    scope.put(":CHANnel2:COUPling", "AC")
    check("AC-coupled PD warns", "WARN" in levels(
        checks.settings_checks(scope.read_settings(), link.prof, ROLES, PLAN), "AC coupled"))
    scope.put(":CHANnel2:COUPling", "DC")
    link.set_channel(2, 0.5, 1.5)               # PD tops at ~5 V: screen to 3.5 V
    check("a PD clipped at 0.5 V/div fails", "FAIL" in levels(run(link), "clipped"))
    link.set_channel(2, 1.0, 1.0)               # screen to 5.0 V: just off screen
    check("a PD running off the screen warns", "WARN" in levels(run(link), "off screen"))
    link.set_channel(2, 5.0, 10.0)
    check("a PD using under a quarter of the screen warns",
          "WARN" in levels(run(link), "smaller V/div"))
    link.set_channel(2, 1.0, 2.7)
    scope.put(":TIMebase:POSition", "-1.0E-02")  # record ends at +35 ms: fine
    scope.put(":TIMebase:SCALe", "2.0E-03")      # 20 ms record, -12..+8 ms: cuts leg 1
    check("a record that cuts a ramp off warns", "WARN" in levels(run(link), "edge of the record"))
    scope.put(":TIMebase:SCALe", "5.0E-03")
    scope.put(":TIMebase:POSition", "-7.0E-03")
    bench.imax = 0.0
    check("no light warns", "WARN" in levels(run(link), "light on"))
    bench.imax = 5.0
    scope.put(":TIMebase:POSition", "6.0E-03")   # record starts at +1 ms
    check("no pre-trigger stretch warns", "WARN" in levels(
        checks.settings_checks(scope.read_settings(), link.prof, ROLES, PLAN), "pre-trigger"))

    print("\nread-back comparison")
    check("short and long mnemonics match",
          hw.same_setting("HRESolution", "HRES") and hw.same_setting("NORMal", "NORM"))
    check("numbers to 0.1 %", hw.same_setting("5.0E-03", "+5.000E-03")
          and not hw.same_setting("5.0E-03", "5.1E-03"))
    check("switches by state", hw.same_setting("ON", "1") and not hw.same_setting("ON", "0"))
    check("different settings differ", not hw.same_setting("NORMal", "AUTO")
          and not hw.same_setting("LEFT", "CENT"))
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("Checks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

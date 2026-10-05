"""The ELL14 driver and the rotator wrapper against a fake serial port that
answers like the real mount (the IN reply is the one S/N 11400318 gave on
5 Oct 2026). No hardware.

    python tests/test_ell14.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rampol import hw  # noqa: E402
from rampol.ell14 import ELL14, ElliptecError, _to_hex32, _from_hex32  # noqa: E402

FAILS = []
PPR = 143360


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


class FakeSerial:
    """Elliptec protocol, address 0: in, gs, gp, ho, ma, mr."""

    def __init__(self, info="0E11400318201913010168" + f"{PPR:08X}"):
        self.timeout = 0.05
        self.is_open = True
        self.info = info
        self.pos = 0                  # pulses
        self.sent = []
        self.out = b""
        self.garbage = None

    def reset_input_buffer(self):
        self.out = b""

    def write(self, data):
        cmd = data.decode()
        self.sent.append(cmd)
        a, c, arg = cmd[0], cmd[1:3], cmd[3:]
        if c == "in":
            r = f"{a}IN{self.info}"
        elif c == "gs":
            r = f"{a}GS00"
        elif c == "gp":
            r = f"{a}PO{_to_hex32(self.pos)}"
        elif c == "ho":
            self.pos = 0
            r = f"{a}PO{_to_hex32(0)}"
        elif c == "ma":
            self.pos = _from_hex32(arg) % PPR
            r = f"{a}GS09\r\n{a}PO{_to_hex32(self.pos)}"     # busy, then landed
        elif c == "mr":
            self.pos = (self.pos + _from_hex32(arg)) % PPR
            r = f"{a}PO{_to_hex32(self.pos)}"
        else:
            r = f"{a}GS03"
        self.out += (self.garbage or r).encode() + b"\r\n"
        return len(data)

    def read_until(self, term=b"\n"):
        i = self.out.find(term)
        if i < 0:
            chunk, self.out = self.out, b""
            return chunk
        chunk, self.out = self.out[:i + 1], self.out[i + 1:]
        return chunk

    def close(self):
        self.is_open = False


def driver_checks():
    print("\nprotocol")
    ser = FakeSerial()
    rot = ELL14(ser=ser, cmd_timeout=0.5, move_timeout=0.5)
    check("identity parsed: ELL14, serial, pulses/rev",
          rot.serial_no == "11400318" and rot.pulses_per_rev == PPR)
    check("constructor sends only 'in'", ser.sent == ["0in"], ser.sent)
    got = rot.goto(45.0)
    check("goto 45 lands (busy status skipped)", abs(got - 45.0) < 0.003, f"{got:.4f}")
    rot.move_by(-50.0)
    check("negative relative move wraps through 0", abs(rot.position() - 355.0) < 0.003,
          f"{rot.position():.4f}")
    rot.zero_offset = 10.0
    check("zero offset: user frame = mount - zero", abs(rot.position() - 345.0) < 0.003)
    rot.goto(0.0)
    check("user 0 is mount 10", abs(ser.pos * 360 / PPR - 10.0) < 0.003)
    print("\nrefusals")
    ser.garbage = "0INxyz"
    try:
        rot.info()
        check("malformed IN reply raises", False)
    except ElliptecError:
        check("malformed IN reply raises", True)
    try:
        ELL14(ser=FakeSerial(info="0611400318201913010168" + f"{PPR:08X}"),
              cmd_timeout=0.3)
        check("a non-ELL14 device is refused", False)
    except ElliptecError:
        check("a non-ELL14 device is refused", True)


def rotator_checks():
    print("\napproach from below")
    ser = FakeSerial()
    dev = ELL14(ser=ser, cmd_timeout=0.5, move_timeout=0.5)
    rot = hw.Rotator(dev, log=lambda *_: None)
    dev.goto(200.0)
    ser.sent.clear()
    got = rot.approach(120.0, backoff=3.0, settle=0.0)
    moves = [c for c in ser.sent if c[1:3] in ("ma", "mr")]
    check("goes to target - backoff, then a positive relative move",
          len(moves) == 2 and moves[0][1:3] == "ma" and moves[1][1:3] == "mr"
          and _from_hex32(moves[1][3:]) > 0, moves)
    check("lands on target", abs(got - 120.0) < 0.003, f"{got:.4f}")
    got = rot.approach(0.0, backoff=3.0, settle=0.0)
    check("target 0 approached from 357, reported near 0 (not 360)", abs(got) < 0.003, f"{got:.4f}")


def main():
    driver_checks()
    rotator_checks()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("ELL14 OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

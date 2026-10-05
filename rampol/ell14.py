"""
Driver for the Thorlabs Elliptec ELL14 rotation mount (SM1), talking the
Elliptec ASCII serial protocol directly.

Copied from EOM-ILC eomilc_polarization_finetune/ell14.py at a556aff (the
reviewed copy - see ELL14_REVIEW.md there). This repository now holds the
canonical copy. First contact with real hardware 5 Oct 2026: ELL14 S/N
11400318 (2019, firmware 13) on COM3 answered `in`, `gs` and `gp`;
143360 pulses/rev (2.511 mdeg/pulse). Motion has not been exercised yet.

Hardware (either works, the code is identical):
  A) ELL14K bundle: mount -> ribbon -> interface board -> USB -> PC,
     board powered by its 5 V adapter. Board shows up as an FTDI COM port.
  B) Bare mount + 3.3 V USB-UART adapter + 5 V supply on the Picoflex header.

    pip install pyserial numpy

Library use:
    from ell14 import ELL14
    with ELL14() as rot:                # auto-finds the port; or ELL14("COM5")
        rot.home()
        rot.goto(45)                    # blocks until settled, verifies landing
        print(rot.position())

Command line:
    python ell14.py --scan              # list COM ports + Elliptec devices
    python ell14.py                     # auto port; self-test: home, 0/45/90/0
    python ell14.py -i                  # interactive: angle, 'h', 'p', 'f', 'q'
    python ell14.py COM5 --zero 12.3    # explicit port; 12.3 deg (mount) = user 0

Angles: "user frame" = mount angle - zero_offset. Set zero_offset so 0 deg is
your analyzer reference axis (find_polarizer_zero() does it with a photodiode).
"""

import time
import math
import re


def _serial_module():
    try:
        import serial
    except ImportError as exc:
        raise ImportError("ELL14 hardware access needs pyserial: pip install pyserial") from exc
    return serial


def _address(value):
    value = str(value).upper()
    if len(value) != 1 or value not in "0123456789ABCDEF":
        raise ValueError("address must be one hexadecimal digit (0-F)")
    return value


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value

STATUS = {
    0: "OK", 1: "communication timeout", 2: "mechanical timeout",
    3: "command error / not supported", 4: "value out of range",
    5: "module isolated", 6: "module out of isolation",
    7: "initializing error", 8: "thermal error", 9: "busy",
    10: "sensor error", 11: "motor error", 12: "out of range",
    13: "over current", 14: "unknown error",
}
BUSY = 9


class ElliptecError(RuntimeError):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


def _to_hex32(n: int) -> str:
    """Signed int -> 8-char uppercase two's-complement hex."""
    if not -(1 << 31) <= n < (1 << 31):
        raise ValueError("position exceeds signed 32-bit range")
    return f"{n & 0xFFFFFFFF:08X}"


def _from_hex32(s: str) -> int:
    """8-char two's-complement hex -> signed int."""
    if not re.fullmatch(r"[0-9A-Fa-f]{8}", s):
        raise ElliptecError(f"malformed position: {s!r}")
    v = int(s, 16)
    return v - (1 << 32) if v & 0x80000000 else v


def _wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


class ELL14:
    def __init__(self, port=None, address="0", ser=None, zero_offset_deg=0.0,
                 cmd_timeout=2.0, move_timeout=10.0):
        """
        port            : serial port (interface board or USB-UART adapter)
        address         : device bus address "0".."F" (factory default "0")
        ser             : existing serial.Serial to share between mounts (multi-drop)
        zero_offset_deg : mount angle defined as user 0 deg
        cmd_timeout     : seconds to wait for replies to non-motion commands
        move_timeout    : seconds to wait for home / move / frequency search
        """
        self.addr = _address(address)
        self.zero_offset = _finite(zero_offset_deg, "zero offset")
        self.cmd_timeout = _finite(cmd_timeout, "command timeout")
        self.move_timeout = _finite(move_timeout, "move timeout")
        if min(self.cmd_timeout, self.move_timeout) <= 0:
            raise ValueError("timeouts must be positive")
        if ser is None and (port is None or str(port).lower() == "auto"):
            port = find_ell14(self.addr)
        self.port = port
        self._own = ser is None
        self.ser = ser if ser is not None else _serial_module().Serial(
            port, 9600, bytesize=8, parity="N", stopbits=1, timeout=0.05,
            write_timeout=self.cmd_timeout)
        if self.ser.timeout is None or self.ser.timeout > 0.1:
            if self._own:
                self.close()
            raise ValueError("serial read timeout must be finite and <= 0.1 s")
        try:
            info = self.info()
            if info["type"] != 0x0E:
                raise ElliptecError(f"device is ELL{info['type']}, not ELL14")
            if info["travel"] != 360 or info["pulses_per_unit"] <= 0:
                raise ElliptecError("invalid ELL14 travel/encoder scale")
            self.pulses_per_rev = info["pulses_per_unit"]
            self.serial_no = info["serial"]
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------- comms
    def _readline(self, deadline: float) -> str:
        """Read one CRLF-terminated reply, accumulating partial reads."""
        buf = b""
        while time.monotonic() < deadline:
            buf += self.ser.read_until(b"\n")
            if buf.endswith(b"\n"):
                try:
                    return buf.decode("ascii").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ElliptecError("non-ASCII serial reply") from exc
        raise ElliptecError(f"reply timed out; incomplete bytes: {buf!r}")

    def _query(self, cmd: str, expect: str, timeout: float = None,
               reply_addr=None, allow_busy=False) -> str:
        """
        Send '<addr><cmd>' and wait for a reply with code `expect` ("PO",
        "IN", "GS", ...). Intermediate 'busy' status replies and lines from
        other bus addresses are skipped. Any other non-zero status raises.
        Returns the payload after <addr><code>.
        """
        timeout = self.cmd_timeout if timeout is None else timeout
        reply_addr = self.addr if reply_addr is None else reply_addr
        self.ser.reset_input_buffer()
        request = f"{self.addr}{cmd}".encode("ascii")
        if self.ser.write(request) != len(request):
            raise ElliptecError("incomplete serial command write")
        deadline = time.monotonic() + timeout
        while True:
            line = self._readline(deadline)
            if not line:
                raise ElliptecError(
                    f"no reply to '{self.addr}{cmd}' within {timeout:.1f} s "
                    f"(check power, TX/RX wiring, port, address)")
            if len(line) < 3 or line[0].upper() != reply_addr:
                continue                                  # other device / noise
            code, payload = line[1:3].upper(), line[3:]
            if code == "GS" and not re.fullmatch(r"[0-9a-fA-F]{2}", payload):
                raise ElliptecError(f"malformed status reply: {line!r}")
            if code == "GS" and expect != "GS":
                st = int(payload[:2], 16)
                if st == BUSY or st == 0:
                    continue                              # still moving / ack
                raise ElliptecError(
                    f"'{cmd}': status {st} ({STATUS.get(st, '?')})", st)
            if code == expect:
                if code == "GS":
                    st = int(payload[:2], 16)
                    if st == BUSY and not allow_busy:
                        continue
                    if st not in (0, BUSY) or (st == BUSY and not allow_busy):
                        raise ElliptecError(
                            f"'{cmd}': status {st} ({STATUS.get(st, '?')})", st)
                return payload
            # unexpected code: keep waiting until the deadline

    def _pulses(self, deg: float) -> int:
        return round(_finite(deg, "angle") / 360.0 * self.pulses_per_rev)

    def _deg(self, pulses: int) -> float:
        return pulses * 360.0 / self.pulses_per_rev

    def _po_to_user(self, payload: str) -> float:
        return (self._deg(_from_hex32(payload)) - self.zero_offset) % 360.0

    # ----------------------------------------------------------- info/state
    def info(self) -> dict:
        p = self._query("in", "IN")
        if len(p) != 30 or not re.fullmatch(r"[0-9a-fA-F]{2}[0-9]{12}[0-9a-fA-F]{16}", p):
            raise ElliptecError(f"malformed device information: {p!r}")
        return {
            "type": int(p[0:2], 16),             # 0x0E = 14 for ELL14
            "serial": p[2:10],
            "year": p[10:14],
            "firmware": p[14:16],
            "thread": "imperial" if int(p[16:18], 16) & 0x80 else "metric",
            "hardware": int(p[16:18], 16) & 0x7F,
            "travel": int(p[18:22], 16),         # 360 for rotation mounts
            "pulses_per_unit": int(p[22:30], 16),
        }

    def status(self) -> int:
        return int(self._query("gs", "GS", allow_busy=True), 16)

    def position(self) -> float:
        """Current angle, user frame, in [0, 360)."""
        return self._po_to_user(self._query("gp", "PO"))

    # --------------------------------------------------------------- motion
    def home(self, ccw=False) -> float:
        """Home the mount (needed after power-up). Returns user-frame angle."""
        return self._po_to_user(
            self._query(f"ho{1 if ccw else 0}", "PO", self.move_timeout))

    def move_to(self, angle_deg: float) -> float:
        """Absolute move (user frame). Returns reported final angle."""
        target = (_finite(angle_deg, "angle") + self.zero_offset) % 360.0
        return self._po_to_user(self._query(
            f"ma{_to_hex32(self._pulses(target) % self.pulses_per_rev)}", "PO", self.move_timeout))

    def move_by(self, delta_deg: float) -> float:
        """Relative move. Returns reported final angle (user frame)."""
        return self._po_to_user(self._query(
            f"mr{_to_hex32(self._pulses(delta_deg))}", "PO", self.move_timeout))

    def goto(self, angle_deg: float, tol=0.05, retries=2, settle=0.05) -> float:
        """
        Robust move for experiments: move, verify the landing angle is within
        `tol` deg, retry on mechanical timeout or miss, then wait `settle` s.
        Returns the reported final angle (user frame).
        """
        last_err = None
        angle_deg = _finite(angle_deg, "angle")
        tol, settle = _finite(tol, "tolerance"), _finite(settle, "settle")
        if tol <= 0 or settle < 0 or not isinstance(retries, int) or retries < 0:
            raise ValueError("positive tolerance, nonnegative settle and integer retries required")
        for _ in range(retries + 1):
            try:
                got = self.move_to(angle_deg)
            except ElliptecError as e:
                if e.status not in (2, 11):      # retry only mech/motor faults
                    raise
                last_err = e
                continue
            time.sleep(settle)
            got = self.position()
            if abs(_wrap180(got - angle_deg)) <= tol:
                return got
            last_err = ElliptecError(
                f"landed at {got:.3f} deg, wanted {angle_deg:.3f} (tol {tol})")
        raise ElliptecError(
            f"goto({angle_deg}) failed after {retries + 1} tries: {last_err}. "
            f"Try search_frequency().")

    # ---------------------------------------------------------- maintenance
    def search_frequency(self):
        """Re-tune both piezo motors' resonance. Run if moves get slow or fail;
        then save_user_data() to keep the result after power cycling."""
        self._query("s1", "GS", self.move_timeout)
        self._query("s2", "GS", self.move_timeout)

    def save_user_data(self):
        """Store tuned frequencies / address in device flash."""
        self._query("us", "GS")

    def set_address(self, new_addr: str):
        """Change bus address (0-F), e.g. to put several mounts on one port.
        Call save_user_data() afterwards to make it permanent."""
        new_addr = _address(new_addr)
        self._query(f"ca{new_addr}", "GS", reply_addr=new_addr)
        self.addr = new_addr

    # ---------------------------------------------------------- housekeeping
    def close(self):
        if self._own and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================ port discovery
def _probe(port: str, addresses="0123456789ABCDEF", timeout=0.3) -> dict:
    """Open `port`, send '<a>in' to each address, return {addr: type}."""
    found = {}
    serial = _serial_module()
    try:
        with serial.Serial(port, 9600, timeout=0.05) as s:
            time.sleep(0.05)
            s.reset_input_buffer()
            for a in addresses:
                s.reset_input_buffer()
                s.write(f"{a}in".encode("ascii"))
                deadline, buf = time.monotonic() + timeout, b""
                while time.monotonic() < deadline and not buf.endswith(b"\n"):
                    buf += s.read_until(b"\n")
                line = buf.decode("ascii", "replace").strip()
                if len(line) >= 5 and line[0].upper() == a and line[1:3] == "IN":
                    found[a] = int(line[3:5], 16)
    except (serial.SerialException, OSError, ValueError):
        pass
    return found


def scan_bus(port: str) -> dict:
    """List Elliptec devices on a port: {address: device_type}.
    Useful with the interface board when several mounts share the bus."""
    return _probe(port)


def find_ell14(address="0"):
    """
    Find the serial port with an ELL14 at `address`. Tries FTDI ports first
    (the Thorlabs interface board uses an FTDI USB-serial chip, VID 0x0403),
    then everything else. Returns the port name or raises.
    """
    _serial_module()
    address = _address(address)
    from serial.tools import list_ports
    ports = sorted(list_ports.comports(), key=lambda p: p.vid != 0x0403)
    for p in ports:
        if _probe(p.device, addresses=address.upper()).get(address.upper()) == 0x0E:
            return p.device
    listing = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    raise ElliptecError(
        f"no ELL14 at address {address} found. Ports seen: {listing}. "
        f"Check board power (green LED), USB cable, and that the Thorlabs "
        f"ELLO software is closed (it holds the port).")


# ================================================================== helpers
def acquire_set(rot: ELL14, grab_trace, angles=(0, 45, 90), tol=0.05,
                settle=0.1):
    """
    For each analyzer angle: goto (verified), settle, then call grab_trace()
    (your scope readout, e.g. a pyvisa waveform query; arm/trigger the scope
    inside it so no trace is taken while the piezo is moving).
    Returns {angle: (actual_angle, trace)}.
    """
    out = {}
    for a in angles:
        got = rot.goto(a, tol=tol, settle=settle)
        out[a] = (got, grab_trace())
    return out


def find_polarizer_zero(rot: ELL14, read_power, span=20.0, step=0.5,
                        settle=0.05):
    """
    Locate the analyzer transmission axis with known linear input light and a
    photodiode/power meter (read_power() -> float). Scans +/- span around the
    current angle, fits Malus' law P = A cos^2(theta - t0) + C, and shifts
    rot.zero_offset so that user 0 deg = max transmission.
    Returns (t0_user_frame_before_shift, angles, powers).
    """
    import numpy as np
    span, step = _finite(span, "span"), _finite(step, "step")
    if span <= 0 or step <= 0 or step > span:
        raise ValueError("need span > 0 and 0 < step <= span")
    start = rot.position()
    angles = start + np.arange(-span, span + step / 2, step)
    powers = []
    for a in angles:
        rot.goto(a, settle=settle)
        powers.append(read_power())
    powers = np.asarray(powers, float)
    if not np.isfinite(powers).all():
        raise ValueError("nonfinite optical power")
    th = np.deg2rad(angles)
    M = np.column_stack([np.cos(2 * th), np.sin(2 * th), np.ones_like(th)])
    coef, _, rank, _ = np.linalg.lstsq(M, powers, rcond=None)
    c, s, _ = coef
    if rank != 3 or np.hypot(c, s) <= 1e-10 * max(1.0, np.max(np.abs(powers))):
        raise ValueError("scan does not resolve a transmission axis")
    t0 = np.rad2deg(0.5 * np.arctan2(s, c))
    rot.zero_offset = (rot.zero_offset + t0) % 360.0
    rot.goto(0.0, settle=settle)
    return t0, angles, powers


# ====================================================================== CLI
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="ELL14 rotation mount control")
    ap.add_argument("port", nargs="?", default="auto",
                    help="serial port (COM5, /dev/ttyUSB0) or 'auto' (default)")
    ap.add_argument("--addr", default="0", help="bus address (default 0)")
    ap.add_argument("--scan", action="store_true",
                    help="list serial ports and Elliptec devices on them, then exit")
    ap.add_argument("--zero", type=float, default=0.0,
                    help="mount angle (deg) that maps to user 0 deg")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--no-home", action="store_true")
    args = ap.parse_args()

    if args.scan:
        from serial.tools import list_ports
        names = {0x0E: "ELL14"}
        for p in list_ports.comports():
            devs = scan_bus(p.device)
            desc = ", ".join(f"addr {a}: {names.get(t, f'type {t}')}"
                             for a, t in devs.items()) or "no Elliptec device"
            print(f"{p.device:15s} {p.description[:35]:35s} {desc}")
        return

    with ELL14(args.port, address=args.addr, zero_offset_deg=args.zero) as rot:
        print(f"port {rot.port} | ELL14 S/N {rot.serial_no}: {rot.pulses_per_rev} pulses/rev "
              f"({360e3 / rot.pulses_per_rev:.2f} mdeg/pulse)")
        if not args.no_home:
            print(f"homed -> {rot.home():.3f} deg")
        if not args.interactive:
            for a in (0, 45, 90, 0):
                t = time.perf_counter()
                got = rot.goto(a)
                print(f"goto {a:5.1f} -> {got:8.3f} deg "
                      f"({(time.perf_counter() - t) * 1e3:.0f} ms)")
            return
        print("angle (deg) | h=home | p=position | f=freq search+save | q=quit")
        while True:
            cmd = input("> ").strip().lower()
            try:
                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd == "h":
                    print(f"{rot.home():.3f}")
                elif cmd == "p":
                    print(f"{rot.position():.3f}")
                elif cmd == "f":
                    rot.search_frequency(); rot.save_user_data(); print("done")
                elif cmd:
                    print(f"{rot.goto(float(cmd)):.3f}")
            except (ValueError, ElliptecError) as e:
                print("error:", e)


if __name__ == "__main__":
    _main()

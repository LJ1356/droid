#!/usr/bin/env python3
"""3Dconnexion SpaceMouse reader — dependency-free, PC-invariant.

Reads a SpaceMouse purely from the Linux input layer with the standard library (no hidapi / evdev /
pyspacemouse / ROS / spacenavd install). It:

  * discovers the device by USB VID:PID (3Dconnexion vendor 0x256f) — never a hard-coded path, so it
    works on any PC and survives re-plugs;
  * reads the raw ``/dev/input/eventN`` stream (``struct input_event``): 6 relative axes
    (REL_X/Y/Z/RX/RY/RZ) + the two buttons (BTN_0/BTN_1);
  * exposes a live, normalized 6-DOF + button state from a background thread.

Only requirement: read access to the event device. If your user can't read ``/dev/input/event*``,
add it to the ``input`` group once (no install):  ``sudo usermod -aG input $USER``  then re-log-in.

Validate + calibrate on the teleop PC:

  python spacemouse.py --probe        # live axes/buttons; push/twist the puck, press the buttons
"""

from __future__ import annotations

import glob
import os
import struct
import sys
import threading
import time

VENDOR_3DCONNEXION = 0x256F

# Linux input event types / codes (see linux/input-event-codes.h).
EV_SYN, EV_KEY, EV_REL = 0x00, 0x01, 0x02
# REL_X..REL_RZ are codes 0..5 in order: X, Y, Z, RX, RY, RZ.
_REL_TO_AXIS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BTN_LEFT, BTN_RIGHT = 0x100, 0x101  # the SpaceMouse Compact's two buttons

# struct input_event on 64-bit Linux: struct timeval{long sec; long usec;} + __u16 type,code + __s32 value.
_FMT = "llHHi"
_SZ = struct.calcsize(_FMT)  # 24 on 64-bit

# The device streams the current deflection while the puck is pushed and a 0 on release; this is a
# backstop that zeros an axis if no event arrives for a beat, so a dropped release can't latch motion.
_STALE_SEC = 0.15


def find_event_devices(vendor: int = VENDOR_3DCONNEXION, product: int | None = None) -> list[str]:
    """All ``/dev/input/eventN`` nodes for the given USB vendor (optionally product) that carry the
    6-DOF axes. A SpaceMouse also exposes an LED/keyboard sub-device with no REL axes — filtered out."""
    found = []
    for path in sorted(glob.glob("/sys/class/input/event*")):
        name = os.path.basename(path)
        try:
            v = int(open(f"{path}/device/id/vendor").read().strip(), 16)
            p = int(open(f"{path}/device/id/product").read().strip(), 16)
        except (OSError, ValueError):
            continue
        if v != vendor or (product is not None and p != product):
            continue
        try:
            rel = int((open(f"{path}/device/capabilities/rel").read().strip() or "0"), 16)
        except (OSError, ValueError):
            rel = 0
        if rel:  # a real 6-DOF node exposes relative axes
            found.append(f"/dev/input/{name}")
    return found


class SpaceMouse:
    """Background reader for a 3Dconnexion SpaceMouse. ``get_state()`` returns the current normalized
    6-DOF deflection (``axes`` in [-1, 1]: [x, y, z, rx, ry, rz]) plus the two button states."""

    def __init__(self, device: str | None = None, vendor: int = VENDOR_3DCONNEXION,
                 product: int | None = None, max_deflection: float = 350.0):
        self.max_deflection = float(max_deflection) or 350.0
        self.device = device or next(iter(find_event_devices(vendor, product)), None)
        if self.device is None:
            raise RuntimeError(
                f"No 3Dconnexion SpaceMouse found (USB vendor {vendor:#06x}). Is it plugged in? "
                f"(`lsusb | grep -i 3dconnexion`)"
            )
        try:
            self._f = open(self.device, "rb", buffering=0)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read {self.device}: {e}. Grant read access to the input device — add your "
                f"user to the 'input' group once:  sudo usermod -aG input $USER  then log out/in "
                f"(no package install needed)."
            ) from e

        self._raw = [0] * 6            # latest per-axis value
        self._t = [0.0] * 6            # wall time of the latest per-axis value (for the stale backstop)
        self._buttons = {"left": False, "right": False}
        self._connected = True
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._reader, name="spacemouse", daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop:
            try:
                data = self._f.read(_SZ)
            except (OSError, ValueError):
                with self._lock:
                    self._connected = False
                return
            if not data or len(data) < _SZ:
                continue
            _sec, _usec, etype, code, value = struct.unpack(_FMT, data)
            now = time.time()
            with self._lock:
                if etype == EV_REL and code in _REL_TO_AXIS:
                    i = _REL_TO_AXIS[code]
                    self._raw[i] = value
                    self._t[i] = now
                elif etype == EV_KEY:
                    if code == BTN_LEFT:
                        self._buttons["left"] = bool(value)
                    elif code == BTN_RIGHT:
                        self._buttons["right"] = bool(value)

    def get_state(self) -> dict:
        now = time.time()
        with self._lock:
            axes = [
                max(-1.0, min(1.0, (self._raw[i] if (now - self._t[i]) < _STALE_SEC else 0) / self.max_deflection))
                for i in range(6)
            ]
            return {"axes": axes, "buttons": dict(self._buttons), "connected": self._connected}

    def close(self):
        self._stop = True
        try:
            self._f.close()
        except OSError:
            pass


def _probe():
    devs = find_event_devices()
    print("SpaceMouse event device(s):", devs or "NONE FOUND")
    if not devs:
        sys.exit("No 3Dconnexion device found. Plug it in and check `lsusb | grep -i 3dconnexion`.")
    try:
        sm = SpaceMouse()
    except PermissionError as e:
        sys.exit(str(e))
    print(f"Reading {sm.device} for 15s — push/twist the puck and press the buttons (Ctrl-C to stop)...")
    labels = ["x", "y", "z", "rx", "ry", "rz"]
    end = time.time() + 15
    try:
        while time.time() < end:
            s = sm.get_state()
            bar = " ".join(f"{labels[i]}:{s['axes'][i]:+.2f}" for i in range(6))
            btn = "".join(k[0].upper() if v else "·" for k, v in s["buttons"].items())
            print(f"\r{bar}   buttons[LR]:{btn}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        sm.close()


if __name__ == "__main__":
    if "--probe" in sys.argv:
        _probe()
    else:
        print(__doc__)
        print("event devices:", find_event_devices())

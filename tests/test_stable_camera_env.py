"""The two things that made a failing ZED unreadable in the logs, and unopenable in phase 3.

A policy leg that could not bring its cameras up reported, over and over::

    [ZED][ERROR] CAMERA NOT DETECTED in sl::ERROR_CODE sl::Camera::open(sl::InitParameters)
    FileLocker - /tmp/.zed_enum_lock - #52 File lock timeout.

-- with no serial, no phase and no attempt count anywhere, because the lines carrying all three are
printed by a forked child whose stdout is block-buffered into a pipe. And the lock timeout was the
SDK complaining that two of our own processes were enumerating at once: phase 3 reopens every camera
behind a 0.75 s stagger, against an open that takes 6-14 s on this rig.

No hardware: ``pyzed.sl`` is replaced in the module's namespace and the capture task is driven on
threads, whose overlap is what is being measured anyway.

    python -m pytest droid/tests/test_stable_camera_env.py -q
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from unittest import mock

import numpy as np
import pytest

from droid import stable_camera_env as sce

W, H, CH = 8, 4, 4


class _Enum:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class FakeCamera:
    """Records when each open() call starts and ends, so overlap is measurable."""

    windows: list[tuple[str, float, float]] = []
    lock = threading.Lock()
    fail_serials: set[int] = set()
    open_duration = 0.05

    def __init__(self):
        self._serial = None

    def open(self, params):
        self._serial = params.serial
        start = time.monotonic()
        time.sleep(self.open_duration)  # a real open is seconds; long enough to overlap
        with FakeCamera.lock:
            FakeCamera.windows.append((str(params.serial), start, time.monotonic()))
        if params.serial in FakeCamera.fail_serials:
            return sce.sl.ERROR_CODE.CAMERA_NOT_DETECTED
        return sce.sl.ERROR_CODE.SUCCESS

    def get_camera_information(self):
        cam = mock.Mock(fx=1.0, fy=2.0, cx=3.0, cy=4.0, disto=[0.0] * 5)
        return mock.Mock(camera_configuration=mock.Mock(
            calibration_parameters=mock.Mock(left_cam=cam, right_cam=cam)))

    def grab(self, _runtime):
        return sce.sl.ERROR_CODE.SUCCESS

    def retrieve_image(self, mat, _view):
        mat.data = np.zeros((H, W, CH), np.uint8)

    def close(self):
        pass


class FakeInitParameters:
    def __init__(self):
        self.serial = None
        self.camera_resolution = None
        self.camera_fps = None
        self.depth_mode = None
        self.camera_image_flip = None

    def set_from_serial_number(self, serial):
        self.serial = serial


class FakeMat:
    def __init__(self):
        self.data = np.zeros((H, W, CH), np.uint8)

    def get_data(self):
        return self.data


@pytest.fixture
def fake_sl(monkeypatch):
    FakeCamera.windows = []
    FakeCamera.fail_serials = set()
    FakeCamera.open_duration = 0.05
    sl = mock.Mock()
    sl.Camera = FakeCamera
    sl.InitParameters = FakeInitParameters
    sl.Mat = FakeMat
    sl.RuntimeParameters = mock.Mock
    sl.RESOLUTION = mock.Mock(HD720=_Enum("HD720"), HD1080=_Enum("HD1080"), HD2K=_Enum("HD2K"))
    sl.DEPTH_MODE = mock.Mock(NONE=_Enum("NONE"))
    sl.FLIP_MODE = mock.Mock(OFF=_Enum("OFF"))
    # No CALIBRATION_FILE_NOT_AVAILABLE, so _OK_STATUSES is just SUCCESS -- a real Mock would
    # answer hasattr() with a truthy Mock and quietly widen it.
    sl.ERROR_CODE = _Enum  # placeholder, replaced below
    codes = mock.Mock(spec=["SUCCESS", "CAMERA_NOT_DETECTED"])
    codes.SUCCESS = _Enum("SUCCESS")
    codes.CAMERA_NOT_DETECTED = _Enum("CAMERA NOT DETECTED")
    sl.ERROR_CODE = codes
    monkeypatch.setattr(sce, "sl", sl)
    return sl


def _run_capture(serial: int, open_lock, *, stop_immediately=True):
    """Drive one _capture_task through both open phases and straight out of the grab loop."""
    shms = [SharedMemory(create=True, size=H * W * CH) for _ in range(2)]
    stop = threading.Event()
    if stop_immediately:
        stop.set()
    counter = type("C", (), {"value": 0})()
    try:
        sce._capture_task(
            serial, W, H, 30,
            init_event=threading.Event(), start_event=_set_event(), stop_event=stop,
            frame_event=threading.Event(), frame_counter=counter, frame_lock=threading.Lock(),
            left_shm_name=shms[0].name, right_shm_name=shms[1].name,
            intrinsics_queue=queue.Queue(), resolution_str="720", reopen_delay_sec=0.0,
            open_lock=open_lock,
        )
    finally:
        for shm in shms:
            shm.close()
            shm.unlink()


def _set_event():
    e = threading.Event()
    e.set()
    return e


def _overlaps(windows):
    """Pairs of open() calls whose [start, end] windows intersect."""
    return [
        (a, b)
        for i, (a, a0, a1) in enumerate(windows)
        for (b, b0, b1) in windows[i + 1:]
        if a0 < b1 and b0 < a1
    ]


def test_two_cameras_never_open_at_the_same_instant(fake_sl):
    """What the SDK's `.zed_enum_lock` timeout was reporting: our own concurrent opens."""
    lock = threading.Lock()
    threads = [threading.Thread(target=_run_capture, args=(s, lock)) for s in (34032889, 13222437)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert [t.is_alive() for t in threads] == [False, False]
    # Two cameras x (phase 1 probe + phase 3 reopen).
    assert len(FakeCamera.windows) == 4, FakeCamera.windows
    assert _overlaps(FakeCamera.windows) == [], "opens overlapped despite the shared lock"


def test_without_the_lock_they_do_overlap(fake_sl):
    """The control: the stagger alone does not serialize anything, which is the bug."""
    threads = [threading.Thread(target=_run_capture, args=(s, None)) for s in (34032889, 13222437)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert _overlaps(FakeCamera.windows), "expected the unlocked opens to collide"


def test_a_failing_camera_names_itself_its_phase_and_its_attempt(fake_sl, capfd):
    """The whole point of the buffering fix: CAMERA NOT DETECTED must be attributable."""
    FakeCamera.fail_serials = {34032889}
    FakeCamera.open_duration = 0.0
    # 20 attempts x a 10 s backoff is minutes; only the first attempt's message is needed here.
    with mock.patch.object(sce.time, "sleep"):
        _run_capture(34032889, threading.Lock())
    out = capfd.readouterr().out
    assert "34032889" in out, "the serial the operator has to act on"
    assert "init" in out and "attempt 1/20" in out, "which phase, and how far into the retries"
    assert "CAMERA NOT DETECTED" in out


def test_the_child_switches_its_stdout_to_line_buffered(fake_sl):
    """A forked child's stdout is a block-buffered pipe, so without this every [capture] line above
    sits in an 8K buffer through minutes of retry backoff and never reaches the session log."""
    with mock.patch.object(sys.stdout, "reconfigure") as reconfigure:
        _run_capture(13222437, threading.Lock())
    reconfigure.assert_called_once_with(line_buffering=True)

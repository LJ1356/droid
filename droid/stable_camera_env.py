"""Stable camera wrapper for policy evaluation.

Uses background capture processes (adapted from ai2_robots) to keep ZED cameras
alive during long policy inference pauses. Drop-in replacement for RobotEnv with
the same observation format.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import time
from copy import deepcopy
from multiprocessing import Event, Lock, Process, Queue, Value
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pyzed.sl as sl

from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.misc.parameters import hand_camera_id
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame
from droid.robot_env import RobotEnv

# Resolution constants per ZED mode
_CHANNELS = 4  # BGRA (matches DROID's raw ZED format)
_RES_DIMS = {
    "720": (1280, 720),
    "1080": (1920, 1080),
    "2k": (2208, 1242),
}
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720


def _discover_zed_serials() -> list[str]:
    """Return connected ZED serial numbers as strings."""
    return [str(cam.serial_number) for cam in sl.Camera.get_device_list()]


def _capture_task(
    serial: int,
    width: int,
    height: int,
    frame_rate: int,
    init_event: multiprocessing.synchronize.Event,
    start_event: multiprocessing.synchronize.Event,
    stop_event: multiprocessing.synchronize.Event,
    frame_event: multiprocessing.synchronize.Event,
    frame_counter,
    frame_lock: multiprocessing.synchronize.Lock,
    left_shm_name: str,
    right_shm_name: str,
    intrinsics_queue: Queue,
    resolution_str: str = "720",
    reopen_delay_sec: float = 0.0,
):
    """Background capture process for a single ZED camera (both left and right views).

    Three-phase startup to avoid USB bandwidth exhaustion:
      1. Open camera, extract intrinsics, CLOSE camera, signal init_event
      2. Wait for start_event (set by parent after ALL cameras have been probed)
      3. Reopen camera and enter continuous grab loop

    Closing between phases ensures only one camera is on the USB bus at a time
    during initialization, preventing LOW USB BANDWIDTH errors.

    ``frame_event`` is set once the FIRST grab has been written to shared memory, and
    ``frame_counter`` counts every write after that. Both exist because the shared buffers are
    zero-filled at creation: until a grab lands, a reader gets a pure black image that is
    indistinguishable from a real one, and there is otherwise nothing in the observation to say so.
    """
    _res_enum_map = {
        "720": sl.RESOLUTION.HD720,
        "1080": sl.RESOLUTION.HD1080,
        "2k": sl.RESOLUTION.HD2K,
    }
    cam_resolution = _res_enum_map.get(resolution_str, sl.RESOLUTION.HD720)

    init_params = sl.InitParameters()
    init_params.set_from_serial_number(serial)
    init_params.camera_resolution = cam_resolution
    init_params.camera_fps = frame_rate
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_image_flip = sl.FLIP_MODE.OFF

    # Statuses that mean the camera opened successfully (calibration warning is non-fatal)
    _OK_STATUSES = {sl.ERROR_CODE.SUCCESS}
    if hasattr(sl.ERROR_CODE, "CALIBRATION_FILE_NOT_AVAILABLE"):
        _OK_STATUSES.add(sl.ERROR_CODE.CALIBRATION_FILE_NOT_AVAILABLE)

    def _open_camera(init_params, phase="init"):
        max_attempts = 20
        retry_backoff_sec = 10.0
        zed = sl.Camera()
        status = None
        for attempt in range(max_attempts):
            status = zed.open(init_params)
            if status in _OK_STATUSES or "CALIBRATION" in str(status):
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"[capture] ZED {serial} {phase}: opened with warning: {status} (continuing)")
                return zed
            print(
                f"[capture] ZED {serial} {phase} attempt {attempt + 1}/{max_attempts} "
                f"failed: {status}, retrying in {retry_backoff_sec + attempt}s..."
            )
            zed = sl.Camera()  # re-create after failed open
            time.sleep(retry_backoff_sec + 1.0 * attempt)
        print(f"[capture] Failed to open ZED {serial} ({phase}) after {max_attempts} attempts: {status}")
        return None

    # Phase 1: Open briefly to extract intrinsics, then close to free USB bandwidth
    zed = _open_camera(init_params, phase="init")
    if zed is None:
        return

    calib = zed.get_camera_information().camera_configuration.calibration_parameters

    def _extract_intrinsics(params):
        return {
            "cameraMatrix": np.array(
                [[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]]
            ),
            "distCoeffs": np.array(list(params.disto)),
        }

    intrinsics_queue.put(
        {
            "left": _extract_intrinsics(calib.left_cam),
            "right": _extract_intrinsics(calib.right_cam),
        }
    )

    # Close camera to free USB bandwidth for other cameras to probe
    zed.close()
    print(f"[capture] ZED {serial} probed and closed, waiting for start signal...")
    init_event.set()

    # Phase 2: Wait for parent to signal that ALL cameras have been probed
    start_event.wait()

    # Stagger reopens so multiple processes don't hit USB open() at the same instant
    if reopen_delay_sec > 0:
        time.sleep(reopen_delay_sec)

    # Phase 3: Reopen camera for continuous capture
    zed = _open_camera(init_params, phase="reopen")
    if zed is None:
        return
    print(f"[capture] ZED {serial} reopened, starting continuous grab loop")

    # Map shared memory buffers
    left_shm = SharedMemory(name=left_shm_name)
    right_shm = SharedMemory(name=right_shm_name)
    left_buf = np.ndarray(
        (height, width, _CHANNELS), dtype=np.uint8, buffer=left_shm.buf
    )
    right_buf = np.ndarray(
        (height, width, _CHANNELS), dtype=np.uint8, buffer=right_shm.buf
    )

    left_img = sl.Mat()
    right_img = sl.Mat()
    runtime = sl.RuntimeParameters()
    sleep_until = time.monotonic()

    try:
        while not stop_event.is_set():
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_img, sl.VIEW.LEFT)
                zed.retrieve_image(right_img, sl.VIEW.RIGHT)
                left_data = left_img.get_data()
                right_data = right_img.get_data()

                with frame_lock:
                    left_buf[...] = left_data
                    right_buf[...] = right_data
                    frame_counter.value += 1
                # Only now does the shared memory hold a picture rather than zeros.
                if not frame_event.is_set():
                    frame_event.set()

            curr_time = time.monotonic()
            sleep_until = max(sleep_until + 1.0 / frame_rate, curr_time)
            if curr_time < sleep_until:
                time.sleep(sleep_until - curr_time)
    finally:
        zed.close()
        left_shm.close()
        right_shm.close()


class BackgroundZedCamera:
    """Per-camera manager that spawns a background capture process.

    The capture process opens the camera and signals init_event, but does NOT
    start grabbing until start_event is set (controlled by StableRobotEnv).

    Construction returns as soon as the camera has been PROBED, which is not the same as it
    producing frames -- use `wait_for_first_frame` before reading any.
    """

    def __init__(
        self,
        serial: str,
        start_event: multiprocessing.synchronize.Event,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps: int = 60,
        resolution_str: str = "720",
        reopen_delay_sec: float = 0.0,
    ):
        self.serial = serial
        self.width = width
        self.height = height

        nbytes = height * width * _CHANNELS
        self._left_shm = SharedMemory(create=True, size=nbytes)
        self._right_shm = SharedMemory(create=True, size=nbytes)
        self._left_buf = np.ndarray(
            (height, width, _CHANNELS), dtype=np.uint8, buffer=self._left_shm.buf
        )
        self._right_buf = np.ndarray(
            (height, width, _CHANNELS), dtype=np.uint8, buffer=self._right_shm.buf
        )

        self._stop_event = Event()
        # Set by the capture process when its first grab reaches shared memory; the counter then
        # advances on every grab, so a caller can tell a live camera from one that has stalled.
        self._frame_event = Event()
        self._frame_counter = Value("L", 0)
        self._frame_lock = Lock()
        init_event = Event()
        intrinsics_queue = Queue()

        self._proc = Process(
            target=_capture_task,
            args=(
                int(serial),
                width,
                height,
                fps,
                init_event,
                start_event,
                self._stop_event,
                self._frame_event,
                self._frame_counter,
                self._frame_lock,
                self._left_shm.name,
                self._right_shm.name,
                intrinsics_queue,
                resolution_str,
                reopen_delay_sec,
            ),
            daemon=True,
            name=f"bg_zed_{serial}",
        )
        self._proc.start()

        # Wait for camera to open (up to 60s)
        init_start = time.monotonic()
        while time.monotonic() - init_start < 60:
            if init_event.wait(1):
                break
            if not self._proc.is_alive():
                raise RuntimeError(
                    f"Background ZED capture for {serial} died "
                    f"(exit code {self._proc.exitcode})"
                )
        else:
            self._proc.terminate()
            self._proc.join()
            raise RuntimeError(f"Timeout waiting for ZED {serial} to initialize")

        # Receive intrinsics from capture process
        self._intrinsics = intrinsics_queue.get(timeout=5)

    @property
    def frame_count(self) -> int:
        """Grabs written into shared memory so far. Static across a step means a stalled camera."""
        return int(self._frame_counter.value)

    def wait_for_first_frame(self, timeout: float) -> bool:
        """Block until this camera has written a real frame. False on timeout.

        Nothing may read frames before this returns True. The buffers are zero-filled at creation,
        so `get_frames()` answers a pure black image until the first grab lands -- a valid-looking
        observation that silently blinds whatever consumes it.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self._frame_event.wait(min(0.5, max(0.0, deadline - time.monotonic()))):
                return True
            if not self._proc.is_alive():
                raise RuntimeError(
                    f"Background ZED capture for {self.serial} died before delivering a frame "
                    f"(exit code {self._proc.exitcode})"
                )
            if time.monotonic() >= deadline:
                return False

    def get_frames(self):
        """Return (left_bgra, right_bgra) copies from shared memory."""
        with self._frame_lock:
            left = self._left_buf.copy()
            right = self._right_buf.copy()
        return left, right

    def get_intrinsics(self):
        """Return intrinsics dict keyed by '{serial}_left' and '{serial}_right'."""
        return {
            f"{self.serial}_left": deepcopy(self._intrinsics["left"]),
            f"{self.serial}_right": deepcopy(self._intrinsics["right"]),
        }

    def close(self):
        self._stop_event.set()
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()
        self._left_shm.close()
        self._left_shm.unlink()
        self._right_shm.close()
        self._right_shm.unlink()

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.is_alive():
            self._stop_event.set()
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
        for attr in ("_left_shm", "_right_shm"):
            if hasattr(self, attr):
                try:
                    shm = getattr(self, attr)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass


class StableRobotEnv:
    """Drop-in replacement for RobotEnv with background camera capture.

    Prevents ZED camera disconnects by continuously grabbing frames in
    dedicated background processes, rather than grabbing on-demand.
    """

    def __init__(
        self,
        action_space: str = "cartesian_velocity",
        gripper_action_space: str | None = None,
        camera_serials: list[str] | None = None,
        frame_rate: int = 30,
        camera_resolutions: dict[str, str] | None = None,
        do_reset: bool = True,
        first_frame_timeout: float = 120.0,
    ):
        """
        Args:
            camera_resolutions: Optional dict mapping serial -> resolution string
                ("720", "1080", "2k"). Cameras not in the dict use "720".
            do_reset: forwarded to RobotEnv. Set False to skip the blocking move to
                reset_joints on construction, so the arm stays wherever it already is
                (e.g. handed off mid-pose from another controller) instead of being
                yanked to the fixed reset pose before teleop starts.
            first_frame_timeout: how long to wait, after the grab loops start, for EVERY camera to
                deliver its first frame. Construction raises rather than return an env whose
                observations are still the zero-filled buffers. Generous by default because phase 3
                reopens each ZED and retries with a 10 s backoff when one is still being released
                by whatever held it last.
        """
        if camera_serials is None:
            camera_serials = _discover_zed_serials()
        elif len(camera_serials) == 0:
            camera_serials = _discover_zed_serials()
        if camera_resolutions is None:
            camera_resolutions = {}

        # Robot-only env (skip DROID's camera system entirely)
        self._env = RobotEnv(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            camera_kwargs={"skip_cameras": True},
            do_reset=do_reset,
        )

        # Shared event: gates the grab loop for ALL cameras.
        # Cameras open sequentially without grabbing, then all start together.
        self._start_event = Event()

        # Open background cameras with staggered opens (no grabbing yet)
        self._cameras: dict[str, BackgroundZedCamera] = {}
        for i, serial in enumerate(camera_serials):
            if not serial:
                continue
            res_str = camera_resolutions.get(serial, "720")
            cam_w, cam_h = _RES_DIMS.get(res_str, (1280, 720))
            # Use lower FPS for high-res modes to stay within USB bandwidth
            cam_fps = 15 if res_str == "2k" else frame_rate
            # Many ZEDs on shared USB controllers: cap FPS to avoid LOW USB BANDWIDTH
            if len(camera_serials) >= 3 and res_str != "2k":
                cam_fps = min(cam_fps, 15)
            print(f"Opening background ZED camera {serial} at {res_str} ({cam_w}x{cam_h} @ {cam_fps}fps)...")
            self._cameras[serial] = BackgroundZedCamera(
                serial,
                self._start_event,
                width=cam_w,
                height=cam_h,
                fps=cam_fps,
                resolution_str=res_str,
                reopen_delay_sec=0.75 * i,
            )
            if i < len(camera_serials) - 1:
                time.sleep(2.0)

        # All cameras opened — now let them all start grabbing
        print(f"All {len(self._cameras)} cameras opened, starting grab loops...")
        self._start_event.set()

        # Block until every camera has actually written a frame.
        #
        # The shared buffers are zero-filled at creation, so until the first grab lands
        # `get_observation()` hands back a PURE BLACK image that no consumer can distinguish from a
        # real one -- no exception, no missing key, nothing to check. A `time.sleep(0.5)` stood here
        # and was never close to enough: phase 3 REOPENS each ZED (staggered, and retried after a
        # 10 s backoff when the camera is still being released by whatever held it last), which
        # routinely takes tens of seconds. A policy started in that window runs blind on black
        # frames and the leg looks like the policy simply did nothing.
        self._wait_for_first_frames(first_frame_timeout)

        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Build camera_reader shim so DataCollecter/GUI can query camera IDs
        cam_stubs = {}
        for serial, cam in self._cameras.items():
            raw_intr = cam.get_intrinsics()  # {serial_left: {...}, serial_right: {...}}
            cam_stubs[serial] = self._CameraStub(serial, raw_intr)
        self._camera_reader_shim = self._CameraReaderShim(cam_stubs)

    def _wait_for_first_frames(self, timeout: float) -> None:
        """Wait for every background camera to deliver its first real frame, or raise."""
        started = time.monotonic()
        for serial, cam in self._cameras.items():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or not cam.wait_for_first_frame(remaining):
                raise RuntimeError(
                    f"ZED {serial} delivered no frame within {timeout:.0f}s of the grab loops "
                    "starting, so its buffer is still black. Check the camera is connected and not "
                    "still held by another process (a tiptop run or another capture)."
                )
        print(
            f"All {len(self._cameras)} cameras delivering frames "
            f"({time.monotonic() - started:.1f}s after the grab loops started)"
        )

    def camera_frame_counts(self) -> dict[str, int]:
        """Grabs written per camera serial. A count that does not advance is a stalled camera.

        The frames themselves cannot answer this: a stalled camera keeps returning its last good
        image, which looks entirely valid.
        """
        return {serial: cam.frame_count for serial, cam in self._cameras.items()}

    # --- Delegated methods (same interface as RobotEnv) ---

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def control_hz(self):
        return self._env.control_hz

    @property
    def _robot(self):
        return self._env._robot

    def step(self, action):
        return self._env.step(action)

    def reset(self, randomize=False):
        return self._env.reset(randomize=randomize)

    def get_state(self):
        return self._env.get_state()

    def update_robot(self, action, **kwargs):
        return self._env.update_robot(action, **kwargs)

    def create_action_dict(self, action):
        return self._env.create_action_dict(action)

    # --- camera_reader compatibility (for DataCollecter / GUI) ---

    @property
    def camera_reader(self):
        """Expose a minimal object matching MultiCameraWrapper's interface."""
        return self._camera_reader_shim

    class _CameraStub:
        """Minimal stand-in for a ZedCamera, enough for calibration."""

        def __init__(self, serial, intrinsics):
            self.serial_number = serial
            self.latency = 0
            self.high_res_calibration = False
            self.current_mode = "trajectory"
            self._intrinsics = intrinsics

        def get_intrinsics(self):
            return deepcopy(self._intrinsics)

        def is_running(self):
            return True

    class _CameraReaderShim:
        """Lightweight stand-in for MultiCameraWrapper used by DataCollecter."""

        def __init__(self, camera_dict):
            self.camera_dict = camera_dict

        def get_camera(self, camera_id):
            return self.camera_dict[camera_id]

        def enable_advanced_calibration(self):
            pass

        def disable_advanced_calibration(self):
            pass

        def set_calibration_mode(self, cam_id):
            pass

        def set_trajectory_mode(self, traj_params_override=None):
            pass

        def start_recording(self, recording_folderpath):
            pass

        def stop_recording(self):
            pass

    def read_cameras(self):
        """Return (obs_dict, timestamp_dict) matching MultiCameraWrapper.read_cameras()."""
        from collections import defaultdict

        full_obs_dict = defaultdict(dict)
        read_start = time_ms()
        for serial, cam in self._cameras.items():
            left, right = cam.get_frames()
            full_obs_dict["image"][f"{serial}_left"] = left
            full_obs_dict["image"][f"{serial}_right"] = right
        read_end = time_ms()
        timestamp_dict = {"read_start": read_start, "read_end": read_end}
        return full_obs_dict, timestamp_dict

    # --- Camera methods ---

    def get_camera_extrinsics(self, state_dict):
        """Same logic as RobotEnv.get_camera_extrinsics."""
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
        return extrinsics

    def get_observation(self):
        """Return observation dict in the same format as RobotEnv.get_observation().

        Keys: robot_state, timestamp, image, camera_type, camera_extrinsics,
        camera_intrinsics.  Image values are BGRA numpy arrays (H x W x 4).
        """
        obs_dict = {"timestamp": {}}

        # Robot state
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera readings from background processes (instant, never blocks on ZED SDK)
        camera_timestamp = {}
        image_dict = {}
        read_start = time_ms()
        for serial, cam in self._cameras.items():
            left, right = cam.get_frames()
            image_dict[f"{serial}_left"] = left
            image_dict[f"{serial}_right"] = right
        camera_timestamp["read_start"] = read_start
        camera_timestamp["read_end"] = time_ms()
        obs_dict["image"] = image_dict
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera info
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        obs_dict["camera_extrinsics"] = self.get_camera_extrinsics(state_dict)

        intrinsics = {}
        for cam in self._cameras.values():
            cam_intr = cam.get_intrinsics()
            for full_cam_id, info in cam_intr.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict

    def close(self):
        for cam in self._cameras.values():
            cam.close()
        self._cameras.clear()

# ruff: noqa
"""SpaceMouse teleoperation + capture driver for the data-collection app's teleop flow.

Replaces the VR-headset teleop (``scripts/main.py`` + the Tk GUI) with a **SpaceMouse** controller,
driven from the browser exactly like the tamp/eval capture drivers (events-file + stdin protocol, see
``data-collection/ARCHITECTURE.md`` §6). The operator moves the end-effector by pushing/twisting the
SpaceMouse puck (6-DOF -> Cartesian velocity) and works the gripper with its two buttons; the operator
decides when each episode ends (from the UI). Every episode is written in the raw episode format (§3)
so ``collect/build_lerobot.py`` builds it exactly like a tamp episode.

Nothing is installed on the NUC — this is a drop-in for the existing PC-side teleop (still talks to the
NUC's ``run_server.py`` over the same ``StableRobotEnv`` -> ServerInterface path); only the controller
changes (SpaceMouse instead of VR). The SpaceMouse is read dependency-free (see ``spacemouse.py``).

Protocol (stdin lines written by the Node server):
  {"cmd":"start"}   begin an episode (at the task prompt)
  {"cmd":"end"}     stop + SAVE the current episode        {"cmd":"discard"} stop + throw it away
  y | n             label the saved episode success/failure   q  finish the session

Events (appended to $TELEOP_EVENTS_FILE): session_start, awaiting_task, rollout_start, rollout_saved,
awaiting_label, labeled, rollout_aborted, session_end.

Run under the DROID conda env (same as the VR ``scripts/main.py``).
"""

import dataclasses
import datetime
import json
import os
import select
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Shared raw-episode helpers (same directory; import triggers no robot/policy code).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_capture import emit, write_meta, write_robot_state_npz, _write_video  # noqa: E402
from spacemouse import SpaceMouse  # noqa: E402

CONTROL_HZ = 15  # matches StableRobotEnv.control_hz + the LeRobot build FPS
EXTERNAL_CAM, EXTERNAL_CAM_2, HAND_CAM = "external_cam.mp4", "external_cam_2.mp4", "hand_cam.mp4"

# SIGINT (force-stop, sent to the DRIVER pid only) discards the in-flight episode and halts the arm,
# keeping the session warm (the ZED background processes survive because the signal isn't sent to the
# group). SIGTERM (graceful-stop escalation) exits cleanly so `finally` halts + closes the env.
_ABORT_EP = {"v": False}


def _install_signal_handlers():
    signal.signal(signal.SIGINT, lambda *_: _ABORT_EP.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# --------------------------------------------------------------------------- #
# stdin line protocol (shared buffer across blocking + non-blocking reads)      #
# --------------------------------------------------------------------------- #
_buf = ""


def _parse(line: str):
    line = line.strip()
    if not line:
        return {}
    if line in ("q", "y", "n"):
        return line
    try:
        return json.loads(line)
    except ValueError:
        return {}


def _read_line_blocking():
    """Block until one full stdin line, returning the parsed command (or 'q' at EOF)."""
    global _buf
    while "\n" not in _buf:
        chunk = os.read(0, 4096)
        if not chunk:
            return "q"
        _buf += chunk.decode(errors="ignore")
    line, _buf = _buf.split("\n", 1)
    return _parse(line)


def _poll_line():
    """Return the next buffered stdin command without blocking, else None ('q' at EOF)."""
    global _buf
    if "\n" not in _buf:
        r, _, _ = select.select([0], [], [], 0)
        if r:
            chunk = os.read(0, 4096)
            if not chunk:
                return "q" if not _buf else None
            _buf += chunk.decode(errors="ignore")
    if "\n" not in _buf:
        return None
    line, _buf = _buf.split("\n", 1)
    return _parse(line)


# --------------------------------------------------------------------------- #
# SpaceMouse -> Cartesian-velocity + gripper                                    #
# --------------------------------------------------------------------------- #
class SpaceMousePolicy:
    """Maps the SpaceMouse 6-DOF deflection to a DROID Cartesian-velocity action
    ``[vx,vy,vz, wx,wy,wz, gripper_vel]`` in [-1,1]. Buttons set the gripper target (left=open,
    right=close); the gripper is velocity-driven toward it. Axis order/signs are configurable so the
    puck can be aligned to the robot base frame without a code edit."""

    def __init__(self, sm: SpaceMouse, *, pos_gain, rot_gain, gripper_gain, deadzone, axis_map, axis_sign):
        self.sm = sm
        self.pos_gain, self.rot_gain, self.gripper_gain = pos_gain, rot_gain, gripper_gain
        self.deadzone = deadzone
        self.axis_map = axis_map      # length-6: which SpaceMouse axis feeds robot DOF i
        self.axis_sign = axis_sign    # length-6: +1/-1 per robot DOF
        self.gripper_target = 0.0     # 0 open .. 1 closed; starts open
        self._prev = {"left": False, "right": False}

    def _dz(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def forward(self, obs):
        st = self.sm.get_state()
        raw = st["axes"]
        a = [self.axis_sign[i] * self._dz(raw[self.axis_map[i]]) for i in range(6)]
        lin = [a[0] * self.pos_gain, a[1] * self.pos_gain, a[2] * self.pos_gain]
        rot = [a[3] * self.rot_gain, a[4] * self.rot_gain, a[5] * self.rot_gain]

        b = st["buttons"]
        if b["left"] and not self._prev["left"]:
            self.gripper_target = 0.0  # open
        if b["right"] and not self._prev["right"]:
            self.gripper_target = 1.0  # close
        self._prev = dict(b)

        grip_meas = float(np.asarray(obs["robot_state"]["gripper_position"]).reshape(-1)[0])
        grip_vel = float(np.clip((self.gripper_target - grip_meas) * self.gripper_gain, -1.0, 1.0))
        action = np.clip(np.asarray(lin + rot + [grip_vel], dtype=np.float64), -1.0, 1.0)
        return action, self.gripper_target


# --------------------------------------------------------------------------- #
# Camera extraction (StableRobotEnv images are BGRA, keyed by "{serial}_left")   #
# --------------------------------------------------------------------------- #
def _rgb(img):
    return None if img is None else np.ascontiguousarray(img[..., :3][..., ::-1])  # BGRA -> RGB


def _extract_cameras(obs, ext_id, ext2_id, hand_id):
    """Return (external, external_2, wrist) RGB frames by matching configured ZED serials."""
    images = obs.get("image") or {}
    if not images:
        raise RuntimeError(
            "No camera frames from the DROID env — the ZED cameras were not read. Check they are "
            "connected and NOT already open in another process (a running tiptop-run / capture holds "
            "them exclusively), and that TIPTOP_EXTERNAL_CAMERA_ID / TIPTOP_HAND_CAMERA_ID are set."
        )

    def pick(serial):
        if not serial:
            return None
        for key, val in images.items():
            if serial in key and "left" in key:
                return _rgb(val)
        return None

    return pick(ext_id), pick(ext2_id), pick(hand_id)


# --------------------------------------------------------------------------- #
# One teleop episode                                                            #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Args:
    events_file: str = ""
    output_root: str = ""       # runs/<workspace>/teleop/<name>; episodes nest under eval/<ts> (staging)
    instruction: str = ""
    config_id: str = "teleop/teleop"
    max_deflection: float = 350.0
    pos_gain: float = 1.0
    rot_gain: float = 1.0
    gripper_gain: float = 3.0
    deadzone: float = 0.06
    axis_map: str = "0,1,2,3,4,5"    # SpaceMouse axis feeding robot DOF [x,y,z,rx,ry,rz]
    axis_sign: str = "1,1,1,1,1,1"   # sign per robot DOF (tune to align the puck to the base frame)
    external_camera_id: str = ""
    external_2_camera_id: str = ""
    hand_camera_id: str = ""


def _halt(env):
    """Command zero Cartesian velocity so the arm stops (velocity control holds the last command)."""
    try:
        env.step(np.zeros(7, dtype=np.float64))
    except Exception as e:  # noqa: BLE001
        print(f"[teleop] halt failed: {e}", flush=True)


def record_episode(env, policy, ep_dir, args, events):
    """Teleop until the operator ends/discards it. Returns ``(n_frames_or_None, quit)`` — n_frames on
    save (None if discarded), and ``quit`` True when the operator asked to finish the whole session."""
    ext_frames, ext2_frames, wrist_frames = [], [], []
    joint_log, grip_log, ft_log, cmd_g_log = [], [], [], []
    _ABORT_EP["v"] = False  # clear any SIGINT that arrived while parked at the prompt
    ended, quit_session = None, False
    try:
        while True:
            cmd = _poll_line()
            if cmd == "q":                 # finish the session (discards the in-flight episode)
                ended, quit_session = "discard", True
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "discard":
                ended = "discard"
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "end":
                ended = "end"
                break
            if _ABORT_EP["v"]:             # force-stop (SIGINT): discard this episode, stay warm
                ended = "discard"
                break

            t0 = time.time()
            obs = env.get_observation()
            ext, ext2, wrist = _extract_cameras(obs, args.external_camera_id, args.external_2_camera_id, args.hand_camera_id)
            if ext is None or wrist is None:
                raise RuntimeError("external and/or wrist camera returned no frame (check the serial env vars)")
            ext_frames.append(ext)
            wrist_frames.append(wrist)
            ext2_frames.append(ext2)

            rs = obs["robot_state"]
            joint_log.append(np.asarray(rs["joint_positions"], dtype=np.float32).reshape(-1))
            grip_log.append(float(np.asarray(rs["gripper_position"]).reshape(-1)[0]))
            ft_log.append(time.time())

            action, gtarget = policy.forward(obs)
            cmd_g_log.append(1.0 if gtarget > 0.5 else 0.0)
            env.step(action)

            dt = time.time() - t0
            if dt < 1.0 / CONTROL_HZ:
                time.sleep(1.0 / CONTROL_HZ - dt)
    finally:
        _halt(env)  # always stop the arm when recording ends (end / discard / error / stop)

    n = min(len(ext_frames), len(wrist_frames), len(joint_log), len(cmd_g_log))
    if ended == "discard" or n < 2:
        return None, quit_session

    ep_dir = Path(ep_dir)
    _write_video(ext_frames[:n], ep_dir / EXTERNAL_CAM)
    _write_video(wrist_frames[:n], ep_dir / HAND_CAM)
    cameras = {"exterior_image_1_left": EXTERNAL_CAM, "wrist_image_left": HAND_CAM}
    if all(f is not None for f in ext2_frames[:n]):
        _write_video(ext2_frames[:n], ep_dir / EXTERNAL_CAM_2)
        cameras["exterior_image_2_left"] = EXTERNAL_CAM_2

    jp = np.stack(joint_log[:n])                                  # measured joints [n,7]
    frame_time = np.asarray(ft_log[:n], dtype=np.float64)
    # The arm is Cartesian-velocity controlled, so there is no commanded joint velocity. Derive the
    # ACHIEVED joint motion from the demonstration (a valid BC target, and honest — unlike the old
    # teleop stub): cmd_joint_velocity[t] = (q[t+1]-q[t])*fps, cmd_joint_position[t] = q[t+1].
    cmd_jv = np.zeros((n, 7), dtype=np.float32)
    cmd_jv[:-1] = ((jp[1:] - jp[:-1]) * CONTROL_HZ).astype(np.float32)
    cmd_jp = np.vstack([jp[1:], jp[-1:]]).astype(np.float32)
    write_robot_state_npz(
        ep_dir / "robot_state.npz",
        joint_position=jp,
        gripper_position=np.asarray(grip_log[:n], dtype=np.float32),
        cmd_joint_position=cmd_jp,
        cmd_joint_velocity=cmd_jv,
        cmd_gripper=np.asarray(cmd_g_log[:n], dtype=np.float32),  # binary target (0 open / 1 closed)
        frame_time=frame_time,
    )
    write_meta(
        ep_dir / "_meta.json",
        instruction=args.instruction, n_frames=n, config_id=args.config_id,
        timestamp=ep_dir.name, cameras=cameras,
        record_start=float(frame_time[0]), record_stop=float(frame_time[-1]),
    )
    return n, quit_session


def main(args: Args):
    events = args.events_file
    args.external_camera_id = args.external_camera_id or os.environ.get("TIPTOP_EXTERNAL_CAMERA_ID", "")
    args.external_2_camera_id = args.external_2_camera_id or os.environ.get("TIPTOP_EXTERNAL_2_CAMERA_ID", "")
    args.hand_camera_id = args.hand_camera_id or os.environ.get("TIPTOP_HAND_CAMERA_ID", "")
    axis_map = [int(x) for x in str(args.axis_map).split(",")]
    axis_sign = [float(x) for x in str(args.axis_sign).split(",")]
    output_root = Path(args.output_root)
    (output_root / "eval").mkdir(parents=True, exist_ok=True)
    _install_signal_handlers()

    emit(events, "session_start")
    env = None
    sm = None
    try:
        sm = SpaceMouse(max_deflection=args.max_deflection)  # raises (no device / no perms) -> error exit
        print(f"[teleop] SpaceMouse on {sm.device}", flush=True)
        policy = SpaceMousePolicy(
            sm, pos_gain=args.pos_gain, rot_gain=args.rot_gain, gripper_gain=args.gripper_gain,
            deadzone=args.deadzone, axis_map=axis_map, axis_sign=axis_sign,
        )
        from droid.stable_camera_env import StableRobotEnv  # lazy: DROID env only
        env = StableRobotEnv(action_space="cartesian_velocity", gripper_action_space=None)
        print("[teleop] created the DROID env", flush=True)

        first = True
        while True:
            emit(events, "awaiting_task")
            cmd = _read_line_blocking()
            if cmd == "q":
                break
            if not (isinstance(cmd, dict) and cmd.get("cmd") == "start"):
                continue

            if not first:
                try:
                    env.reset()
                except Exception as e:  # noqa: BLE001
                    print(f"[teleop] env.reset() failed: {e}", flush=True)
            first = False

            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ep_dir = output_root / "eval" / ts   # staging bucket; the label moves it to success/failure
            ep_dir.mkdir(parents=True, exist_ok=True)
            emit(events, "rollout_start", dir=str(ep_dir), instruction=args.instruction)
            try:
                n, quit_session = record_episode(env, policy, ep_dir, args, events)
            except Exception as e:  # noqa: BLE001
                _halt(env)
                shutil.rmtree(ep_dir, ignore_errors=True)
                emit(events, "error", message=f"episode failed: {e}")
                continue
            if n is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                emit(events, "rollout_aborted", dir=str(ep_dir))
                if quit_session:
                    break
                continue
            emit(events, "rollout_saved", dir=str(ep_dir), n_frames=n)

            # label -> move the staged episode into success/ or failure/
            emit(events, "awaiting_label", dir=str(ep_dir))
            lab = _read_line_blocking()
            if lab == "q":
                break
            success = (lab == "y") or (isinstance(lab, dict) and bool(lab.get("success")))
            status = "success" if success else "failure"
            dest = output_root / status / ts
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(ep_dir), str(dest))
            except OSError as e:
                print(f"[teleop] could not move episode to {status}: {e}", flush=True)
                dest = ep_dir
            emit(events, "labeled", dir=str(dest), success=success)
    finally:
        emit(events, "session_end")
        if env is not None:
            _halt(env)
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        if sm is not None:
            sm.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))

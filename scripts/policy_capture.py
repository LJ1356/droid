# ruff: noqa
"""Drive ONE human-in-the-loop phase with a trained policy instead of a teleoperator.

A HITL-TAMP task runs as ``tamp -> teleop -> tamp``: cuTAMP carries out the phases it can express and
a person carries out the ones it cannot ("fold the cloth", "open the box"). This driver is that
middle leg with the person replaced by a policy trained on the person -- a LeRobot ``DiffusionPolicy``
behaviour-cloned on the teleop legs of earlier runs of the same task
(``hitl-baseline/diffusion_policy``). It is spawned by ``tiptop_run._run_policy_phase`` once the arm
and the ZEDs have been released, runs closed-loop for a fixed number of control steps, writes the leg
in the raw episode format (``data-collection/ARCHITECTURE.md`` §3) and exits, handing the hardware
back exactly as a teleop session does.

    python scripts/policy_capture.py \\
        --output-root runs/<ws>/tamp/<name> --instruction "fold the cloth" \\
        --checkpoint .../checkpoints/1_toy_puzzle/checkpoints/last \\
        --trajectory-id <16 hex> --result-file /tmp/leg.json

**Two processes, and why.** The arm and the cameras need ``droid`` and ``pyzed``, which are in the
DROID conda env; the policy needs LeRobot and torch, which are in the diffusion project's venv and
not here. So this driver starts ``python -m hitl_dp.serve`` under that venv and talks to it over a
loopback socket (``hitl_dp.wire``, imported by path). Same split as ``eval_capture.py`` makes for
openpi, for the same reason.

**The leg ends on a step count, not on success.** A behaviour-cloning policy has no termination
signal -- no reward, no done head -- so nothing in it knows the phase is finished. ``--max-steps``
ends the leg and the VLM phase verification that tiptop runs afterwards is what decides whether what
it did counts. That check, the phase's atoms and the operator's label are all unchanged from the
teleop path.

**It reports nothing on the events file.** The trajectory belongs to tiptop, which is still the
session's driver while this runs; it emits ``policy_phase_start`` / ``policy_phase_done`` around this
process, and what this one writes is progress on stdout (streamed into the session log) plus a small
JSON ``--result-file`` for tiptop to read. Two writers on one events file is a race nobody needs.

Run under the DROID conda env (same as ``teleop_capture.py`` / ``eval_capture.py``).
"""

import argparse
import datetime
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Shared raw-episode helpers, and the camera-serial matching the teleop leg already uses. Same
# directory; importing these triggers no robot or policy code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_capture import write_meta, write_robot_state_npz, _write_video  # noqa: E402
from teleop_capture import _extract_cameras  # noqa: E402

CONTROL_HZ = 15  # StableRobotEnv.control_hz, the capture frequency, and the LeRobot build FPS
EXTERNAL_CAM, EXTERNAL_CAM_2, HAND_CAM = "external_cam.mp4", "external_cam_2.mp4", "hand_cam.mp4"

# What a policy leg is called in `_meta.json` and in the merged episode's `segments/NN_<source>_<ts>`
# directory. A third value beside "tamp" and "teleop", not a reuse of either: the action arrays on
# this leg are a policy's, and `collect/droid_action.py` resolves each leg in its own provenance.
SEGMENT_SOURCE = "policy"

# How long the policy server gets to import torch, load the checkpoint and bind its socket. Measured
# at ~2 s warm; the budget is for a cold page cache and a GPU another process is still releasing.
SERVER_READY_TIMEOUT_S = 300.0

# How long to wait for both cameras to return an actual PICTURE before giving up on the leg.
# `StableRobotEnv` already blocks until each camera has written its first frame; this is the second,
# independent check, and it catches what that one cannot -- a camera that is grabbing happily and
# returning black anyway (mid auto-exposure, a covered lens).
CAMERA_WARMUP_TIMEOUT_S = 30.0

# A frame whose pixel standard deviation is at or below this carries no picture. Real frames from
# these ZEDs measure ~50-65; a zero-filled shared-memory buffer measures exactly 0.
BLANK_FRAME_MAX_STD = 1.0

# Consecutive control steps a camera may go without a new grab before it is called out. At 15 Hz
# this is one second, and the capture loop runs at 15-30 fps, so a healthy camera never reaches it.
STALE_FRAME_WARN_STEPS = 15

# A force-stop (SIGINT) stops stepping the policy at the next step and still WRITES what was
# captured: the leg is part of a trajectory the operator will label, and throwing it away would take
# the tamp legs before it with it. SIGTERM exits so `finally` halts the arm and closes the env.
_ABORT = {"v": False}


def _install_signal_handlers():
    signal.signal(signal.SIGINT, lambda *_: _ABORT.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def log(msg: str) -> None:
    print(f"[policy] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# The policy server (diffusion venv) this driver talks to                       #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    """An unused local TCP port, picked right before the server binds it (mirrors eval_capture)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_policy_server(python: str, policy_dir: str, checkpoint: str, port: int, open_loop_horizon: int,
                        device: str, num_inference_steps: int = 0):
    """Spawn ``python -m hitl_dp.serve`` and block until it prints READY. Returns the process.

    Waiting on the READY line rather than polling the port is deliberate: the policy takes seconds to
    load and has no socket at all until it is loaded, so a connect-retry loop cannot tell "still
    loading" from "died on the way up" and would spend the whole timeout on a checkpoint that failed
    to open. The server's own log is pumped to our stderr behind a prefix so it lands in the session
    log next to everything else.
    """
    cmd = [python, "-m", "hitl_dp.serve", "--checkpoint", checkpoint, "--host", "127.0.0.1",
           "--port", str(port), "--open-loop-horizon", str(open_loop_horizon), "--device", device]
    if num_inference_steps:
        cmd += ["--num-inference-steps", str(num_inference_steps)]
    log(f"starting the policy server: {' '.join(cmd)}")
    env = dict(os.environ)
    # The checkpoint is on disk and every weight it needs is in it; a Hub lookup here can only turn a
    # local mistake into a network hang (the same reason hitl_dp.train sets it).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(policy_dir) / "src"), env.get("PYTHONPATH")]))
    proc = subprocess.Popen(cmd, cwd=policy_dir, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    ready = threading.Event()
    def _pump():
        for line in proc.stdout:
            sys.stderr.write("[policy-server] " + line)
            sys.stderr.flush()
            if line.startswith("READY "):
                ready.set()
    threading.Thread(target=_pump, daemon=True).start()

    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while not ready.wait(0.5):
        if proc.poll() is not None:
            raise RuntimeError(f"the policy server exited with code {proc.returncode} before it was ready")
        if time.time() > deadline:
            proc.terminate()
            raise TimeoutError(f"the policy server did not come up within {SERVER_READY_TIMEOUT_S:.0f}s")
    return proc


def stop_policy_server(proc) -> None:
    """Ask the server to exit, then insist. It holds a GPU allocation the next leg wants back."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log("the policy server did not exit on SIGTERM; killing it")
        proc.kill()
        proc.wait(timeout=5)


# --------------------------------------------------------------------------- #
# One policy leg                                                                #
# --------------------------------------------------------------------------- #
def _frame_is_blank(frame) -> bool:
    """No picture: the zero-filled shared buffer, or any other uniform field."""
    return frame is None or float(np.asarray(frame, dtype=np.float32).std()) <= BLANK_FRAME_MAX_STD


def wait_for_live_cameras(env, args, timeout: float = CAMERA_WARMUP_TIMEOUT_S) -> None:
    """Block until BOTH cameras the policy consumes return a real picture. Raise on timeout.

    This is the check whose absence ruined the 2026-09-09 toy-puzzle leg. The ZEDs had only just
    been released by tiptop, `StableRobotEnv` returned while its shared buffers were still
    zero-filled, and the first 169 of 450 steps -- 38% of the rollout -- were inferred from a pure
    black exterior image. Nothing anywhere said so: the policy still answered with an action, the
    mp4 was still written, and the leg simply read as a policy that had chosen not to move.
    """
    deadline = time.time() + timeout
    blank = ["external", "wrist"]
    while time.time() < deadline:
        obs = env.get_observation()
        ext, _, wrist = _extract_cameras(
            obs, args.external_camera_id, args.external_2_camera_id, args.hand_camera_id
        )
        blank = [n for n, f in (("external", ext), ("wrist", wrist)) if _frame_is_blank(f)]
        if not blank:
            return
        log(f"waiting for a picture from the {' and '.join(blank)} camera")
        time.sleep(0.5)
    raise RuntimeError(
        f"the {' and '.join(blank)} camera returned a blank frame for {timeout:.0f}s. Refusing to "
        "start the leg: the policy would be driven on a black image. Check the ZED is connected, "
        "uncovered, and not still held by another process."
    )


def warm_up_policy(env, sock, args) -> None:
    """One throwaway inference, before the leg starts.

    The first call through the whole path -- socket, resize, cuDNN autotune, the U-Net's first
    launch -- measures ~269 ms against a 66 ms control period; every one after it is ~52 ms. Paying
    that inside the leg makes step 0 a four-period stall on the one step where the policy is deciding
    what to do with the pose TAMP just handed it. It also proves the wire, the checkpoint and the
    cameras work together before a single frame is recorded, rather than on the first recorded one.

    `run_leg` sends `reset` before its first step, so the observation this leaves in the policy's
    history is discarded rather than carried into the leg.
    """
    from hitl_dp.wire import request

    obs = env.get_observation()
    ext, _, wrist = _extract_cameras(
        obs, args.external_camera_id, args.external_2_camera_id, args.hand_camera_id
    )
    rs = obs["robot_state"]
    state = np.concatenate([
        np.asarray(rs["joint_positions"], dtype=np.float32).reshape(-1),
        [float(np.asarray(rs["gripper_position"]).reshape(-1)[0])],
    ]).astype(np.float32)
    t0 = time.time()
    request(sock, {"op": "infer"}, {"state": state, "exterior": ext, "wrist": wrist})
    log(f"policy warm-up inference: {1000 * (time.time() - t0):.0f} ms (discarded)")


def _halt(env, gripper: float) -> None:
    """Zero joint velocity, holding the gripper where the policy last put it.

    Joint-velocity control holds the last command, so leaving the loop without this leaves the arm
    driving toward the final target while the mp4s encode. The gripper channel is a POSITION, so
    zeroing it would fling the fingers open on the way out -- it carries the last commanded value.
    """
    try:
        action = np.zeros(8, dtype=np.float64)
        action[-1] = float(gripper)
        env.step(action)
    except Exception as e:  # noqa: BLE001
        log(f"halt failed: {e}")


def run_leg(env, sock, args, ep_dir: Path) -> tuple[int, bool, dict]:
    """Run the policy closed-loop and write the leg. Returns (n_frames, aborted, stats).

    Every array recorded here is the one a teleop leg records, produced the same way: the state rows
    are the measured joints and gripper, and ``cmd_joint_velocity`` / ``cmd_joint_position`` come from
    the DROID ``action_dict`` that ``env.step`` returns -- the IK command the arm actually consumed,
    not the policy's output re-derived. That is what makes this leg mergeable with the tamp legs
    around it: ``collect/droid_action.py`` recognises a stored ``cmd_joint_velocity`` that satisfies
    the DROID identity and exports it unchanged, exactly as it does for teleop.

    ``stats`` reports what the loop actually managed -- the achieved control rate and any camera
    that stopped delivering. The policy was trained at a fixed 15 Hz, so a leg that ran slower ran
    off-distribution, and until this was measured that went unnoticed: the 2026-09-09 leg averaged
    9.2 Hz because every eighth step (the one where the action queue empties and the U-Net denoises)
    took 400 ms rather than 66. See ``--num-inference-steps``.
    """
    from hitl_dp.wire import request

    ext_frames, ext2_frames, wrist_frames = [], [], []
    joint_log, grip_log, ft_log = [], [], []
    cmd_jv_log, cmd_jp_log, cmd_g_log = [], [], []
    step_dt: list[float] = []

    # Frame counters, when the env exposes them (StableRobotEnv does). The frames themselves cannot
    # answer "is this camera still live": a stalled ZED keeps handing back its last good image.
    frame_counts = getattr(env, "camera_frame_counts", None)
    last_counts = frame_counts() if frame_counts else {}
    stale = dict.fromkeys(last_counts, 0)
    stalled = set()

    request(sock, {"op": "reset"})  # no observation or action history carried in from an earlier leg
    _ABORT["v"] = False
    aborted = False
    last_gripper = 0.0
    period = 1.0 / CONTROL_HZ
    started = time.time()
    try:
        for step in range(args.max_steps):
            if _ABORT["v"]:
                aborted = True
                log(f"force-stopped after {len(joint_log)} steps; keeping what was captured")
                break
            t0 = time.time()

            obs = env.get_observation()
            ext, ext2, wrist = _extract_cameras(
                obs, args.external_camera_id, args.external_2_camera_id, args.hand_camera_id
            )
            if ext is None or wrist is None:
                raise RuntimeError("external and/or wrist camera returned no frame (check the serials)")
            ext_frames.append(ext)
            wrist_frames.append(wrist)
            # Appended every step even when None, so it stays index-aligned with the other cameras; a
            # single dropped frame skips ext2 at write time rather than shifting every later one.
            ext2_frames.append(ext2)

            if frame_counts:
                counts = frame_counts()
                for serial, count in counts.items():
                    stale[serial] = 0 if count != last_counts.get(serial) else stale[serial] + 1
                    if stale[serial] >= STALE_FRAME_WARN_STEPS and serial not in stalled:
                        stalled.add(serial)
                        log(f"WARNING: camera {serial} has not produced a frame in "
                            f"{stale[serial]} steps; the policy is seeing a frozen image")
                last_counts = counts

            rs = obs["robot_state"]
            joints = np.asarray(rs["joint_positions"], dtype=np.float32).reshape(-1)
            gripper = float(np.asarray(rs["gripper_position"]).reshape(-1)[0])
            joint_log.append(joints)
            grip_log.append(gripper)
            ft_log.append(time.time())

            # The policy's observation is exactly the dataset's: measured joints and the MEASURED
            # gripper (not the last command), which is what `convert.build_features` wrote as
            # observation.state. Raw camera frames go over the wire and the server resizes them, so
            # the training-time scaler is applied by the project that owns it (hitl_dp.serve).
            _, out = request(
                sock,
                {"op": "infer"},
                {
                    "state": np.concatenate([joints, [gripper]]).astype(np.float32),
                    "exterior": ext,
                    "wrist": wrist,
                },
            )
            action = np.asarray(out["action"], dtype=np.float64).reshape(-1)
            if action.shape != (8,):
                raise RuntimeError(f"expected an 8-dim action from the policy, got {action.shape}")

            # Joint velocities scale; the gripper does not -- it is a position target, and scaling it
            # would move the fingers somewhere the policy never asked for.
            jv = np.clip(action[:7] * args.velocity_scale, -1.0, 1.0)
            grip_cmd = 1.0 if action[7] > 0.5 else 0.0
            last_gripper = grip_cmd

            # env.step returns DROID's action_dict for the JOINT-VELOCITY space: joint_velocity is the
            # command as given (already the [-1,1] DROID convention the dataset stores), and
            # joint_position is measured + joint_velocity * max_joint_delta -- the IK target that went
            # to the controller. Both are recorded, as teleop does.
            info = env.step(np.concatenate([jv, [grip_cmd]]))
            if not isinstance(info, dict) or "joint_velocity" not in info or "joint_position" not in info:
                raise RuntimeError(
                    "env.step did not return the DROID action_dict with joint_velocity/joint_position; "
                    "cannot capture the commanded joint velocity"
                )
            cmd_jv_log.append(np.asarray(info["joint_velocity"], dtype=np.float32).reshape(7))
            cmd_jp_log.append(np.asarray(info["joint_position"], dtype=np.float32).reshape(7))
            cmd_g_log.append(grip_cmd)

            if step and step % 75 == 0:
                log(f"step {step}/{args.max_steps} ({time.time() - started:.0f}s elapsed)")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
            step_dt.append(max(dt, period))
    finally:
        _halt(env, last_gripper)

    stats = _rate_stats(step_dt, period)
    if stats:
        log("control rate: %.1f Hz achieved against the %d Hz the policy was trained at "
            "(median step %.0f ms, slowest %.0f ms, %d of %d steps over period)"
            % (stats["control_hz"], CONTROL_HZ, 1000 * stats["step_s_median"],
               1000 * stats["step_s_max"], stats["steps_over_period"], stats["steps"]))
        if stats["control_hz"] < 0.9 * CONTROL_HZ:
            log("  the loop could not keep up; every action was held longer than in training. "
                "--num-inference-steps trades denoising fidelity for a shorter inference step.")
    if stalled:
        stats["stalled_cameras"] = sorted(stalled)

    n = min(len(ext_frames), len(wrist_frames), len(joint_log), len(cmd_jv_log), len(cmd_jp_log))
    if n < 2:
        # Nothing usable: a force-stop in the first moment, or a leg that failed immediately. The
        # caller deletes the dir; the trajectory carries on with the legs it already has.
        return 0, aborted, stats

    _write_video(ext_frames[:n], ep_dir / EXTERNAL_CAM)
    _write_video(wrist_frames[:n], ep_dir / HAND_CAM)
    cameras = {"exterior_image_1_left": EXTERNAL_CAM, "wrist_image_left": HAND_CAM}
    if all(f is not None for f in ext2_frames[:n]):
        _write_video(ext2_frames[:n], ep_dir / EXTERNAL_CAM_2)
        cameras["exterior_image_2_left"] = EXTERNAL_CAM_2

    frame_time = np.asarray(ft_log[:n], dtype=np.float64)
    write_robot_state_npz(
        ep_dir / "robot_state.npz",
        joint_position=np.stack(joint_log[:n]),
        gripper_position=np.asarray(grip_log[:n], dtype=np.float32),
        cmd_joint_position=np.stack(cmd_jp_log[:n]).astype(np.float32),
        cmd_joint_velocity=np.stack(cmd_jv_log[:n]).astype(np.float32),
        cmd_gripper=np.asarray(cmd_g_log[:n], dtype=np.float32),
        frame_time=frame_time,
    )
    write_meta(
        ep_dir / "_meta.json",
        instruction=args.instruction, n_frames=n, config_id=args.config_id,
        timestamp=ep_dir.name, cameras=cameras,
        record_start=float(frame_time[0]), record_stop=float(frame_time[-1]),
        trajectory_id=args.trajectory_id or None, segment_source=SEGMENT_SOURCE,
        preempted=aborted,
    )
    return n, aborted, stats


def _rate_stats(step_dt: list[float], period: float) -> dict:
    """What the control loop achieved, for the log and the result file."""
    if not step_dt:
        return {}
    dt = np.asarray(step_dt, dtype=np.float64)
    return {
        "steps": int(dt.size),
        "control_hz": float(1.0 / dt.mean()),
        "step_s_median": float(np.median(dt)),
        "step_s_max": float(dt.max()),
        "steps_over_period": int((dt > period * 1.05).sum()),
    }


def _write_result(path: str, **fields) -> None:
    """The one thing tiptop reads back off this process. Absent means it died before finishing."""
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(fields))
    except OSError as e:
        log(f"could not write the result file {path}: {e}")


def main(args) -> int:
    from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id

    # Same resolution order as teleop/eval: explicit flag > TIPTOP_*_CAMERA_ID (already applied by
    # parameters.py) > the rig's default serials. An unset environment still finds the real cameras.
    args.external_camera_id = args.external_camera_id or varied_camera_1_id
    args.external_2_camera_id = args.external_2_camera_id or varied_camera_2_id
    args.hand_camera_id = args.hand_camera_id or hand_camera_id
    sys.path.insert(0, str(Path(args.policy_dir) / "src"))  # hitl_dp.wire, stdlib + numpy only

    from hitl_dp.wire import connect

    output_root = Path(args.output_root)
    (output_root / "eval").mkdir(parents=True, exist_ok=True)
    _install_signal_handlers()

    server = env = sock = None
    ep_dir = None
    try:
        port = args.port or _free_port()
        server = start_policy_server(args.policy_python, args.policy_dir, args.checkpoint,
                                     port, args.open_loop_horizon, args.device,
                                     args.num_inference_steps)
        sock = connect("127.0.0.1", port, timeout=600.0)
        from hitl_dp.wire import request

        spec, _ = request(sock, {"op": "spec"})
        log(f"policy ready: {spec}")

        # do_reset=False is the whole point of a hand-off leg: the arm stays exactly where the tamp
        # leg parked it, holding whatever it was holding, and the policy takes over from there.
        # Homing first would throw away the state the plan just established.
        #
        # StableRobotEnv, not RobotEnv, and camera_serials left to its own discovery -- both exactly
        # as teleop_capture does on the same hand-off. The background capture processes are what keep
        # the ZEDs alive across a re-plan stall (~335 ms every open_loop_horizon steps), which is
        # precisely the pause an on-demand grab drops out on. gripper_action_space="position" because
        # the policy's last channel is the binary gripper TARGET the dataset stores, not a velocity.
        from droid.stable_camera_env import StableRobotEnv
        env = StableRobotEnv(action_space="joint_velocity", gripper_action_space="position", do_reset=False)
        log("created the DROID env (joint velocity, gripper position, arm left where TAMP parked it)")

        # Before anything is recorded: the ZEDs were released by tiptop moments ago and
        # StableRobotEnv's buffers start black. It now blocks until each camera has grabbed, and
        # this confirms the two the policy actually reads carry a picture.
        wait_for_live_cameras(env, args)
        log("both cameras are delivering real frames")
        warm_up_policy(env, sock, args)

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ep_dir = output_root / "eval" / ts  # staging, same as a teleop hand-off leg; never labeled here
        ep_dir.mkdir(parents=True, exist_ok=True)
        log(f"running up to {args.max_steps} steps into {ep_dir}")
        n, aborted, stats = run_leg(env, sock, args, ep_dir)
        if not n:
            shutil.rmtree(ep_dir, ignore_errors=True)
            log("nothing usable captured; the leg was discarded")
            _write_result(args.result_file, ok=True, n_frames=0, dir=None, preempted=aborted,
                          **stats)
            return 0
        log(f"leg saved: {n} frames in {ep_dir}" + (" (force-stopped)" if aborted else ""))
        _write_result(args.result_file, ok=True, n_frames=n, dir=str(ep_dir), preempted=aborted,
                      **stats)
        return 0
    except Exception as e:  # noqa: BLE001
        log(f"policy leg failed: {type(e).__name__}: {e}")
        if env is not None:
            _halt(env, 0.0)
        if ep_dir is not None and not (ep_dir / "_meta.json").is_file():
            shutil.rmtree(ep_dir, ignore_errors=True)
        _write_result(args.result_file, ok=False, error=f"{type(e).__name__}: {e}", n_frames=0, dir=None)
        raise
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        # Last, and always: it holds the GPU and the port. tiptop reopens the cameras the moment we
        # exit, so anything still alive here is something it will collide with.
        stop_policy_server(server)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True, help="runs/<ws>/tamp/<name>; the leg lands in eval/<ts>")
    p.add_argument("--instruction", default="", help="the phase's instructions, stamped into _meta.json")
    p.add_argument("--config-id", default="tamp/policy", help="config id recorded on the leg")
    p.add_argument("--trajectory-id", default="", help="the tamp trajectory this leg is a segment of")
    p.add_argument("--checkpoint", required=True, help="LeRobot checkpoint dir (holding pretrained_model/)")
    p.add_argument("--policy-python", required=True, help="interpreter for hitl_dp.serve (the DP venv)")
    p.add_argument("--policy-dir", required=True, help="hitl-baseline/diffusion_policy")
    p.add_argument("--open-loop-horizon", type=int, default=8)
    p.add_argument(
        "--num-inference-steps", type=int, default=0,
        help="denoising steps per inference; 0 keeps the checkpoint's own value (DDPM: 100, which "
             "costs ~400 ms and drops the loop to ~9 Hz on every step where the action queue "
             "empties). Lower it to hold 15 Hz, at some cost in action quality.",
    )
    p.add_argument("--max-steps", type=int, default=450)
    p.add_argument("--velocity-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=0, help="0 picks a free one")
    p.add_argument("--result-file", default="", help="where to write {ok, n_frames, dir} for tiptop")
    p.add_argument("--external-camera-id", default="")
    p.add_argument("--external-2-camera-id", default="")
    p.add_argument("--hand-camera-id", default="")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(_parse_args()))

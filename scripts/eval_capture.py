# ruff: noqa
"""Closed-loop Pi-0.5 policy EVALUATION driver for the data-collection app's Eval tab.

Adapted from ``droid/scripts/main2.py`` (the wired openpi + droid ``RobotEnv`` closed-loop loop), this
driver is instead orchestrated by the data-collection Node server: it

  1. starts an openpi policy server for a checkpoint (a subprocess run with the *openpi* venv python),
  2. runs closed-loop rollouts on the real robot (droid ``RobotEnv``, joint-velocity + gripper-position),
  3. writes each rollout in the data-collection RAW EPISODE FORMAT (external_cam/hand_cam/external_cam_2
     mp4 + robot_state.npz + _meta.json) under ``<output_root>/<task>/<timestamp>/`` so the same
     ``collect/build_lerobot.py`` + episode media/plot serving apply,
  4. talks to the server over a JSON events file + stdin, exactly like ``tiptop_run`` (see
     ``data-collection/ARCHITECTURE.md`` §6):

       stdout/stderr .......... streamed to the session log
       $EVAL_EVENTS_FILE ...... one JSON object per line (session_start, rollout_start, rollout_saved,
                                awaiting_label, rubric_saved, awaiting_task, rollout_aborted, session_end)
       stdin (line protocol) .. {"task":"<id>"} start a rollout | {"rubric":{id:bool,...}} rate the last
                                rollout | q  finish the session

The heavy imports (``droid.robot_env``, ``openpi_client``, ``moviepy``) are lazy so the pure
data-format helpers below can be unit-tested under any numpy env (``python eval_capture.py selftest``).

Run under the DROID conda env (same as ``main2.py`` / teleop ``main.py``).
"""

import dataclasses
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

# A force-stop (Node sends SIGINT to the DRIVER pid only, never the process group, so the co-spawned
# policy server survives) sets this flag; run_rollout checks it each step and aborts the rollout,
# keeping the warm server for the next episode. Outside a rollout the flag is harmless (cleared at the
# next rollout start), so a stray SIGINT while parked at a prompt never crashes the driver.
_ABORT = {"v": False}


def _install_signal_handlers():
    # SIGINT (force-stop): flag only -> abort the in-flight rollout, keep the warm server.
    signal.signal(signal.SIGINT, lambda *_: _ABORT.__setitem__("v", True))
    # SIGTERM (graceful stop escalation): exit cleanly so `finally` tears the policy server down
    # instead of orphaning it (SystemExit propagates through the blocking read/infer per PEP 475).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _free_port() -> int:
    """An unused local TCP port for the policy server. Chosen in the driver (right before serve_policy
    binds it, AFTER any checkpoint download) so the pick->bind window is ~ms, not minutes."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

DROID_CONTROL_FREQUENCY = 15  # match DROID data-collection frequency (also the LeRobot build FPS)

# Camera filenames the data-collection raw format + build_lerobot expect (fixed names).
EXTERNAL_CAM = "external_cam.mp4"      # -> exterior_image_1_left (the policy's external camera)
EXTERNAL_CAM_2 = "external_cam_2.mp4"  # -> exterior_image_2_left (the other external camera)
HAND_CAM = "hand_cam.mp4"             # -> wrist_image_left


# --------------------------------------------------------------------------- #
# Events + pure data-format helpers (no robot / policy imports -- unit testable) #
# --------------------------------------------------------------------------- #
def emit(events_file, event, **fields):
    """Append one JSON event line to the events file the Node server tails."""
    if not events_file:
        return
    rec = {"event": event, **fields}
    try:
        with open(events_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    except OSError:
        pass


def score_rubric(criteria: dict, max_points_hint=None) -> dict:
    """One rollout's checked criteria -> whole-number score (mirrors scene_specs.score_env)."""
    met = {c: (1 if bool(v) else 0) for c, v in (criteria or {}).items()}
    points = int(sum(met.values()))
    max_points = int(max_points_hint) if max_points_hint else len(met)
    return {
        "points": points,
        "max_points": max_points,
        "dense": (points / max_points) if max_points > 0 else 0.0,
        "success": 1 if (max_points > 0 and points == max_points) else 0,
    }


def write_robot_state_npz(path, *, joint_position, gripper_position, cmd_joint_position,
                          cmd_joint_velocity, cmd_gripper, frame_time):
    """Write the robot_state.npz arrays in the ARCHITECTURE.md §3 schema (gripper forced binary,
    frame_time kept float64 -- float32 near the current epoch collapses every frame to one time)."""
    jp = np.asarray(joint_position, dtype=np.float32).reshape(-1, 7)
    n = len(jp)
    gp = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    cjp = np.asarray(cmd_joint_position, dtype=np.float32).reshape(-1, 7)
    cjv = np.asarray(cmd_joint_velocity, dtype=np.float32).reshape(-1, 7)
    cg = np.asarray(cmd_gripper, dtype=np.float32).reshape(-1)
    cg = np.where(cg > 0.5, 1.0, 0.0).astype(np.float32)  # binary 0/1 (never a continuous echo)
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    assert len(gp) == len(cjp) == len(cjv) == len(cg) == len(ft) == n, "state arrays disagree on length"
    np.savez(
        path, joint_position=jp, gripper_position=gp, cmd_joint_position=cjp,
        cmd_joint_velocity=cjv, cmd_gripper=cg, frame_time=ft,
    )
    return n


def write_meta(path, *, instruction, n_frames, config_id, timestamp, cameras, record_start, record_stop):
    meta = {
        "instruction": instruction,
        "fps": DROID_CONTROL_FREQUENCY,
        "n_frames": int(n_frames),
        "config_id": config_id,
        "timestamp": timestamp,
        "source": "pi05-eval",
        "cameras": cameras,
        "record_start": float(record_start),
        "record_stop": float(record_stop),
    }
    Path(path).write_text(json.dumps(meta))


def write_rubric(ep_dir, *, policy, task, scene_id, prompt, criteria, max_points, duration_steps):
    """Write rubric.json for a rated rollout and return the score dict."""
    score = score_rubric(criteria, max_points)
    rec = {
        "policy": policy, "task": task, "scene_id": scene_id, "prompt": prompt,
        "criteria": {k: bool(v) for k, v in (criteria or {}).items()},
        "duration_steps": int(duration_steps),
        "rated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **score,
    }
    (Path(ep_dir) / "rubric.json").write_text(json.dumps(rec))
    return score


def _write_video(frames, path):
    """Write HWC-RGB uint8 frames to an mp4 at the control frequency (lazy moviepy import)."""
    from moviepy.editor import ImageSequenceClip

    arr = [np.asarray(f, dtype=np.uint8) for f in frames]
    ImageSequenceClip(arr, fps=DROID_CONTROL_FREQUENCY).write_videofile(
        str(path), codec="libx264", logger=None
    )


# --------------------------------------------------------------------------- #
# Checkpoint resolution + policy server lifecycle                               #
# --------------------------------------------------------------------------- #
def resolve_checkpoint(url: str, openpi_python: str, openpi_dir: str, cache_dir: str) -> str:
    """Return a directory/URL serve_policy can load. gs://, hf://, and local paths pass through;
    a plain https://huggingface.co/<user>/<repo> is snapshot-downloaded (via the openpi venv, which
    has huggingface_hub) into ``cache_dir`` because serve_policy can't enumerate a single https file."""
    url = (url or "").strip()
    if url.startswith("https://huggingface.co/"):
        repo = url[len("https://huggingface.co/"):].strip("/")
        # strip a /tree/<rev> or /resolve/<rev> suffix if present
        for sep in ("/tree/", "/resolve/", "/blob/"):
            if sep in repo:
                repo = repo.split(sep, 1)[0]
        local = str(Path(cache_dir) / repo.replace("/", "__"))
        print(f"[eval] snapshot-downloading HF checkpoint {repo} -> {local}", flush=True)
        code = (
            "import sys; from huggingface_hub import snapshot_download; "
            "print(snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], "
            "allow_patterns=['params/**','assets/**','_CHECKPOINT_METADATA','*.json','model.safetensors']))"
        )
        subprocess.run([openpi_python, "-c", code, repo, local], cwd=openpi_dir, check=True)
        return local
    return url  # gs:// | hf:// | local path -> serve_policy.maybe_download handles it


def start_policy_server(openpi_python, openpi_dir, config, checkpoint, port, log_prefix="[server] "):
    """Spawn serve_policy.py (openpi venv) and stream its output to our stderr with a prefix."""
    cmd = [
        openpi_python, "scripts/serve_policy.py",
        "--port", str(port),
        "policy:checkpoint",
        f"--policy.config={config}",
        f"--policy.dir={checkpoint}",
    ]
    print(f"[eval] starting policy server: {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    proc = subprocess.Popen(
        cmd, cwd=openpi_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    # Pump the server's log to our stderr so it shows in the session log.
    import threading

    def _pump():
        for line in proc.stdout:
            sys.stderr.write(log_prefix + line)
            sys.stderr.flush()

    threading.Thread(target=_pump, daemon=True).start()
    return proc


def wait_for_server(host, port, proc, timeout=1800.0):
    """Block until the policy server accepts TCP connections (or dies / times out)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited early with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"policy server not ready on {host}:{port} after {timeout:.0f}s")


# --------------------------------------------------------------------------- #
# Observation extraction (mirrors main2._extract_observation)                   #
# --------------------------------------------------------------------------- #
def _extract_observation(obs_dict, left_id, right_id, wrist_id):
    if "image" not in obs_dict or not obs_dict["image"]:
        raise RuntimeError(
            "No camera frames from the DROID env. The ZED cameras were not read -- check they are "
            "connected and NOT already open in another process (a running tiptop-run or a data-collection "
            "capture session holds these same cameras exclusively)."
        )
    images = obs_dict["image"]
    left_image = right_image = wrist_image = None
    for key in images:
        if left_id and left_id in key and "left" in key:
            left_image = images[key]
        elif right_id and right_id in key and "left" in key:
            right_image = images[key]
        elif wrist_id and wrist_id in key and "left" in key:
            wrist_image = images[key]
    assert wrist_image is not None, "Could not find wrist camera image"
    left_image = left_image[..., :3][..., ::-1] if left_image is not None else None   # BGRA -> RGB
    right_image = right_image[..., :3][..., ::-1] if right_image is not None else None
    wrist_image = wrist_image[..., :3][..., ::-1]
    robot_state = obs_dict["robot_state"]
    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }


# --------------------------------------------------------------------------- #
# One rollout                                                                   #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Args:
    events_file: str = ""            # JSON events file the Node server tails ($EVAL_EVENTS_FILE)
    output_root: str = ""            # runs/<workspace>/eval/<policy>; rollouts nest under <task>/<ts>
    tasks_file: str = ""             # JSON {task_id: {prompt, scene_id, criteria:[ids]}}
    policy_name: str = ""            # policy config basename (for rubric.json)
    config: str = "pi05_droid"       # openpi TrainConfig name (--policy.config)
    url: str = ""                    # checkpoint gs:// | hf:// | https://huggingface.co/... | local
    external_camera: str = "left"    # which external cam feeds the policy: left|right
    max_steps: int = 600
    open_loop_horizon: int = 8
    velocity_scale: float = 1.0
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    openpi_python: str = ""          # interpreter for serve_policy.py (openpi venv)
    openpi_dir: str = ""             # openpi repo dir (cwd for serve_policy.py)
    # Camera ids default from the same env vars tiptop/main2 use.
    left_camera_id: str = ""
    right_camera_id: str = ""
    wrist_camera_id: str = ""


class RolloutAborted(Exception):
    """Raised to abort the current rollout (SIGINT / force-stop) without ending the session."""


def run_rollout(env, policy_client, args, task, ep_dir, image_tools):
    """Execute one closed-loop rollout of ``task`` and write the raw episode. Returns n_frames.

    Raises RolloutAborted if a force-stop (SIGINT) interrupts the rollout -- the caller discards it.
    """
    instruction = task["prompt"]
    second_camera = "left" if args.external_camera == "right" else "right"

    ext_frames, wrist_frames, ext2_frames = [], [], []
    joint_log, grip_log, ft_log, cmd_jv_log, cmd_g_log = [], [], [], [], []

    actions_done = 0
    chunk = None
    aborted = False
    _ABORT["v"] = False  # clear any SIGINT that arrived while parked at the prompt
    for _t in range(args.max_steps):
        if _ABORT["v"]:  # force-stop: abort this rollout, keep the warm server for the next episode
            aborted = True
            break
        t0 = time.time()
        obs = _extract_observation(env.get_observation(), args.left_camera_id, args.right_camera_id, args.wrist_camera_id)

        ext_img = obs[f"{args.external_camera}_image"]
        if ext_img is None:
            raise RuntimeError(f"external camera '{args.external_camera}' returned no image")
        ext_frames.append(ext_img)
        wrist_frames.append(obs["wrist_image"])
        # Append the second exterior every step (even if None) so it stays index-aligned 1:1 with the
        # timeline; a dropped frame here would otherwise shift every later ext2 frame. If ANY step
        # dropped it, we skip ext2 at write time (the build then duplicates exterior_1).
        ext2_frames.append(obs.get(f"{second_camera}_image"))

        joint_log.append(np.asarray(obs["joint_position"], dtype=np.float32).reshape(-1))
        grip_log.append(float(np.asarray(obs["gripper_position"]).reshape(-1)[0]))
        ft_log.append(time.time())  # float64 epoch seconds, captured at the observation for this frame

        # Re-query when we've exhausted the open-loop window OR the served chunk (whichever is
        # shorter) -- capping at len(chunk) avoids an IndexError if the model's action horizon is
        # shorter than open_loop_horizon (e.g. a horizon-10 checkpoint with --open-loop-horizon 16).
        if chunk is None or actions_done >= min(args.open_loop_horizon, len(chunk)):
            actions_done = 0
            req = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(ext_img, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_image"], 224, 224),
                "observation/joint_position": obs["joint_position"],
                "observation/gripper_position": obs["gripper_position"],
                "prompt": instruction,
            }
            # The SIGINT handler only sets a flag (PEP 475 auto-retries the socket call), so this
            # blocking server call is never interrupted mid-flight -- we abort at the next step check.
            chunk = policy_client.infer(req)["actions"]
            assert chunk.ndim == 2 and chunk.shape[1] == 8, f"expected (T,8) action chunk, got {chunk.shape}"

        action = np.asarray(chunk[actions_done], dtype=np.float64)
        actions_done += 1
        if args.velocity_scale != 1.0:
            action = np.concatenate([action[:-1] * args.velocity_scale, action[-1:]])
        grip = 1.0 if action[-1] > 0.5 else 0.0
        action = np.clip(np.concatenate([action[:-1], [grip]]), -1, 1)

        cmd_jv_log.append(action[:7].astype(np.float32))
        cmd_g_log.append(grip)

        env.step(action)

        elapsed = time.time() - t0
        if elapsed < 1 / DROID_CONTROL_FREQUENCY:
            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)

    if aborted:
        raise RolloutAborted()

    # Truncate every stream to a common length so frames + state stay aligned 1:1.
    n = min(len(ext_frames), len(wrist_frames), len(joint_log), len(cmd_jv_log))
    if n < 2:
        raise RolloutAborted()  # nothing usable captured
    ep_dir = Path(ep_dir)
    _write_video(ext_frames[:n], ep_dir / EXTERNAL_CAM)
    _write_video(wrist_frames[:n], ep_dir / HAND_CAM)
    cameras = {"exterior_image_1_left": EXTERNAL_CAM, "wrist_image_left": HAND_CAM}
    if len(ext2_frames) >= n and all(f is not None for f in ext2_frames[:n]):
        _write_video(ext2_frames[:n], ep_dir / EXTERNAL_CAM_2)
        cameras["exterior_image_2_left"] = EXTERNAL_CAM_2

    joints = np.stack(joint_log[:n])
    frame_time = np.asarray(ft_log[:n], dtype=np.float64)
    write_robot_state_npz(
        ep_dir / "robot_state.npz",
        joint_position=joints,
        gripper_position=np.asarray(grip_log[:n], dtype=np.float32),
        # A velocity policy issues no explicit target position; the measured q is a NaN-free fill for
        # the secondary action_joint_position channel (the trained action is cmd_joint_velocity).
        cmd_joint_position=joints,
        cmd_joint_velocity=np.stack(cmd_jv_log[:n]),
        cmd_gripper=np.asarray(cmd_g_log[:n], dtype=np.float32),
        frame_time=frame_time,
    )
    write_meta(
        ep_dir / "_meta.json",
        instruction=instruction, n_frames=n, config_id=f"eval/{args.policy_name}/{task['id']}",
        timestamp=ep_dir.name, cameras=cameras,
        record_start=float(frame_time[0]), record_stop=float(frame_time[-1]),
    )
    return n


def _read_command():
    """Read one stdin line (a JSON command or 'q'); return the parsed value or None at EOF."""
    line = sys.stdin.readline()
    if line == "":
        return None
    line = line.strip()
    if not line:
        return {}
    if line == "q":
        return "q"
    try:
        return json.loads(line)
    except ValueError:
        return {}


def main(args: Args):
    events = args.events_file
    tasks = json.loads(Path(args.tasks_file).read_text()) if args.tasks_file else {}
    args.left_camera_id = args.left_camera_id or os.environ.get("TIPTOP_EXTERNAL_CAMERA_ID", "")
    args.right_camera_id = args.right_camera_id or os.environ.get("TIPTOP_EXTERNAL_2_CAMERA_ID", "")
    args.wrist_camera_id = args.wrist_camera_id or os.environ.get("TIPTOP_HAND_CAMERA_ID", "")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _install_signal_handlers()

    emit(events, "session_start")
    server = None
    try:
        # 1) resolve checkpoint (may snapshot-download), then pick a free port + start the server.
        cache_dir = os.path.join(args.openpi_dir or ".", "checkpoints", "eval_cache")
        checkpoint = resolve_checkpoint(args.url, args.openpi_python, args.openpi_dir, cache_dir)
        args.remote_port = _free_port()  # picked here (post-download) so pick->bind is ~ms
        server = start_policy_server(args.openpi_python, args.openpi_dir, args.config, checkpoint, args.remote_port)
        wait_for_server(args.remote_host, args.remote_port, server)

        # 2) robot env + policy client (lazy imports: DROID env only).
        from openpi_client import image_tools, websocket_client_policy
        from droid.robot_env import RobotEnv

        env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
        print("[eval] created the DROID env", flush=True)
        policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

        # 3) rollout loop, driven by the Node server over stdin.
        first = True
        while True:
            emit(events, "awaiting_task")
            cmd = _read_command()
            if cmd is None or cmd == "q":
                break
            if not isinstance(cmd, dict) or "task" not in cmd:
                continue
            task_id = str(cmd["task"])
            task = tasks.get(task_id)
            if task is None:
                emit(events, "error", message=f"unknown task {task_id!r}")
                continue
            task.setdefault("id", task_id)

            if not first:
                try:
                    env.reset()
                except Exception as e:  # noqa: BLE001
                    print(f"[eval] env.reset() failed: {e}", flush=True)
            first = False

            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ep_dir = output_root / task_id / ts
            ep_dir.mkdir(parents=True, exist_ok=True)
            emit(events, "rollout_start", dir=str(ep_dir), task=task_id, prompt=task["prompt"])
            try:
                n = run_rollout(env, policy_client, args, task, ep_dir, image_tools)
            except RolloutAborted:
                shutil.rmtree(ep_dir, ignore_errors=True)
                emit(events, "rollout_aborted", dir=str(ep_dir), task=task_id)
                continue
            except Exception as e:  # noqa: BLE001
                shutil.rmtree(ep_dir, ignore_errors=True)
                emit(events, "error", message=f"rollout failed: {e}")
                continue
            emit(events, "rollout_saved", dir=str(ep_dir), task=task_id, n_frames=n)

            # 4) wait for the rubric (or a finish/quit). Loop so a stray/blank line doesn't drop the
            #    rating -- the server only ever writes {"rubric":...} or "q" here.
            emit(events, "awaiting_label", dir=str(ep_dir), task=task_id)
            finish = False
            while True:
                rub = _read_command()
                if rub is None or rub == "q":
                    finish = True
                    break
                if isinstance(rub, dict) and "rubric" in rub:
                    submitted = rub.get("rubric") or {}
                    # Score ONLY the task's criteria (drop any unknown/extra ids, default missing to
                    # False) so points can never exceed max_points and dense stays in [0,1].
                    crits = task.get("criteria") or list(submitted.keys())
                    criteria = {c: bool(submitted.get(c, False)) for c in crits}
                    score = write_rubric(
                        ep_dir, policy=args.policy_name, task=task_id, scene_id=task.get("scene_id"),
                        prompt=task["prompt"], criteria=criteria, max_points=len(crits), duration_steps=n,
                    )
                    emit(events, "rubric_saved", dir=str(ep_dir), task=task_id, **score)
                    break
                # anything else: ignore and keep awaiting the rubric
            if finish:
                break
        emit(events, "session_end")  # normal end only -- a warmup/rollout crash skips this so Node
        #                              sees the nonzero exit as an error (not a masked "done").
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:  # noqa: BLE001
                server.kill()


# --------------------------------------------------------------------------- #
# Offline self-test of the pure data-format helpers (no robot / policy needed)  #
# --------------------------------------------------------------------------- #
def _selftest():
    import tempfile

    p = 0
    f = 0

    def ok(c, m):
        nonlocal p, f
        if c:
            p += 1
            print("  ✓", m)
        else:
            f += 1
            print("  ✗", m)

    print("[eval_capture selftest]")
    s = score_rubric({"a": True, "b": False, "c": True}, 3)
    ok(s == {"points": 2, "max_points": 3, "dense": 2 / 3, "success": 0}, "score_rubric partial")
    ok(score_rubric({"a": True}, 1)["success"] == 1, "score_rubric all-met -> success")
    ok(score_rubric({}, 0) == {"points": 0, "max_points": 0, "dense": 0.0, "success": 0}, "score_rubric empty")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        emit(str(d / "ev.jsonl"), "rollout_saved", dir="/x", n_frames=3)
        line = (d / "ev.jsonl").read_text().strip()
        ok(json.loads(line) == {"event": "rollout_saved", "dir": "/x", "n_frames": 3}, "emit writes one JSON line")

        n = write_robot_state_npz(
            d / "robot_state.npz",
            joint_position=np.zeros((3, 7)), gripper_position=[0, 0.5, 1.0],
            cmd_joint_position=np.ones((3, 7)), cmd_joint_velocity=np.full((3, 7), 0.2),
            cmd_gripper=[0.1, 0.6, 0.9],  # -> binarized 0,1,1
            frame_time=[1.78e9, 1.78e9 + 0.066, 1.78e9 + 0.132],
        )
        with np.load(d / "robot_state.npz") as sd:
            ok(n == 3 and sd["joint_position"].shape == (3, 7), "npz has [N,7] joints")
            ok(sd["frame_time"].dtype == np.float64, "frame_time is float64")
            ok(list(sd["cmd_gripper"]) == [0.0, 1.0, 1.0], "cmd_gripper forced binary 0/1")
            ok(sd["frame_time"][-1] - sd["frame_time"][0] > 0.13, "float64 frame_time keeps sub-second deltas")

        write_meta(d / "_meta.json", instruction="do it", n_frames=3, config_id="eval/p/toys",
                   timestamp="2026", cameras={"exterior_image_1_left": EXTERNAL_CAM},
                   record_start=1.78e9, record_stop=1.78e9 + 0.132)
        m = json.loads((d / "_meta.json").read_text())
        ok(m["fps"] == 15 and m["source"] == "pi05-eval" and m["record_stop"] > m["record_start"], "meta ok")

        sc = write_rubric(d, policy="p", task="toys", scene_id=6, prompt="do it",
                          criteria={"a": True, "b": False}, max_points=2, duration_steps=100)
        rj = json.loads((d / "rubric.json").read_text())
        ok(sc["points"] == 1 and rj["criteria"] == {"a": True, "b": False} and rj["max_points"] == 2, "rubric.json written")

    print(f"  ==== {p} passed, {f} failed ====")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    import tyro

    main(tyro.cli(Args))

# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import json
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    # Defaults come from the env (TIPTOP_*_CAMERA_ID, same vars as tiptop.yml) so a swapped-in
    # camera works without a code edit; --wrist-camera-id etc. still override.
    left_camera_id: str = ""  # e.g., "24259877"
    right_camera_id: str = os.environ.get("TIPTOP_EXTERNAL_CAMERA_ID", "32439448")  # e.g., "24514023"
    wrist_camera_id: str = os.environ.get("TIPTOP_HAND_CAMERA_ID", "14846828")  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Multiplier applied to the 7 joint-velocity action dims before clipping (the gripper dim is
    # left untouched). 1.0 = unchanged. Values >1 make the robot traverse the same joint path
    # faster. This pushes the policy off the velocity distribution it was trained on, so overshoot
    # risk grows with the value AND with open_loop_horizon (more steps execute before re-observing) --
    # raise it gradually (e.g. 1.25) and consider lowering open_loop_horizon to re-observe sooner.
    velocity_scale: float = 1

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # Output parameters
    # Root directory where results are saved. One subdirectory is created per rollout.
    # Defaults to <repo_parent>/pi05-results (i.e. a sibling of the droid repo).
    results_dir: str = ""

    # Task instruction for the policy. If set, it is used for the first rollout instead of
    # prompting on stdin (so the instruction can be passed non-interactively, e.g. from a
    # wrapper script). Subsequent rollouts ("do one more eval?") still prompt interactively.
    prompt: str = ""


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Resolve the results directory. By default we save into <repo_parent>/pi05-results,
    # i.e. a sibling of the droid repo (this file lives at droid/scripts/main2.py).
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(os.path.dirname(repo_root), "pi05-results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to {results_dir}")

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "instruction", "run_dir"])

    first_rollout = True
    while True:
        # Use the instruction passed via --prompt for the first rollout (non-interactive),
        # otherwise fall back to prompting on stdin.
        if first_rollout and args.prompt:
            instruction = args.prompt
            print(f"Enter instruction: {instruction}")
        else:
            instruction = input("Enter instruction: ")
        first_rollout = False

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout. Each rollout gets its own subdirectory.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(results_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        video = []  # external (policy) camera frames
        wrist_video = []  # wrist camera frames
        # The non-policy exterior camera, logged as DROID exterior_2 when its id is set.
        second_camera = "left" if args.external_camera == "right" else "right"
        video_2 = []  # second external camera frames (exterior_2)
        # Per-frame robot state/action, captured for the LeRobot (DROID-format) export.
        joint_pos_log = []  # [7] joint positions
        gripper_pos_log = []  # [1] gripper position
        cartesian_pos_log = []  # [6] cartesian pose
        action_log = []  # [8] executed action (7 joint-velocity + gripper)
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk (into this rollout's directory)
                    save_to_disk=t_step == 0,
                    save_dir=run_dir,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])
                wrist_video.append(curr_obs["wrist_image"])
                second_image = curr_obs.get(f"{second_camera}_image")
                if second_image is not None:
                    video_2.append(second_image)

                # Log robot state aligned with this frame (for the LeRobot DROID export).
                joint_pos_log.append(np.asarray(curr_obs["joint_position"], dtype=np.float32).reshape(-1))
                gripper_pos_log.append(np.atleast_1d(np.asarray(curr_obs["gripper_position"], dtype=np.float32)).reshape(-1))
                cartesian_pos_log.append(np.asarray(curr_obs["cartesian_position"], dtype=np.float32).reshape(-1))

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.ndim == 2 and pred_action_chunk.shape[1] == 8, (
                        f"Expected action chunk of shape (T, 8), got {pred_action_chunk.shape}"
                    )

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Optionally speed up motion by scaling the joint-velocity dims (not the gripper).
                # The subsequent clip to [-1, 1] still bounds the result for safety.
                if args.velocity_scale != 1.0:
                    action = np.concatenate([action[:-1] * args.velocity_scale, action[-1:]])

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                # Log the executed action aligned with this frame.
                action_log.append(np.asarray(action, dtype=np.float32).reshape(-1))

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        # Save the external (policy) and wrist camera videos into this rollout's directory.
        video = np.stack(video)
        exterior_path = os.path.join(run_dir, "exterior.mp4")
        ImageSequenceClip(list(video), fps=10).write_videofile(exterior_path, codec="libx264")

        wrist_path = os.path.join(run_dir, "wrist.mp4")
        if len(wrist_video) > 0:
            wrist_arr = np.stack(wrist_video)
            ImageSequenceClip(list(wrist_arr), fps=10).write_videofile(wrist_path, codec="libx264")

        exterior_2_path = os.path.join(run_dir, "exterior_2.mp4")
        if len(video_2) > 0:
            video_2_arr = np.stack(video_2)
            ImageSequenceClip(list(video_2_arr), fps=10).write_videofile(exterior_2_path, codec="libx264")

        # Dump per-frame state/action so the rollout can be exported to LeRobot DROID
        # format (built post-hoc by scripts/lerobot_export.py, run from run-pi05.sh).
        # Truncate every stream to a common length so frames and states stay aligned.
        n_frames = min(len(video), len(joint_pos_log), len(gripper_pos_log), len(cartesian_pos_log), len(action_log))
        if n_frames > 0:
            cameras = {"observation.images.exterior_1_left": os.path.basename(exterior_path)}
            if len(video_2) >= n_frames:
                cameras["observation.images.exterior_2_left"] = os.path.basename(exterior_2_path)
            if len(wrist_video) > 0:
                cameras["observation.images.wrist_left"] = os.path.basename(wrist_path)
            lerobot_raw = {
                "fps": DROID_CONTROL_FREQUENCY,
                "instruction": instruction,
                "robot_type": "franka",
                "cameras": cameras,
                "joint_position": np.stack(joint_pos_log[:n_frames]).tolist(),
                "gripper_position": np.stack(gripper_pos_log[:n_frames]).tolist(),
                "cartesian_position": np.stack(cartesian_pos_log[:n_frames]).tolist(),
                "action": np.stack(action_log[:n_frames]).tolist(),
            }
            with open(os.path.join(run_dir, "_lerobot_raw.json"), "w") as f:
                json.dump(lerobot_raw, f)

        success: float | None = None
        while success is None:
            raw = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            ).strip()
            if raw.lower() == "y":
                success = 1.0
            elif raw.lower() == "n":
                success = 0.0
            else:
                try:
                    value = float(raw) / 100
                except ValueError:
                    print(f"Please enter 'y', 'n', or a number 0-100 (got: {raw!r})")
                    continue
                if not (0 <= value <= 1):
                    print(f"Success must be a number in [0, 100] but got: {value * 100}")
                    continue
                success = value

        # Write a per-run metadata file so each rollout is self-describing.
        metadata = {
            "timestamp": timestamp,
            "instruction": instruction,
            "success": success,
            "duration": int(t_step),
            "external_camera": args.external_camera,
            "exterior_video": os.path.basename(exterior_path),
            "wrist_video": os.path.basename(wrist_path) if len(wrist_video) > 0 else None,
            "max_timesteps": args.max_timesteps,
            "open_loop_horizon": args.open_loop_horizon,
        }
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # DataFrame.append was removed in pandas 2.0; use pd.concat instead.
        row = pd.DataFrame(
            [
                {
                    "success": success,
                    "duration": t_step,
                    "instruction": instruction,
                    "run_dir": os.path.relpath(run_dir, results_dir),
                }
            ]
        )
        df = row if df.empty else pd.concat([df, row], ignore_index=True)

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    csv_timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join(results_dir, f"eval_{csv_timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False, save_dir="."):
    # No "image" key means read_cameras() returned no frames at all -- i.e. the ZED
    # cameras weren't opened/read. The usual cause is the cameras being held by another
    # process (ZED cameras are exclusive-access, and a running `tiptop-run --enable-recording`
    # opens these same three), or being unplugged. Fail with that context instead of a bare
    # KeyError so the cause is obvious.
    if "image" not in obs_dict or not obs_dict["image"]:
        raise RuntimeError(
            "No camera frames returned by the DROID env (obs_dict has no 'image'). The ZED "
            "cameras were not read -- check they are connected and not already open in another "
            "process (e.g. a running tiptop-run, which holds the same cameras exclusively)."
        )
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id and args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id and args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id and args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]
    
    
    assert wrist_image is not None, "Could not find wrist camera"

    # Drop the alpha dimension
    left_image = left_image[..., :3] if left_image is not None else None
    right_image = right_image[..., :3] if right_image is not None else None
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1] if left_image is not None else None
    right_image = right_image[..., ::-1] if right_image is not None else None
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        imgs = [im for im in [left_image, wrist_image, right_image] if im is not None]
        combined_image = np.concatenate(imgs, axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save(os.path.join(save_dir, "robot_camera_views.png"))

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

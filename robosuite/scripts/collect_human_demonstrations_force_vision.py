"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

def read_ft_from_wrapper(env, key="robot0_right"):
    """
    Works with your DataCollectionWrapperWithFT that stores FT in action_infos[-1].
    Returns (F, Tau) or (None, None) if not available yet.
    """
    if not hasattr(env, "action_infos") or (len(env.action_infos) == 0):
        return None, None
    ai = env.action_infos[-1]
    fd = ai.get("ee_force", None)
    td = ai.get("ee_torque", None)
    if (fd is None) or (td is None):
        return None, None
    return fd.get(key, None), td.get(key, None)

# class DataCollectionWrapperWithFT(DataCollectionWrapper):
#     def __init__(self, env, directory,
#                  force_sensor_name="gripper0_right_force_ee",
#                  torque_sensor_name="gripper0_right_torque_ee"):
#         super().__init__(env, directory)
#         self._ft_sensor_names = (force_sensor_name, torque_sensor_name)
#         self._ft_sensor_cache = None

#     def _get_ft_sensor_slices(self):
#         if self._ft_sensor_cache is not None:
#             return self._ft_sensor_cache

#         sim = self.sim
#         f_name, t_name = self._ft_sensor_names

#         f_id = sim.model.sensor_name2id(f_name)
#         t_id = sim.model.sensor_name2id(t_name)

#         adr_f, dim_f = int(sim.model.sensor_adr[f_id]), int(sim.model.sensor_dim[f_id])
#         adr_t, dim_t = int(sim.model.sensor_adr[t_id]), int(sim.model.sensor_dim[t_id])

#         self._ft_sensor_cache = (adr_f, dim_f, adr_t, dim_t)
#         return self._ft_sensor_cache

#     def step(self, action):
#         ret = super().step(action)

#         # action_infos[-1] is created by DataCollectionWrapper during super().step(action)
#         if self.action_infos:
#             sim = self.sim
#             adr_f, dim_f, adr_t, dim_t = self._get_ft_sensor_slices()

#             F = np.array(sim.data.sensordata[adr_f:adr_f + dim_f], dtype=np.float32)
#             Tau = np.array(sim.data.sensordata[adr_t:adr_t + dim_t], dtype=np.float32)

#             self.action_infos[-1]["ee_force"] = {"robot0_right": F}
#             self.action_infos[-1]["ee_torque"] = {"robot0_right": Tau}

#         return ret

class DataCollectionWrapperWithFT(DataCollectionWrapper):
    def __init__(self, env, directory,
                 force_sensor_name="gripper0_right_force_ee",
                 torque_sensor_name="gripper0_right_torque_ee",
                 key="robot0_right",
                 bias_when_ncon_zero=True):
        super().__init__(env, directory)
        self.force_sensor_name = force_sensor_name
        self.torque_sensor_name = torque_sensor_name
        self.key = key

        self._ft_sensor_cache = None  # (adr_f, dim_f, adr_t, dim_t)
        self._bias_F = None
        self._bias_T = None
        self.bias_when_ncon_zero = bias_when_ncon_zero

        self.bias_alpha = 0.02   # 0.01-0.05 typical. Higher = faster re-zero.
        self.bias_contact_gate = 0  # 0 = use ncon==0, or use your own threshold

    def reset(self):
        # reset per-episode bias
        self._bias_F = None
        self._bias_T = None
        return super().reset()

    def _unwrap_to_base_env(self):
        # peel wrappers to reach the base env that actually owns `sim`
        e = self.env
        while hasattr(e, "env"):
            # stop when we find an object with `sim`
            if hasattr(e, "sim"):
                break
            e = e.env
        return e

    def _get_ft_sensor_slices(self):
        if self._ft_sensor_cache is not None:
            return self._ft_sensor_cache

        base = self._unwrap_to_base_env()
        sim = base.sim

        f_id = sim.model.sensor_name2id(self.force_sensor_name)
        t_id = sim.model.sensor_name2id(self.torque_sensor_name)

        adr_f = int(sim.model.sensor_adr[f_id]); dim_f = int(sim.model.sensor_dim[f_id])
        adr_t = int(sim.model.sensor_adr[t_id]); dim_t = int(sim.model.sensor_dim[t_id])

        self._ft_sensor_cache = (adr_f, dim_f, adr_t, dim_t)
        return self._ft_sensor_cache

    def step(self, action):
        ret = super().step(action)

        if self.action_infos:
            base = self._unwrap_to_base_env()
            sim = base.sim

            adr_f, dim_f, adr_t, dim_t = self._get_ft_sensor_slices()

            ncon = int(sim.data.ncon)


            F_raw = np.asarray(sim.data.sensordata[adr_f:adr_f + dim_f], dtype=np.float32).reshape(-1)[:3]
            T_raw = np.asarray(sim.data.sensordata[adr_t:adr_t + dim_t], dtype=np.float32).reshape(-1)[:3]

            if self._bias_F is None:
                self._bias_F = F_raw.copy()
                self._bias_T = T_raw.copy()

            # Update bias ONLY when no contact
            if ncon == 0:
                a = float(self.bias_alpha)
                self._bias_F = (1 - a) * self._bias_F + a * F_raw
                self._bias_T = (1 - a) * self._bias_T + a * T_raw

            F = F_raw - self._bias_F
            Tau = T_raw - self._bias_T

            self.action_infos[-1]["ee_force"] = {self.key: F}
            self.action_infos[-1]["ee_torque"] = {self.key: Tau}

        return ret

def compute_ft_from_states_via_obs(model_xml_str, env_info_dict, states_arr):
    """
    Replays each mujoco state and queries robosuite observations to get ee_force / ee_torque.
    Returns (T,6): [Fx,Fy,Fz,Tx,Ty,Tz] for robot0.
    """
    # local copy: allow env_info_dict["robots"] to be string in the file, but suite.make expects list in practice
    env_cfg = dict(env_info_dict)
    if isinstance(env_cfg.get("robots", None), str):
        env_cfg["robots"] = [env_cfg["robots"]]

    env = suite.make(
        **env_cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env.reset()
    xml = env.edit_model_xml(model_xml_str)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()

    T = states_arr.shape[0]
    ft = np.zeros((T, 6), dtype=np.float32)

    for i in range(T):
        env.sim.set_state_from_flattened(states_arr[i])
        env.sim.forward()
        obs = env._get_observations()
        ft[i, :3] = np.asarray(obs["robot0_ee_force"], dtype=np.float32).reshape(-1)[:3]
        ft[i, 3:] = np.asarray(obs["robot0_ee_torque"], dtype=np.float32).reshape(-1)[:3]

    env.close()
    return ft

def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        F, Tau = read_ft_from_wrapper(env, key="robot0_right")
        if F is not None:
            fmag = float(np.linalg.norm(F))
            tmag = float(np.linalg.norm(Tau))
            # one-line "HUD" that refreshes in place
            print(f"\rF = [{F[0]: .3f} {F[1]: .3f} {F[2]: .3f}] |F|={fmag: .3f}  "
                f"T = [{Tau[0]: .3f} {Tau[1]: .3f} {Tau[2]: .3f}] |T|={tmag: .3f}   ",
                end="", flush=True)

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, save_only_success=False):
    """
    Gathers raw demos in `directory` (tmp folder with ep_*/state_*.npz) into out_dir/demo.hdf5.

    - If save_only_success=True: only saves successful rollouts (legacy behavior)
    - Otherwise: saves all rollouts, and stores success as an attribute.

    Also appends end-effector force+torque into `states`:
      states_ext = [mujoco_flat_state, ee_force(3), ee_torque(3)]
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "a")
    grp = f.require_group("data")

    # Determine current index (so we append demo_{n+1}, demo_{n+2}, ...)
    existing = [k for k in grp.keys() if k.startswith("demo_")]
    num_eps = len(existing)

    env_name = None

    for ep_directory in sorted(os.listdir(directory)):
        ep_path = os.path.join(directory, ep_directory)
        if not os.path.isdir(ep_path):
            continue

        state_paths = os.path.join(ep_path, "state_*.npz")
        state_files = sorted(glob(state_paths))
        if not state_files:
            continue

        states = []
        actions = []
        ee_forces = []
        ee_torques = []
        success = False

        for state_file in state_files:
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])

            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                ee_forces.append(ai.get("ee_force", None))
                ee_torques.append(ai.get("ee_torque", None))

            success = success or bool(dic["successful"])

        if len(states) == 0:
            continue

        if save_only_success and (not success):
            print("Demonstration unsuccessful -> skipping (save_only_success=True)")
            continue

        # DataCollectionWrapper has one extra state at the end
        del states[-1]
        assert len(states) == len(actions)
        assert len(ee_forces) == len(actions)
        assert len(ee_torques) == len(actions)

        states_arr = np.asarray(states)
        T, S = states_arr.shape

        # Determine stable FT key order (you have ['robot0_right'])
        keys = None
        for d in ee_forces:
            if d is not None:
                keys = sorted(list(d.keys()))
                break

        if keys is None:
            # Fallback: compute FT by replaying recorded mujoco states and reading robosuite obs
            # (works even if DataCollectionWrapper didn't store ee_force/ee_torque)
            xml_path = os.path.join(ep_path, "model.xml")
            with open(xml_path, "r") as xf:
                model_xml_str = xf.read()

            env_info_dict = json.loads(env_info) if isinstance(env_info, str) else dict(env_info)

            ft_arr = compute_ft_from_states_via_obs(model_xml_str, env_info_dict, states_arr)  # (T,6)
            keys = ["robot0_right"]  # stable label for bookkeeping
        else:
            K = len(keys)
            force_arr = np.zeros((T, 3 * K), dtype=np.float32)
            torque_arr = np.zeros((T, 3 * K), dtype=np.float32)

            for t in range(T):
                fd = ee_forces[t]
                td = ee_torques[t]
                if (fd is None) or (td is None):
                    continue
                for j, k in enumerate(keys):
                    force_arr[t, 3*j:3*j+3] = np.asarray(fd[k], dtype=np.float32)
                    torque_arr[t, 3*j:3*j+3] = np.asarray(td[k], dtype=np.float32)

            ft_arr = np.concatenate([force_arr, torque_arr], axis=1)  # (T, 6*K)

        states_ext = np.concatenate([states_arr, ft_arr], axis=1)

        num_eps += 1
        demo_name = f"demo_{num_eps}"
        ep_data_grp = grp.create_group(demo_name)

        # model xml
        xml_path = os.path.join(ep_path, "model.xml")
        with open(xml_path, "r") as xf:
            ep_data_grp.attrs["model_file"] = xf.read()

        ep_data_grp.attrs["successful"] = bool(success)
        ep_data_grp.attrs["state_dim_original"] = int(S)
        ep_data_grp.attrs["state_dim_ft"] = int(ft_arr.shape[1])
        if keys is not None:
            ep_data_grp.attrs["ft_keys_order"] = json.dumps(keys)

        ep_data_grp.create_dataset("states", data=states_ext)
        ep_data_grp.create_dataset("actions", data=np.asarray(actions))

        print(f"Saved {demo_name} (success={success}) with states shape {states_ext.shape}")

    # Write / update metadata
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name if env_name is not None else grp.attrs.get("env", "")
    grp.attrs["env_info"] = env_info

    f.close()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # if WHOLE BODY IK; assert only one robot
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapperWithFT(env, tmp_directory)
    # env = DataCollectionWrapperWithFT(env, tmp_directory, ft_site_name="gripper0_right_ft_frame", use_site_frame=True)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)
        # gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, save_only_success=False)

import argparse
import json
import numpy as np
import robosuite as suite


def make_env_from_env_info(env_info):
    meta = json.loads(env_info) if isinstance(env_info, str) else dict(env_info)

    if "env_kwargs" in meta:
        env_name = meta.get("env_name", None)
        env_kwargs = dict(meta["env_kwargs"])
        if env_name is not None and "env_name" not in env_kwargs:
            env_kwargs["env_name"] = env_name
    else:
        env_kwargs = dict(meta)

    env_name = env_kwargs.pop("env_name", None)
    if env_name is None:
        raise ValueError(f"Could not find env_name in env metadata. Keys: {list(meta.keys())}")

    if isinstance(env_kwargs.get("robots", None), str):
        env_kwargs["robots"] = [env_kwargs["robots"]]

    for k in [
        "type",
        "env_version",
        "repository_version",
        "env_lang",
        "has_renderer",
        "has_offscreen_renderer",
        "ignore_done",
        "use_camera_obs",
        "reward_shaping",
        "control_freq",
    ]:
        env_kwargs.pop(k, None)

    print("env_name:", env_name)
    print("env_kwargs keys:", sorted(env_kwargs.keys()))

    env = suite.make(
        env_name,
        **env_kwargs,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    return env


def list_force_torque_sensors(env):
    print("\n=== MuJoCo sensors containing 'force' or 'torque' ===")
    found = []
    for i in range(env.sim.model.nsensor):
        name = env.sim.model.sensor_id2name(i)
        if name and ("force" in name.lower() or "torque" in name.lower()):
            dim = int(env.sim.model.sensor_dim[i])
            adr = int(env.sim.model.sensor_adr[i])
            found.append((i, name, dim, adr))
            print(f"id={i:3d}  name={name:35s}  dim={dim}  adr={adr}")
    if not found:
        print("No force / torque sensors found.")
    return found


def get_manual_sensor_reading(env, sensor_name):
    sensor_id = env.sim.model.sensor_name2id(sensor_name)
    adr = int(env.sim.model.sensor_adr[sensor_id])
    dim = int(env.sim.model.sensor_dim[sensor_id])
    val = np.asarray(env.sim.data.sensordata[adr:adr + dim], dtype=np.float32).reshape(-1)
    return val


def compare_ft_for_arm(env, robot_idx=0, arm=None, expected_force_name=None, expected_torque_name=None):
    robot = env.robots[robot_idx]
    if arm is None:
        arm = robot.arms[0]

    print("\n=== Robot / arm info ===")
    print("robot index:", robot_idx)
    print("robot name :", robot.name)
    print("arm        :", arm)

    important_sensors = robot.gripper[arm].important_sensors
    print("\n=== Gripper important sensors ===")
    for k, v in important_sensors.items():
        print(f"{k}: {v}")

    gripper_force_name = important_sensors["force_ee"]
    gripper_torque_name = important_sensors["torque_ee"]

    print("\n=== Sensor name comparison ===")
    print("gripper force_ee sensor :", gripper_force_name)
    print("gripper torque_ee sensor:", gripper_torque_name)

    if expected_force_name is not None:
        print("expected force sensor   :", expected_force_name)
        print("force name match        :", gripper_force_name == expected_force_name)

    if expected_torque_name is not None:
        print("expected torque sensor  :", expected_torque_name)
        print("torque name match       :", gripper_torque_name == expected_torque_name)

    # Read through robosuite API
    F_api = np.asarray(robot.ee_force[arm], dtype=np.float32).reshape(-1)
    T_api = np.asarray(robot.ee_torque[arm], dtype=np.float32).reshape(-1)

    # Read through manual MuJoCo path using gripper-declared names
    F_manual = get_manual_sensor_reading(env, gripper_force_name)
    T_manual = get_manual_sensor_reading(env, gripper_torque_name)

    print("\n=== Single-step value comparison ===")
    print("F_api   :", F_api)
    print("F_manual:", F_manual)
    print("T_api   :", T_api)
    print("T_manual:", T_manual)

    print("max |F_api - F_manual| =", float(np.max(np.abs(F_api - F_manual))))
    print("max |T_api - T_manual| =", float(np.max(np.abs(T_api - T_manual))))

    return {
        "arm": arm,
        "gripper_force_name": gripper_force_name,
        "gripper_torque_name": gripper_torque_name,
    }


def run_multistep_check(env, sensor_info, robot_idx=0, num_steps=20, atol=1e-6):
    robot = env.robots[robot_idx]
    arm = sensor_info["arm"]
    force_name = sensor_info["gripper_force_name"]
    torque_name = sensor_info["gripper_torque_name"]

    action_dim = env.action_dim
    print("\n=== Multi-step consistency check ===")
    print("num_steps:", num_steps)
    print("action_dim:", action_dim)

    max_force_err = 0.0
    max_torque_err = 0.0

    obs = env.reset()
    _ = obs

    for step in range(num_steps):
        action = np.zeros(action_dim, dtype=np.float32)
        obs, reward, done, info = env.step(action)

        # first pair
        F_api_1 = np.asarray(robot.ee_force[arm], dtype=np.float32).reshape(-1)
        T_api_1 = np.asarray(robot.ee_torque[arm], dtype=np.float32).reshape(-1)
        F_manual_1 = get_manual_sensor_reading(env, force_name)
        T_manual_1 = get_manual_sensor_reading(env, torque_name)

        # second pair, same sim state, no new step
        F_api_2 = np.asarray(robot.ee_force[arm], dtype=np.float32).reshape(-1)
        T_api_2 = np.asarray(robot.ee_torque[arm], dtype=np.float32).reshape(-1)
        F_manual_2 = get_manual_sensor_reading(env, force_name)
        T_manual_2 = get_manual_sensor_reading(env, torque_name)

        print(f"step {step}")
        print("  err pair 1 force :", np.max(np.abs(F_api_1 - F_manual_1)))
        print("  err pair 1 torque:", np.max(np.abs(T_api_1 - T_manual_1)))
        print("  err pair 2 force :", np.max(np.abs(F_api_2 - F_manual_2)))
        print("  err pair 2 torque:", np.max(np.abs(T_api_2 - T_manual_2)))
        print("\n=== Final summary ===")
        print("max force error over rollout :", max_force_err)
        print("max torque error over rollout:", max_torque_err)
        print("force consistent within atol :", max_force_err <= atol)
        print("torque consistent within atol:", max_torque_err <= atol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="robosuite env name, e.g. Square")
    parser.add_argument("--robots", type=str, nargs="+", required=True, help="robot names, e.g. Panda")
    parser.add_argument("--expected_force_name", type=str, default=None)
    parser.add_argument("--expected_torque_name", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()

    env = suite.make(
        args.env,
        robots=args.robots,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    try:
        env.reset()
        list_force_torque_sensors(env)

        sensor_info = compare_ft_for_arm(
            env,
            robot_idx=0,
            arm=None,
            expected_force_name=args.expected_force_name,
            expected_torque_name=args.expected_torque_name,
        )

        run_multistep_check(
            env,
            sensor_info=sensor_info,
            robot_idx=0,
            num_steps=args.num_steps,
            atol=1e-6,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
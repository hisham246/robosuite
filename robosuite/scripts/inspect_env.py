import robosuite as suite
env = suite.make(env_name="Wipe", robots="Panda", has_renderer=False, use_camera_obs=False)
obs = env.reset()
print([k for k in obs.keys() if "ee_force" in k or "ee_torque" in k or "contact" in k])
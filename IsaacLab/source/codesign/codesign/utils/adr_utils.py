import torch

def init_adr_obs_act_noise(env):
    env.object_pos_bias_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.object_rot_bias_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.object_pos_bias = torch.zeros(env.num_envs, 1, device=env.device)
    env.object_rot_bias = torch.zeros(env.num_envs, 1, device=env.device)
    env.object_pos_noise_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.object_rot_noise_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.robot_bias_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.robot_bias = torch.zeros(env.num_envs, 1, device=env.device)
    env.robot_state_noise_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.arm_action_noise_width = torch.zeros(env.num_envs, 1, device=env.device)
    env.hand_action_noise_width = torch.zeros(env.num_envs, 1, device=env.device)

def update_adr_obs_act_noise(env, env_ids):
    # object noise
    num_ids = env_ids.shape[0]
    env.object_pos_bias_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("object_state_noise", "object_pos_bias") * torch.rand(num_ids, device=env.device)
    env.object_rot_bias_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("object_state_noise", "object_rot_bias") * torch.rand(num_ids, device=env.device)

    env.object_pos_bias[env_ids, 0] = env.object_pos_bias_width[env_ids, 0] * (torch.rand(num_ids, device=env.device) - 0.5)
    env.object_rot_bias[env_ids, 0] = env.object_rot_bias_width[env_ids, 0] * (torch.rand(num_ids, device=env.device) - 0.5)
    env.object_pos_noise_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("object_state_noise", "object_pos_noise") * torch.rand(num_ids, device=env.device)
    env.object_rot_noise_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("object_state_noise", "object_rot_noise") * torch.rand(num_ids, device=env.device)
    
    # robot noise
    env.robot_bias_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("robot_state_noise", "robot_bias") * torch.rand(num_ids, device=env.device)
    env.robot_bias[env_ids, 0] = env.robot_bias_width[env_ids, 0] * (torch.rand(num_ids, device=env.device) - 0.5)
    env.robot_state_noise_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("robot_state_noise", "robot_noise") * torch.rand(num_ids, device=env.device)
    env.hand_action_noise_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise") * torch.rand(num_ids, device=env.device)

def update_adr_act_noise(env, env_ids):
    # robot noise
    num_ids = env_ids.shape[0]
    env.robot_bias_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("robot_state_noise", "robot_bias") * torch.rand(num_ids, device=env.device)
    env.robot_bias[env_ids, 0] = env.robot_bias_width[env_ids, 0] * (torch.rand(num_ids, device=env.device) - 0.5)
    env.robot_state_noise_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("robot_state_noise", "robot_noise") * torch.rand(num_ids, device=env.device)
    env.hand_action_noise_width[env_ids, 0] =\
        env.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise") * torch.rand(num_ids, device=env.device)

def apply_object_wrench(env, object, object_name):
    body_ids = None # targets all bodies
    env_ids = None # targets all envs

    # Wrench tensors
    if not hasattr(env, 'wrench_object_applied_force'):
        env.wrench_object_applied_force = {}
    if not hasattr(env, 'wrench_object_applied_torque'):
        env.wrench_object_applied_torque = {}
    if object_name not in env.wrench_object_applied_force:
        env.wrench_object_applied_force[object_name] = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env.wrench_object_applied_torque[object_name] = torch.zeros(env.num_envs, 1, 3, device=env.device)

    num_bodies = object.num_bodies
    
    max_linear_accel = env.leap_adr.get_custom_param_value("object_wrench", "max_linear_accel")
    max_force = (max_linear_accel * env.object_mass).unsqueeze(2)
    max_torque = (env.object_mass * max_linear_accel * env.cfg.torsional_radius).unsqueeze(2)
    forces =\
        max_force * 2 * (torch.rand(env.num_envs, num_bodies, 3, device=env.device) - 0.5)
    torques =\
        max_torque * 2 * (torch.rand(env.num_envs, num_bodies, 3, device=env.device) - 0.5)
    
    env.wrench_object_applied_force[object_name] = torch.where(
        (env.episode_length_buf.view(-1, 1, 1) % env.cfg.wrench_trigger_every) == 0,
        forces,
        env.wrench_object_applied_force[object_name]
    )
    env.wrench_object_applied_force[object_name] = torch.where(
        env.apply_wrench[:, None, None],
        env.wrench_object_applied_force[object_name],
        torch.zeros_like(env.wrench_object_applied_force[object_name])
    )
    env.wrench_object_applied_torque[object_name] = torch.where(
        (env.episode_length_buf.view(-1, 1, 1) % env.cfg.wrench_trigger_every) == 0,
        torques,
        env.wrench_object_applied_torque[object_name]
    )
    env.wrench_object_applied_torque[object_name] = torch.where(
        env.apply_wrench[:, None, None],
        env.wrench_object_applied_torque[object_name],
        torch.zeros_like(env.wrench_object_applied_torque[object_name])
    )
    
    object.set_external_force_and_torque(
        forces=env.wrench_object_applied_force[object_name],
        torques=env.wrench_object_applied_torque[object_name],
        body_ids = body_ids,
        env_ids = env_ids
    )
    object.write_data_to_sim()
    
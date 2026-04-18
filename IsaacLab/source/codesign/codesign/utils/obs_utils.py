import torch
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_from_euler_xyz, quat_box_minus, combine_frame_transforms

def create_obs_with_history(env, curr_obs):
    #If restarted, fill with the obs across with the first observation
    idx_restarted = (env.episode_length_buf == 0)
    env.obs_hist_buf[idx_restarted,:,:] = curr_obs[idx_restarted, :].unsqueeze(-1)

    env.obs_hist_buf = torch.roll(env.obs_hist_buf, 1, 2)
    env.obs_hist_buf[:,:,1] = curr_obs 
    return env.obs_hist_buf.transpose(2,1).flatten(start_dim=1, end_dim=2).clone()

def create_obs_with_history_latency(env, curr_obs):
    idx_restarted = (env.episode_length_buf == 0)
    env.obs_hist_buf[idx_restarted,:,:] = curr_obs[idx_restarted, :].unsqueeze(-1)
    env.obs_hist_buf = torch.roll(env.obs_hist_buf, 1, 2)
    env.obs_hist_buf[:,:,1] = curr_obs 

    for i in range(0, env.cfg.obs_timesteps):
        env.output_obs_hist_buf[:,:,i] =  env.obs_hist_buf.gather(2, (env.obs_latency.unsqueeze(2) + i).long()).squeeze(2)
    return env.output_obs_hist_buf.transpose(2,1).flatten(start_dim=1, end_dim=2).clone()

def create_action_latency(env, curr_act):
    idx_restarted = (env.episode_length_buf == 0)
    env.act_hist_buf[idx_restarted,:,:] = torch.zeros_like(curr_act[idx_restarted, :].unsqueeze(-1)).to(torch.float)
    
    env.act_hist_buf = torch.roll(env.act_hist_buf, 1, 2)
    env.act_hist_buf[:,:,0] = curr_act
    env_ids = torch.arange(env.num_envs, device=curr_act.device).unsqueeze(1).expand(-1, env.cfg.action_space)
    act_ids = torch.arange(env.cfg.action_space, device=curr_act.device).unsqueeze(0).expand(env.num_envs, -1)
    latency_ids = env.act_latency.long()  

    delayed_actions = env.act_hist_buf[env_ids, act_ids, latency_ids]
    return delayed_actions
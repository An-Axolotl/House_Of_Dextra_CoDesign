# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import json
from isaaclab.app import AppLauncher

import hand_selection  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Folder to save videos.")
parser.add_argument("--video_name", type=str, default="rl-video")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--run_name", type=str, default="change_the_name_lol", help="Name of the policy run to use and export.")
parser.add_argument("--use_unexported", action="store_true", default=False, help="Use the unexported policy.")
parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to run")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to run")
parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds")

# add with your other args
parser.add_argument(
    "--metrics_out",
    type=str,
    default="",
    help="Write GHS_* metrics as JSON Lines to this file (printed to stdout as well)."
)
hand_selection.add_hand_selection_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
hand_selection.apply_hand_selection_args(args_cli)
    
    
metrics_out_path = getattr(args_cli, "metrics_out", "") or os.environ.get("GHS_METRICS_OUT", "")
if metrics_out_path:
    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)

def _emit_metrics(line: str):
    """Print to stdout and (optionally) append to metrics_out_path."""
    print(line, flush=True)
    if metrics_out_path:
        # best-effort append
        try:
            with open(metrics_out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as _e:
            # don’t crash eval if filesystem hiccups
            print(f"[warn] failed to append metrics to {metrics_out_path}: {_e}", flush=True)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import time
import torch
import numpy as np

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import codesign.tasks  # noqa: F401

def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    
    env_cfg.viewer.eye = (1.498, 1.498, 1.498)
    env_cfg.viewer.lookat = (0.413, 0.413, 0.413)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        if args_cli.video_folder is None:
            video_folder = os.path.join(log_root_path, log_dir, "videos", "play")
        else:
            video_folder = args_cli.video_folder
        
        video_kwargs = {
            "video_folder": video_folder,
            "name_prefix": args_cli.video_name,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    
    save_dir = "./jit_exports"  # Update path as needed
    if not args_cli.use_unexported:
        os.makedirs(save_dir, exist_ok=True)
    
    # Additional save directory
    real_control_dir = "../RealControl/agents"
    os.makedirs(real_control_dir, exist_ok=True)
    
    def export_models():
        """Export the RL-Games trained model to TorchScript format with RNN support."""
        import copy
         
        # Get the model from the agent
        model = agent.model
        
        print(f"\n=== Exporting RL-Games Model ===")
        
        # Get model info
        original_device = next(model.parameters()).device
        model.eval()

        print(f"Model type: {type(model)}")
        print(f"Model device: {original_device}")
        
        # Check if RNN model
        is_rnn = agent.is_rnn
        print(f"Is RNN: {is_rnn}")
        
        # Get input/output dimensions
        if isinstance(agent.obs_shape, dict):
            obs_dim = sum(np.prod(shape) for shape in agent.obs_shape.values())
            print(f"Dictionary observation space detected")
            print(f"Total observation dimensions: {obs_dim}")
        else:
            obs_dim = np.prod(agent.obs_shape)
            print(f"Observations: {obs_dim}")
        
        actions_num = agent.actions_num
        print(f"Actions: {actions_num}")
        
        # Check for input normalization
        normalize_input = agent.normalize_input
        print(f"Normalize input: {normalize_input}")
        
        if is_rnn:
            print("\n=== Creating RNN Policy Wrapper ===")

            try:
                # --- A) prep dims & dummy inputs ---
                # Get model info
                original_device = next(model.parameters()).device
                model.eval()

                # --- use a COPY for tracing; do NOT mutate agent.model ---
                model_cpu = copy.deepcopy(model).eval().to('cpu')
                if isinstance(agent.obs_shape, dict):
                    obs_dim = int(sum(np.prod(s) for s in agent.obs_shape.values()))
                else:
                    obs_dim = int(np.prod(agent.obs_shape))
                dummy_obs = torch.randn(1, obs_dim)

                # fetch rnn sizes
                net = model_cpu.a2c_network if hasattr(model_cpu, 'a2c_network') else model_cpu
                r = getattr(net, 'rnn', getattr(net, 'gru', None))
                layers = int(getattr(r, 'num_layers', 1))
                hidden_size = int(getattr(r, 'hidden_size', 256))
                hidden = torch.zeros(layers, 1, hidden_size)

                # --- B) trace a tiny head that returns (actions, next_hidden) ---
                class CoreHead(torch.nn.Module):
                    def __init__(self, core):
                        super().__init__()
                        self.core = core

                    def forward(self,
                                obs: torch.Tensor,
                                hidden: torch.Tensor,
                                seq_len: torch.Tensor,
                                is_train: torch.Tensor):
                        # build the dict INSIDE (tracing will just follow tensor ops)
                        out = self.core({
                            'obs': obs,
                            'rnn_states': [hidden],      # list created here is fine
                            'seq_length': seq_len,
                            'is_train': is_train,
                        })
                        
                        if 'mus' in out:
                            actions = out['mus']                           # deterministic means
                        elif 'mean_actions' in out:                        # some rl-games variants
                            actions = out['mean_actions']
                        elif 'actions' in out:
                            actions = out['actions']                       # fallback
                        elif 'action' in out:
                            actions = out['action']
                        else:
                            raise RuntimeError("No action/mu key in model output")
                        
                        next_h = out['rnn_states'][0]
                        return actions, next_h

                core_head = CoreHead(model_cpu).eval()
                with torch.no_grad():
                    traced_core = torch.jit.trace(
                        core_head,
                        (dummy_obs, hidden, torch.tensor(1), torch.tensor(False)),
                        check_trace=False
                    )
                    
                clip = float(agent.clip_actions) if np.isfinite(agent.clip_actions) else float("inf")

                # --- C) scripted wrapper that manages hidden state and calls traced_core ---
                class RNNPolicyWrapper(torch.nn.Module):
                    def __init__(self, traced_core, layers, hidden_size, clip=float('inf'), act_dim=None):
                        super().__init__()
                        self.core = traced_core.eval()
                        self.rnn_layers = int(layers)
                        self.rnn_hidden_size = int(hidden_size)
                        self.register_buffer("hidden_state",
                                            torch.zeros(self.rnn_layers, 1, self.rnn_hidden_size))
                        self.register_buffer("one", torch.tensor(1, dtype=torch.long))
                        self.register_buffer("false", torch.tensor(False))
                        self.clip = float(clip)
                        self.act_dim = int(act_dim) if act_dim is not None else -1

                    def forward(self, obs: torch.Tensor) -> torch.Tensor:
                        if obs.dim() == 1:
                            obs = obs.unsqueeze(0)
                        B = obs.size(0)
                        if self.hidden_state.size(1) != B:
                            self.hidden_state = self.hidden_state[:, :1, :].expand(
                                self.rnn_layers, B, self.rnn_hidden_size
                            ).contiguous()

                        actions, next_h = self.core(obs, self.hidden_state, self.one, self.false)
                        self.hidden_state = next_h
                        
                        if hasattr(self, "clip") and self.clip < float("inf"):
                            actions = torch.clamp(actions, -self.clip, self.clip)
                        
                        if self.act_dim > 0 and actions.size(-1) != self.act_dim:
                            raise RuntimeError(f"Action dim mismatch: got {actions.size(-1)}, expected {self.act_dim}")
                        
                        return actions[0] if actions.size(0) == 1 else actions
                    
                    @torch.jit.export
                    def reset_mask(self, done: torch.Tensor) -> None:
                        # done: [B] bool or [B,1]
                        if done.dim() == 2:
                            done = done.squeeze(1)
                        # JIT-safe: no kwargs, no python tuple outputs
                        idx = torch.nonzero(done).squeeze(1)
                        if idx.numel() > 0:
                            self.hidden_state.index_fill_(1, idx, 0)

                    @torch.jit.export
                    def reset_memory(self) -> None:
                        self.hidden_state.zero_()

                wrapper = RNNPolicyWrapper(traced_core, layers, hidden_size, clip=clip, act_dim=actions_num).cpu()

                # smoke test
                with torch.no_grad():
                    _ = wrapper(dummy_obs)

                print("\n=== Scripting RNN Policy ===")
                scripted_policy = torch.jit.script(wrapper)
                
                # Save to both locations
                policy_path = os.path.join(save_dir, f"rlgames_{args_cli.run_name}_policy.pt")
                scripted_policy.save(policy_path)
                print(f"✅ RNN policy exported to: {policy_path}")
                
                real_control_policy_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_policy.pt")
                scripted_policy.save(real_control_policy_path)
                print(f"✅ RNN policy also saved to: {real_control_policy_path}")

                # Save metadata to both locations
                metadata = {
                    'num_observations': obs_dim,
                    'num_actions': actions_num,
                    'normalize_input': normalize_input,
                    'is_rnn': True,
                    'rnn_layers': layers,
                    'rnn_hidden_size': hidden_size,
                    'obs_shape': agent.obs_shape,
                    'clip_actions': agent.clip_actions,
                }
                torch.save(metadata, os.path.join(save_dir, "rlgames_metadata.pt"))
                torch.save(metadata, os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_metadata.pt"))
                print("✅ Metadata saved to both locations")
                return policy_path
                
            except Exception as e:
                print(f"❌ Failed to script RNN policy: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback: save complete model to both locations
                print("\n=== Fallback: Saving Complete Model ===")
                model_path = os.path.join(save_dir, "rlgames_model_complete.pt")
                real_control_model_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_model_complete.pt")
                
                model_data = {
                    'model_state_dict': model.state_dict(),
                    'model_type': type(model).__name__,
                    'normalize_input': normalize_input,
                    'obs_shape': agent.obs_shape,
                    'actions_num': actions_num,
                    'is_rnn': True,
                    'rnn_layers': wrapper.rnn_layers if 'wrapper' in locals() else 1,
                    'rnn_hidden_size': wrapper.rnn_hidden_size if 'wrapper' in locals() else 256,
                }
                
                torch.save(model_data, model_path)
                torch.save(model_data, real_control_model_path)
                print(f"✅ Complete model saved to: {model_path}")
                print(f"✅ Complete model also saved to: {real_control_model_path}")
                return model_path
        
        else:
            print("\n=== Creating Feedforward Policy Wrapper ===")
            
            # For non-RNN models, use the simpler approach
            def policy_wrapper(obs):
                with torch.no_grad():
                    processed_obs = obs
                    
                    if normalize_input and hasattr(model, 'running_mean_std'):
                        processed_obs = model.running_mean_std(processed_obs)
                    
                    input_dict = {
                        'is_train': False,
                        'obs': processed_obs,
                    }
                    
                    result = model(input_dict)
                    
                    if 'mus' in result:
                        return result['mus']
                    elif 'logits' in result:
                        return torch.argmax(result['logits'], dim=-1)
                    else:
                        return result.get('actions', result.get('action'))
            
            # Disable gradients
            for param in model.parameters():
                param.requires_grad_(False)
            
            # Test and trace
            dummy_obs = torch.randn(1, obs_dim, device=original_device)
            
            print("\n=== Tracing Feedforward Policy ===")
            try:
                with torch.no_grad():
                    traced_policy = torch.jit.trace(policy_wrapper, dummy_obs)
                    
                    # Save to both locations
                    policy_path = os.path.join(save_dir, "rlgames_policy.pt")
                    traced_policy.save(policy_path)
                    print(f"✅ Feedforward policy exported to: {policy_path}")
                    
                    real_control_policy_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_policy.pt")
                    traced_policy.save(real_control_policy_path)
                    print(f"✅ Feedforward policy also saved to: {real_control_policy_path}")
                    
                    # Save metadata to both locations
                    metadata = {
                        'num_observations': obs_dim,
                        'num_actions': actions_num,
                        'normalize_input': normalize_input,
                        'is_rnn': False,
                        'obs_shape': agent.obs_shape,
                        'clip_actions': agent.clip_actions,
                    }
                    
                    if normalize_input and hasattr(model, 'running_mean_std'):
                        metadata['running_mean_std_state'] = model.running_mean_std.state_dict()
                    
                    metadata_path = os.path.join(save_dir, "rlgames_metadata.pt")
                    torch.save(metadata, metadata_path)
                    
                    real_control_metadata_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_metadata.pt")
                    torch.save(metadata, real_control_metadata_path)
                    
                    print(f"✅ Metadata saved to: {metadata_path}")
                    print(f"✅ Metadata also saved to: {real_control_metadata_path}")
                    
                    return policy_path
                    
            except Exception as e:
                print(f"❌ Failed to trace policy: {e}")
                import traceback
                traceback.print_exc()
                return None

    # Only export if not using unexported flag
    if not args_cli.use_unexported:
        exported_policy_path = export_models()
        exported_policy = None
        if exported_policy_path:
            try:
                exported_policy = torch.jit.load(exported_policy_path).to(agent.device)
                exported_policy.eval()
                print(f"[INFO] Using exported policy: {exported_policy_path}")
            except Exception as e:
                print(f"[warn] Failed to load exported TorchScript policy from '{exported_policy_path}': {e}")
                print("[INFO] Falling back to unexported agent policy.")
        use_exported = exported_policy is not None
    else:
        print("Skipping model export for evaluation...")
        exported_policy = None
        use_exported = False
        
    def py_cast(v):
        # tensors
        if torch.is_tensor(v):
            return float(v.item()) if v.numel() == 1 else [float(x) for x in v.flatten().tolist()]
        # numpy scalars/arrays
        if isinstance(v, (np.generic,)):
            return float(v.item())
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return float(v.item())
            return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()
        # plain types
        if isinstance(v, (float, int, bool, str)) or v is None:
            return v
        # last resort
        try:
            return float(v)
        except Exception:
            return str(v)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
        
    # tracking variables
    episode_count = 0
    step_count = 0

    if use_exported and hasattr(exported_policy, "reset_memory"):
        exported_policy.reset_memory()
        
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
        
    # BEFORE the loop
    run_start_time = time.time()

    agg_count = 0                          # number of finished episodes
    agg_sum = {}                           # sum over numeric metrics
    agg_min = {}
    agg_max = {}
        
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()

        # Check termination conditions
        if args_cli.max_episodes and episode_count >= args_cli.max_episodes:
            print(f"Reached maximum episodes: {episode_count}")
            break
        if args_cli.max_steps and step_count >= args_cli.max_steps:
            print(f"Reached maximum steps: {step_count}")
            break
        if args_cli.timeout and (time.time() - run_start_time) >= args_cli.timeout:
            print(f"Reached timeout: {time.time() - run_start_time:.1f}s", flush=True)
            # import sys; sys.exit(2)
            break
        
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)                 # [B, D] on agent.device
            if use_exported:
                actions = exported_policy(obs_t)            # [B, A]
            else:
                actions = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)

            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):        # <— add this
                obs = obs["obs"]
            
            # Aggregate/print metrics when any env finishes an episode
            dones_any = bool(len(dones) > 0 and any(dones))

            if dones_any:
                n_done = int(sum(dones))

                # pull env-aggregated (per step) metrics you already log
                extras_log = {}
                base_env = getattr(env, "env", None)
                if base_env is not None:
                    unwrapped = getattr(base_env, "unwrapped", None)
                    if unwrapped is not None and hasattr(unwrapped, "extras") and "log" in unwrapped.extras:
                        raw = unwrapped.extras["log"]
                        extras_log = {k: py_cast(v) for k, v in raw.items()}

                # only aggregate scalar numeric metrics
                for k, v in extras_log.items():
                    if isinstance(v, (int, float)):
                        val = float(v)
                        agg_sum[k] = agg_sum.get(k, 0.0) + val * n_done
                        agg_min[k] = min(agg_min.get(k, float('inf')), val)
                        agg_max[k] = max(agg_max.get(k, float('-inf')), val)
                        
                # roll up per-env episode counters into per-design accumulators
                done_env_ids = torch.nonzero(torch.as_tensor(dones), as_tuple=False).squeeze(-1).to(env.unwrapped.device)
                if done_env_ids.numel() > 0 and hasattr(env, "env") and hasattr(env.env, "unwrapped"):
                    try:
                        env.env.unwrapped._rollup_done_episodes(done_env_ids)
                    except Exception as e:
                        print(f"[warn] rollup failed: {e}")

                agg_count += n_done
                episode_count += n_done   # <-- make max_episodes work

            # Handle RNN state resets
            if use_exported and len(dones) > 0:
                d = torch.as_tensor(dones, device=actions.device, dtype=torch.bool)
                if hasattr(exported_policy, "reset_mask"):
                    exported_policy.reset_mask(d)
            elif len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            
        step_count += 1
        
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
            
    base_env = getattr(env, "env", None)
    if base_env is not None:
        unwrapped = getattr(base_env, "unwrapped", None)
        if unwrapped is not None and hasattr(unwrapped, 'force_aggregate_incomplete_episodes'):
            print("[play.py] Forcing aggregation of incomplete episodes...")
            unwrapped.force_aggregate_incomplete_episodes()
            
    # If the task already keeps rolling aggregates in extras["log"], use those
    final_avg = {}
    base_env = getattr(env, "env", None)
    if base_env is not None:
        unwrapped = getattr(base_env, "unwrapped", None)
        if unwrapped is not None and hasattr(unwrapped, "extras") and "log" in unwrapped.extras:
            raw = unwrapped.extras["log"]
            tmp = {k: py_cast(v) for k, v in raw.items()}
            final_avg = {k: float(v) for k, v in tmp.items() if isinstance(v, (int, float))}

    # Fallback to your counters only if you truly built per-episode sums yourself
    if not final_avg and agg_sum:
        final_avg = {k: (agg_sum[k] / max(1, agg_count)) for k in agg_sum.keys()}

    score = 0.0
    
     # Print per-hand design metrics if available
    hand_metrics_summary = {}
    if base_env is not None:
        unwrapped = getattr(base_env, "unwrapped", None)
        if unwrapped is not None:
            # GPU-aggregated summary
            hand_metrics_summary = unwrapped.get_hand_design_metrics_summary_gpu()

            if hand_metrics_summary:
                print(f"\n[hand-metrics] Found {len(hand_metrics_summary)} unique hand designs")
                
                # Print detailed performance for each hand design
                if hasattr(unwrapped, "print_hand_design_performance"):
                    unwrapped.print_hand_design_performance()
                
                # Print JSON format for GHS parser
                hand_metrics_json = {
                    "num_hand_designs": len(hand_metrics_summary),
                    "hand_design_metrics": hand_metrics_summary
                }
                _emit_metrics("GHS_HAND_METRICS_JSON " + json.dumps(hand_metrics_json, separators=(",", ":"), default=str))



    # Compute overall RPS across all designs for a single scalar score
    overall_score = 0.0
    overall_rps = 0.0
    if hand_metrics_summary:
        total_omega_score = 0.0
        total_designs = 0
        total_contact_omega = 0.0
        total_contact_steps = 0

        for _, metrics in hand_metrics_summary.items():
            # Prefer omega_score if available
            omega_score = metrics.get("omega_score", None)
            if omega_score is not None:
                total_omega_score += float(omega_score)
                total_designs += 1
            
            # Also accumulate raw omega data for RPS calculation
            avg_abs_omega_z = metrics.get("raw_omega_z_rad_per_sec", metrics.get("avg_abs_omega_z", 0.0))
            contact_omega = avg_abs_omega_z * metrics.get("contact_steps", 0)
            total_contact_omega += float(contact_omega)
            total_contact_steps += int(metrics.get("contact_steps", 0))
        
        if total_designs > 0:
            # Primary score: average omega_score across designs
            overall_score = total_omega_score / total_designs
        else:
            overall_score = 0.0
        
        # Secondary metric: overall RPS from raw omega data
        if total_contact_steps > 0:
            avg_omega_z = total_contact_omega / total_contact_steps
            overall_rps = avg_omega_z / (2.0 * 3.14159)  # convert rad/s to rev/s
        else:
            overall_rps = 0.0
            
    # note the above is kept just for visibility in the final payload
    # Compute overall normalized reward across designs for a single scalar score
    overall_norm = 0.0
    if hand_metrics_summary:
        tot = 0.0
        n = 0
        for _, m in hand_metrics_summary.items():
            nr = m.get("norm_reward", None)
            if nr is None:
                # fallback if norm_reward missing: compute from average_reward & reward_cap if present
                avg_r = float(m.get("average_reward", 0.0))
                cap = float(m.get("reward_cap", 5.0))
                nr = max(0.0, min(avg_r / cap, 1.0 - 1e-6))
            tot += float(nr)
            n += 1
        overall_norm = (tot / n) if n > 0 else 0.0

    # Make normalized reward the primary score
    score = float(overall_norm)

    # Also mirror into avg_metrics so the parser can find it
    try:
        final_avg["norm_reward"] = float(overall_norm)
    except Exception:
        pass

    final_payload = {
        "episode_count": int(agg_count),
        "num_envs": int(env.unwrapped.num_envs),
        "avg_metrics": final_avg,
        "min_metrics": agg_min,
        "max_metrics": agg_max,
        # PRIMARY score now normalized reward:
        "score": score,
        "norm_reward": float(overall_norm),

        # keep these for visibility (optional)
        "omega_score": float(overall_score),       # if you still computed it
        "rps_from_omega": float(overall_rps),
        "turns_per_second": float(overall_rps),
        "rps": float(overall_rps),
        "turns_per_minute": float(overall_rps * 60.0),
        "hand_design_metrics": hand_metrics_summary,
        "num_hand_designs": len(hand_metrics_summary),
    }

    _emit_metrics("GHS_METRICS_JSON_FINAL " + json.dumps(final_payload, separators=(",", ":"), allow_nan=False))
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

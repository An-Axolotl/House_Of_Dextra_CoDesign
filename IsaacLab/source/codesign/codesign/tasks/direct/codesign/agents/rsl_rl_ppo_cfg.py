# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class LeapHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 200
    experiment_name = "leap_hand_reorient"
    device = "cuda:0"
    empirical_normalization = True
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_units=256,
        rnn_layers=1,
        use_layernorm=True,
        rnn_after_mlp=False, 
        concat_rnn_input=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=1,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True
    )
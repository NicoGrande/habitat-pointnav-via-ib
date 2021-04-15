#!/bin/bash

echo "Using setup for training"
. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/pointnav/ddppo_pointnav.yaml \
    --run-type train \
    RL.PPO.start_beta 1e1 \
    RL.PPO.decay_start_step 1e8 \
    RL.PPO.beta_decay_steps 1e8 \
    RL.PPO.final_beta 1e1 \
    RL.PPO.num_steps 96 \
    RL.PPO.use_aux_losses True \
    RL.PPO.use_info_bot True \
    RL.PPO.use_odometry False \
    RL.DDPPO.pretrained_encoder True \
    RL.DDPPO.train_encoder False \
    FORCE_TORCH_SINGLE_THREADED True \
    BASE_TASK_CONFIG_PATH configs/tasks/challenge_pointnav2020.local.rgbd.yaml \
    TENSORBOARD_DIR tb/test_pointav_via_ib\
    EVAL_CKPT_PATH_DIR data/checkpoints/test_pointav_via_ib \
    CHECKPOINT_FOLDER data/checkpoints/test_pointav_via_ib \
    NUM_PROCESSES 4
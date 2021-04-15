#!/bin/bash

echo "Using setup for evaluation"
. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/pointnav/ddppo_pointnav.yaml \
    --run-type eval \
    RL.PPO.start_beta 1e1 \
    RL.PPO.decay_start_step 1e8 \
    RL.PPO.beta_decay_steps 1e8 \
    RL.PPO.final_beta 1e1 \
    RL.PPO.use_odometry False \
    BASE_TASK_CONFIG_PATH configs/tasks/challenge_pointnav2019.local.rgbd_test_scene.yaml \
    TENSORBOARD_DIR tb/test_pointav_via_ib \
    EVAL_CKPT_PATH_DIR data/checkpoints/test_pointav_via_ib \
    CHECKPOINT_FOLDER data/checkpoints/test_pointav_via_ib \
    NUM_PROCESSES 4

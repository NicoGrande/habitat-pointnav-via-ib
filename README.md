Realisic PointGoal Navigation via Information Bottleneck
==============================

![alt text](res/img/model-figure.png?raw=true)

Our code is builds on the [Habitat](https://github.com/facebookresearch/habitat-lab/) framework and thus only changes some files. Please refer to the original Habitat documentation for an overview of the framework as well as installation instructions. Furthermore, the task configurations used in this work are as defined in the [Habitat Challenge](https://github.com/facebookresearch/habitat-challenge). Specifically, we use the 2019 Habitat Challenge Setting without access to ground truth GPS+Compass and the 2020 Habitat Challenge Setting. 

## Requirements
1. Python 3.6+
2. Recommended: [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html

## Installation
To install Habitat, please follow the installation guidelines for [habitat-sim](https://github.com/facebookresearch/habitat-sim#installation) and [habitat-lab](https://github.com/facebookresearch/habitat-lab#installation). No additional packages are used in our paper. This codebase is compatible with https://github.com/facebookresearch/habitat-sim/releases/tag/v0.1.7 (we use headless and with-cuda installation)

### Data
We use the Gibson dataset as distributed by Habitat. You'll need to download the Gibson scene data from [habitat-sim](https://github.com/facebookresearch/habitat-sim#datasets) and the Gibson PointNav data from [habitat-lab](https://github.com/facebookresearch/habitat-lab#data). Download and extract these datasets as specified in the Habitat instructions.

## Training
You can train the model depicted in the paper by running `train.sh`. The training configuration file for the 2020 Habitat Challenge setting can be found at `configs/tasks/challenge_pointnav2020.local.rgbd.yaml` while the training configuration file for the 2019 Habitat Challenge setting can be found at `configs/tasks/challenge_pointnav2019.local.rgbd.yaml`. All model hyperparameters as well as the datasets used to perform the task can be found at `habitat_baselines/config/pointnav/ddppo_pointnav.yaml`. Dynamic Success Reward (DSR) can be disabled by modifying line 72 in `habitat_baselines/common/environments.py` to read `reward += self._rl_config.SUCCESS_REWARD`. In order to reproduce the ablation results presented in the paper, simply toggle the `RL.PPO.use_aux_losses` and 
 `RL.PPO.use_info_bot` boolean flags in `train.sh` as desired. 

## Evaluation
You can evaluate trained models by running `eval.sh` where you can configure the desired checkpoing by specifying a value for `EVAL_CKPT_PATH_DIR`. To generate videos of your agent, enable `VIDEO_OPTION` in `habitat_baselines/config/pointnav/ddppo_pointnav.yaml`.

## Results
As mentioned in the paper, our approach achieves important performance gains both in the semi-idealized 2019 Habitat Challenge Setting as well as the realistic 2020 Habitat Challenge Setting. The results as reported in the paper are presented below:

### Semi-Idealized Setting (2019 Habitat Challenge Setting):
- 'baseline' : 10.95 SPL | 68.30 Soft SPL | 11.47 Success | 1.987 Distance to Goal
- 'ego-localization' : 50.80 SPL | 81.30 Soft SPL | 53.50 Success | 0.959 Distance to Goal
- 'ours' : 65.59 SPL | 76.39 Soft SPL | 74.95 Success | 0.756 Distance to Goal
- 'ours + DSR' : 68.87 SPL | 78.62 Soft SPL | 75.05 Success | 0.725 Distance to Goal

### Realistic Setting (2020 Habitat Challenge Setting):
- 'baseline' : 2.83 SPL | 50.71 Soft SPL | 3.92 Success | 2.666 Distance to Goal
- 'ego-localization' | 4.70 SPL | 57.60 Soft SPL | 6.00 Success | 1.843 Distance to Goal
- 'ours' : 16.59 SPL | 55.36 Soft SPL | 23.14 Success | 1.575 Distance to Goal
- 'ours + DSR' : 19.86 SPL | 60.20 Soft SPL | 26.66 Success | 1.474 Distance to Goal

### 2020 Habitat Challenge Results:
- 'ours + DSR' : 12.20 SPL | 56.10 Soft SPL | 16.30 Success | 2.08 Distance to Goal

## References and Citation
If you use this work, you can cite it as
```
@inproceedings{grande2021ib,
    title={Realistic PointGoal Navigation via Information Bottleneck},
    author={Guillermo Grande and Erik Wijmans and Dhruv Batra},
    year={2021},
}
```

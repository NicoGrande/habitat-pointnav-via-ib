#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import Flatten
from habitat_baselines.rl.aux_losses import AuxLosses
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.transformer_state_encoder import (
    TransformerStateEncoder,
)
from habitat_baselines.rl.ppo import Net, Policy


@torch.jit.script
def _update_pg_gps_compass(pg, gps, compass):
    n = compass.size(0)

    cos = torch.cos(compass)
    sin = torch.sin(compass)
    # Pre-tranpose this matrix in row major form
    # i.e. torch.tensor([1, 2, 3, 4]) in tranposed row-major is
    # torch.tensor([1, 3, 2, 4])
    inv_rot_mat = torch.cat([cos, sin, -sin, cos], dim=compass.dim() - 1).view(n, 2, 2)

    # this will do rot_mat.inv() @ (goal_observations - gps_pred)
    pg = torch.bmm(inv_rot_mat, (pg - gps).unsqueeze(gps.dim())).squeeze(gps.dim())

    return pg


@torch.jit.script
def _update_pg_gps(pg, gps):
    return pg - gps


@torch.jit.script
def _to_mag_and_unit_vec(xy):
    rho = torch.norm(xy, dim=xy.dim() - 1, p=2, keepdim=True)
    return torch.cat([rho, xy / rho], dim=xy.dim() - 1)


@torch.jit.script
def _angle_to_unit_vec(theta):
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=theta.dim() - 1)


@torch.jit.script
def _angular_distance_loss(pred, gt):
    pred = _angle_to_unit_vec(pred)
    gt = _angle_to_unit_vec(gt)

    dot_prod = torch.einsum("...i, ...i -> ...", pred, gt)
    dot_prod = torch.clamp(dot_prod, -0.99, 0.99)

    return torch.acos(dot_prod)


def _subsampled_mean(x, p=0.2):
    x = x.view(-1)
    numel = x.size(0)
    inds = torch.randperm(numel, device=x.device)[0 : int(p * numel)]

    return torch.gather(x, dim=0, index=inds).mean()


class VIBLayer(nn.Module):
    def __init__(self, state_size, output_size, use_info_bot=True, use_odometry=False, beta=1e-6):
        super().__init__()

        self.priv_embed = nn.Linear(3, 32)

        self.encoder = nn.Sequential(
            nn.Linear(state_size + 32, 4 * output_size, bias=False),
            nn.LayerNorm(4 * output_size),
            nn.ReLU(True),
            nn.Linear(4 * output_size, 2 * output_size),
        )
        self.register_buffer("beta", torch.tensor(beta))
        self.output_size = output_size
        self.use_info_bot = use_info_bot
        self.use_odometry = use_odometry

    def forward(self, s, obs):
        if "pointgoal_with_gps" not in obs:
            return self.prior(s).sample()
        
        if self.use_odometry:
            privileged_info = obs["pointgoal_with_gps_compass"]
        else:
            privileged_info = obs["pointgoal_with_gps"]

        mu, sigma = torch.chunk(
            self.encoder(
                torch.cat(
                    [
                        s,
                        self.priv_embed(
                            _to_mag_and_unit_vec(privileged_info)
                        ),
                    ],
                    -1,
                )
            ),
            2,
            s.dim() - 1,
        )

        if not self.use_info_bot:
            mu = torch.zeros_like(mu)
            sigma = torch.ones_like(sigma)

        sigma = F.softplus(sigma)
        dist = torch.distributions.Normal(mu, sigma)

        x = dist.rsample()

        # The following code block is for running with Selective Noise Injection

        # if self.training:
        #    x = dist.rsample()
        #else:
        #    x = dist.mean


        if AuxLosses.is_active():
            AuxLosses.register_loss(
                "information",
                _subsampled_mean(
                    torch.distributions.kl_divergence(dist, self.prior(s))
                ),
                # torch.distributions.kl_divergence(dist, self.prior(s)).mean(),
                self.beta,
            )

        return x

    def prior(self, x) -> torch.distributions.Normal:
        size = list(x.size())
        size[x.dim() - 1] = self.output_size
        mu = torch.zeros(size, device=x.device, dtype=x.dtype)
        return torch.distributions.Normal(mu, torch.ones_like(mu))


class VIBCompleteLayer(VIBLayer):
    def __init__(self, state_size, output_size, use_info_bot=True, use_odometry=False, beta=1e-6):
        super().__init__(state_size, output_size, use_info_bot, use_odometry, beta)

        self.gps_head = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.ReLU(True),
            nn.Linear(state_size // 2, 2),
        )
        self.compass_head = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.ReLU(True),
            nn.Linear(state_size // 2, 1),
        )
        self.predicted_embed = nn.Sequential(nn.Linear(3, output_size))

        self.combine_layer = nn.Sequential(
            nn.Linear(output_size * 2, output_size), nn.ReLU(True)
        )
        
        self.use_odometry = use_odometry

    def forward(self, s, obs):
        priv_emb = super().forward(s, obs)

        gps = self.gps_head(s)
        compass = self.compass_head(s)

        if self.use_odometry:
            pg = obs['pointgoal_with_egomotion_prediciton']
        else:
            pg = _update_pg_gps(obs["pointgoal"], gps)

        embed_pg = self.predicted_embed(_to_mag_and_unit_vec(pg.detach()))

        if AuxLosses.is_active():
            if self.use_odometry:
                AuxLosses.register_loss(
                    "egomotion_error",
                    torch.norm(pg - obs["pointgoal_with_gps_compass"], dim=-1).mean(),
                    0.0,
                )
            else:
                AuxLosses.register_loss(
                    "egomotion_error",
                    torch.norm(pg - obs["pointgoal_with_gps"], dim=-1).mean(),
                    0.0,
                )
            AuxLosses.register_loss(
                "compass_loss",
                _subsampled_mean(_angular_distance_loss(compass, obs["compass"])),
            )
            AuxLosses.register_loss(
                "gps_loss",
                _subsampled_mean(
                    F.mse_loss(gps, obs["gps"], reduction="none").mean(-1)
                ),
            )

        return self.combine_layer(torch.cat([priv_emb, embed_pg], dim=-1))


class VBBLayer(VIBLayer):
    def __init__(self, state_size, privileged_size, output_size, use_info_bot=True, beta=1e-6):
        super().__init__(state_size, privileged_size, output_size, use_info_bot, beta)

        self.channel_cap_network = nn.Sequential(
            nn.Linear(state_size, state_size // 2, bias=False),
            nn.LayerNorm(state_size // 2),
            nn.ReLU(True),
            nn.Linear(state_size // 2, 1),
        )
        self.encoder = nn.Sequential(
            nn.Linear(state_size + privileged_size, 4 * output_size, bias=False),
            nn.LayerNorm(4 * output_size),
            nn.ReLU(True),
            nn.Linear(4 * output_size, output_size),
        )

    def _priv(self, s, g):
        return self.encoder(torch.cat([s, g], -1))

    def _d_cap(self, s):
        return self.channel_cap_network(s)

    def forward(self, s, g):
        if g is None:
            return prior(s).sample()

        priv_logits = self._priv(s, g)
        d_cap_logits = self._d_cap(s)

        d_cap = torch.sigmoid(d_cap_logits)
        priv = torch.sigmoid(priv_logits)
        x = d_cap * priv + (1 - d_cap) * prior(s).sample()

        if AuxLosses.is_active():
            priv_log_prob = F.logsigmoid(priv)
            log_d_cap = F.logsigmoid(d_cap_logits)
            AuxLosses.register_loss(
                "information",
                (
                    -d_cap * log_d_cap
                    + (1 - d_cap) * (1 + priv_log_prob)
                    - (log_d_cap + priv_log_prob)
                ).mean(),
                self.beta,
            )

        return x


class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        final_beta,
        start_beta,
        beta_decay_steps,
        decay_start_step,
        use_info_bot,
        use_odometry,
        goal_sensor_uuid="pointgoal_with_gps",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
    ):
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=False,
                use_info_bot=use_info_bot,
                use_odometry=use_odometry,
            ),
            action_space.n,
        )

        self.final_beta = final_beta
        self.start_beta = start_beta
        self.beta_decay_steps = beta_decay_steps
        self.decay_start_step = decay_start_step
        # self.period = None

    def update_ego_error_threshold(self, count_steps):
        if self.net.ib:
            return
        start_steps = 1e8
        end_steps = start_steps + 1e8
        final_thresh = 10.0
        start_thresh = 0.01
        alpha = pow(final_thresh / start_thresh, 1.0 / (end_steps - start_steps))
        if count_steps > start_steps and count_steps < end_steps:
            self.net.ego_error_threshold[...] = start_thresh * pow(
                alpha, count_steps - start_steps
            )
        elif count_steps >= end_steps:
            self.net.ego_error_threshold[...] = final_thresh
        else:
            self.net.ego_error_threshold[...] = start_thresh

    # Commented code if for running beta updating a cyclical value as opposed to exponential decay

    def update_ib_beta(self, count_steps):
        if not self.net.ib:
            return
        start_steps = self.decay_start_step
        end_steps = start_steps + self.beta_decay_steps

        # period = self.get_period(count_steps, self.period)
        # self.net.bottleneck.beta[...] = self.start_beta + (0.5 * (self.final_beta - self.start_beta) * (1 + np.cos(count_steps * np.pi / period + np.pi)))

        if count_steps > start_steps and count_steps < end_steps:
            self.net.bottleneck.beta[...] = self.start_beta * pow(
                self.final_beta / self.start_beta,
                (count_steps - start_steps) / (end_steps - start_steps),
            )
        elif count_steps >= end_steps:
            self.net.bottleneck.beta[...] = self.final_beta
        else:
            self.net.bottleneck.beta[...] = self.start_beta

    # def get_period(self, count_steps, current_period):
    #     if current_period is None:
    #         self.period = 5e6
    #         return 5e6
    #     else:
    #         if count_steps >= current_period:
    #             self.period = current_period * 2
    #             return current_period * 2
    #         return current_period


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
        normalize_visual_inputs=False,
        obs_transform=None,
    ):
        super().__init__()
        obs_transform = None
        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = self.obs_transform.transform_observation_space(
                observation_space
            )

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0:2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0:2]
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            self.initial_pool = nn.AvgPool2d(3)
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            spatial_size = tuple(int((s - 1) // 3 + 1) for s in spatial_size)
            for _ in range(self.backbone.spatial_compression_steps):
                spatial_size = tuple(int((s - 1) // 2 + 1) for s in spatial_size)

            self.output_shape = (
                self.backbone.final_channels,
                spatial_size[0],
                spatial_size[1],
            )

            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / np.prod(spatial_size))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.output_shape[0],
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            compression_shape = list(self.output_shape)
            compression_shape[0] = num_compression_channels
            self.compression_shape = tuple(compression_shape)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        if self.obs_transform:
            cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        x = torch.cat(cnn_input, dim=1)
        x = self.initial_pool(x)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        #  x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        use_info_bot,
        use_odometry,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, hidden_size)
        self._n_prev_action = self.prev_action_embedding.embedding_dim

        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._tgt_proj = nn.Linear(self._n_input_goal, hidden_size)
        self._n_input_goal = 32

        self.ib = True
        self.use_info_bot = use_info_bot
        self.use_odometry = use_odometry

        if self.ib:
            self.bottleneck = VIBCompleteLayer(self._hidden_size, self._n_input_goal, self.use_info_bot, self.use_odometry)

        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (
                        self.visual_encoder.output_shape[1]
                        * self.visual_encoder.output_shape[2]
                    )
                )
            )
            self.compression = nn.Sequential(
                resnet.BasicBlock(
                    self.visual_encoder.output_shape[0],
                    self.visual_encoder.output_shape[0],
                    1,
                ),
                resnet.BasicBlock(
                    self.visual_encoder.output_shape[0],
                    num_compression_channels,
                    1,
                    downsample=nn.Conv2d(
                        self.visual_encoder.output_shape[0], num_compression_channels, 1
                    ),
                ),
            )

            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.compression_shape),
                    self._hidden_size - self._hidden_size // 4,
                    bias=False,
                ),
                nn.LayerNorm(self._hidden_size - self._hidden_size // 4),
                nn.ReLU(True),
            )

            self.visual_flow_encoder = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.compression_shape),
                    self._hidden_size // 2,
                    bias=False,
                ),
                nn.LayerNorm(self._hidden_size // 2),
                nn.ReLU(True),
                nn.Linear(self._hidden_size // 2, self._hidden_size // 4, bias=False),
                nn.LayerNorm(self._hidden_size // 4),
                nn.ReLU(True),
            )

            self.delta_egomotion_predictor = nn.Linear(self._hidden_size // 4, 3)

        if rnn_type != "transformer":
            self.state_encoder = RNNStateEncoder(
                self._hidden_size,
                self._hidden_size,
                rnn_type=rnn_type,
                num_layers=num_recurrent_layers,
            )
        else:
            self.state_encoder = TransformerStateEncoder(
                input_size=self._hidden_size, d_model=self._hidden_size
            )

        self.goal_mem_layer = nn.Sequential(
            nn.Linear(
                self._hidden_size + (self._n_input_goal if self.ib else 0),
                self.output_size,
            ),
            nn.ReLU(True),
        )

        self.pg_with_gps_pred = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size // 2),
            nn.ReLU(True),
            nn.Linear(self._hidden_size // 2, 3),
        )

        self.train()

        self.register_buffer("ego_error_threshold", torch.tensor([[0.01]]))

    @property
    def output_size(self):
        return self._hidden_size // 2

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _get_tgt_encoding_hand_code(self, observations, gps_pred, compass_pred):
        initial_pg = observations[self.goal_sensor_uuid]
        pred_pg = self._update_pg(initial_pg, gps_pred, compass_pred)

        if "pointgoal_with_gps" in observations:
            true_pg = observations["pointgoal_with_gps"]

            error = (pred_pg - true_pg).norm(dim=-1, keepdim=True)

            if AuxLosses.is_active():
                AuxLosses.register_loss("egomotion_error", error.mean(), 0.1)

            valid_ego_preds = error.detach() < self.ego_error_threshold

            pg = torch.where(valid_ego_preds, pred_pg.detach(), true_pg)
        else:
            pg = pred_pg

        # Back-propping from the policy into the goal seems kinda odd -- doesn't make sense
        # to move the goal to make the policy more/less likely to predict a given action
        # We also have prefect supervision on what this should be!
        goal_observations = _to_mag_and_unit_vec(pg.detach())

        return self.tgt_embeding(goal_observations)

    def get_tgt_encoding(self, observations, x):
        return self.bottleneck(x, observations)

    def forward(self, observations, prev_observations, rnn_hidden_states, prev_actions, masks):
        if AuxLosses.is_active():
            AuxLosses.obs = observations

        depth_flag = False
        rgb_flag = False

        if "depth" in observations:
            depth_flag = True
        if "rgb" in observations:
            rgb_flag = True

        if "visual_features" in observations:
            visual_features = observations["visual_features"]
            prev_visual_features = observations["prev_visual_features"]
        
        elif masks.size(0) != rnn_hidden_states.size(1):
            obs_input = {}
            N = rnn_hidden_states.size(1)
            T = masks.size(0) // N

            if depth_flag:
                prev_obs = prev_observations["depth"].view(T, N, *prev_observations["depth"].size()[1:])
                obs = observations["depth"].view(T, N, *observations["depth"].size()[1:])
                obs_input["depth"] = torch.cat((prev_obs[0:1], obs), dim=0)
                obs_input["depth"] = obs_input["depth"].view((T + 1) * N, *obs_input["depth"].size()[2:])

            if rgb_flag:
                prev_obs = prev_observations["rgb"].view(T, N, *prev_observations["rgb"].size()[1:])
                obs = observations["rgb"].view(T, N, *observations["rgb"].size()[1:])
                obs_input["rgb"] = torch.cat((prev_obs[0:1], obs), dim=0)
                obs_input["rgb"] = obs_input["rgb"].view((T + 1) * N, *obs_input["rgb"].size()[2:])

            obs_features = self.visual_encoder(obs_input)
            prev_visual_features = obs_features[:T*N, :, :, :]
            visual_features = obs_features[-T*N:, :, :, :]
                
        else:
            obs_input = {}

            if depth_flag:
                obs_input["depth"] = torch.cat((prev_observations["depth"], observations["depth"]), dim=0)
            if rgb_flag:
                obs_input["rgb"] = torch.cat((prev_observations["rgb"], observations["rgb"]), dim=0)

            obs_features = self.visual_encoder(obs_input)
            prev_visual_features, visual_features = obs_features.split(obs_features.size()[0] // 2, dim=0)

        visual_features = self.compression(visual_features)

        visual_emb = self.visual_fc(visual_features)
	
	    # difference of frames (unit 1)
        flow_emb = self.visual_flow_encoder(
            (visual_features - self.compression(prev_visual_features))
            * masks.view(-1, 1, 1, 1)
        )

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        context_emb = prev_actions + self._tgt_proj(observations["pointgoal"])
        x, rnn_hidden_states = self.state_encoder(
            torch.cat([visual_emb, flow_emb], dim=-1) + context_emb,
            rnn_hidden_states,
            masks,
        )

        tgt_encoding = self.get_tgt_encoding(observations, x)

        x = torch.cat([x, tgt_encoding], dim=-1)
        x = self.goal_mem_layer(x)

        if AuxLosses.is_active():
            n = rnn_hidden_states.size(1)
            t = int(x.size(0) / n)

            delta_ego = self.delta_egomotion_predictor(flow_emb).view(t, n, 3)
            gps_gt = observations["gps"].view(t, n, 2)
            compass_gt = observations["compass"].view(t, n, 1)
            masks = masks.view(t, n, 1)

            gt_delta = gps_gt[1:] - gps_gt[:-1]
            gt_delta = _update_pg_gps_compass(
                gt_delta.view((t - 1) * n, 2),
                torch.zeros_like(gt_delta).view((t - 1) * n, 2),
                compass_gt[:-1].view((t - 1) * n, 1),
            ).view(t - 1, n, 2)
            AuxLosses.register_loss(
                "delta_gps",
                _subsampled_mean(
                    torch.masked_select(
                        F.mse_loss(
                            delta_ego[1:, :, 0:2], gt_delta, reduction="none"
                        ).mean(dim=-1),
                        masks[1:, :, 0].bool(),
                    )
                ),
            )

            AuxLosses.register_loss(
                "delta_compass",
                _subsampled_mean(
                    torch.masked_select(
                        _angular_distance_loss(
                            delta_ego[1:, :, 2:], compass_gt[1:] - compass_gt[:-1]
                        ),
                        masks[1:, :, 0].bool(),
                    )
                ),
            )

        return x, rnn_hidden_states
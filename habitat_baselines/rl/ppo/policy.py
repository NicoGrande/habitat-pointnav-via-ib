#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.aux_losses import AuxLosses
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.supervise_stop = False

        if self.supervise_stop:
            self.non_stop_action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions - 1
            )

            self.stop_action_distribution = CategoricalNet(self.net.output_size, 2)
        else:
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )

        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self, observations, prev_observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ):
        features, rnn_hidden_states = self.net(
            observations, prev_observations, rnn_hidden_states, prev_actions, masks
        )

        value = self.critic(features)

        if self.supervise_stop:

            stop_distribution = self.stop_action_distribution(features)
            non_stop_distribution = self.non_stop_action_distribution(features)
            if deterministic:
                stop = stop_distribution.mode()
                non_stop = non_stop_distribution.mode()
            else:
                stop = stop_distribution.sample()
                non_stop = non_stop_distribution.sample()

            action = torch.where(stop == 1, torch.zeros_like(stop), non_stop + 1)
            action_log_probs = torch.where(
                action == 0,
                stop_distribution.log_probs(torch.full_like(action, 1)),
                stop_distribution.log_probs(torch.full_like(action, 0))
                + non_stop_distribution.log_probs(
                    torch.max(action - 1, torch.zeros_like(action))
                ),
            )
        else:
            action_distribution = self.action_distribution(features)

            if deterministic:
                action = action_distribution.mode()
            else:
                action = action_distribution.sample()

            action_log_probs = action_distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, prev_observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, prev_observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(
        self, observations, prev_observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, _ = self.net(observations, prev_observations, rnn_hidden_states, prev_actions, masks)
        value = self.critic(features)

        if self.supervise_stop:
            stop_distribution = self.stop_action_distribution(features)
            non_stop_distribution = self.non_stop_action_distribution(features)

            action_log_probs = torch.where(
                action == 0,
                stop_distribution.log_probs(torch.full_like(action, 1)),
                stop_distribution.log_probs(torch.full_like(action, 0))
                + non_stop_distribution.log_probs(
                    torch.max(action - 1, torch.zeros_like(action))
                ),
            )

            distribution_entropy = (
                -1.0
                * (
                    stop_distribution.probs[:, -1] * stop_distribution.logits[:, -1]
                    + (
                        stop_distribution.probs[:, 0:1]
                        * non_stop_distribution.probs
                        * (
                            stop_distribution.logits[:, 0:1]
                            + non_stop_distribution.logits
                        )
                    ).sum(-1)
                ).mean()
            )

            stop_loss = F.cross_entropy(
                stop_distribution.logits,
                observations["stop_oracle"].long().squeeze(-1),
                weight=torch.tensor(
                    [1.0, 1.0 / np.sqrt(100.0)], device=features.device
                ),
            )

            AuxLosses.register_loss("stop_loss", stop_loss)
        else:
            action_distribution = self.action_distribution(features)

            action_log_probs = action_distribution.log_probs(action)
            distribution_entropy = action_distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(
        self, observation_space, action_space, goal_sensor_uuid, hidden_size=512
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
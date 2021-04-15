import numpy as np

import torch
import torch.nn as nn

from .resnet_encoder_odometer import ResNetEncoderForOdometer

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg
        self.repr_size = self.cfg.MODEL.REPR_SIZE

        self.input_modality = cfg.MODEL.INPUT_MODALITY
        self.cnn_in_channels = 0
        
        self.is_depth, self.is_rgb = False, False
        if "rgb" in self.input_modality.split(","):
            self.cnn_in_channels += 3
            self.is_rgb = True
        if "depth" in self.input_modality.split(","):
            self.cnn_in_channels += 1
            self.is_depth = True

        self.is_action_input = "action" in self.input_modality.split(",")

        self.cnn_dims = (360,640)

        self.cnn = ResNetEncoderForOdometer(
            input_dims=self.cnn_dims,
            input_channels=self.cnn_in_channels,
            repr_size=self.repr_size,
            baseplanes=32,
            ngroups=(32 // 2),
        )

        self.task_head_input_size = self.repr_size
        if self.is_action_input:
            self.task_head_input_size += self.cfg.MODEL.ACTION_EMBEDDING_SIZE

            self.action_embedding = nn.Embedding(
                num_embeddings=self.cfg.MODEL.N_ACTIONS,
                embedding_dim=self.cfg.MODEL.ACTION_EMBEDDING_SIZE
            )
        
        self.is_action = "action" in self.cfg.TASK.NAME
        self.is_egomotion = "egomotion" in self.cfg.TASK.NAME
        if self.is_action:
            self.classifier = nn.Sequential(
                nn.Linear(
                    self.task_head_input_size,
                    self.cfg.TASK.NUM_CLASSES
                )
            )
        
        if self.is_egomotion:
            ego_head_layers = [
                nn.Linear(
                    self.task_head_input_size,
                    self.cfg.TASK.NUM_REGRESSION_TARGETS
                )
            ]
            if self.cfg.DATA.TYPE in [
                "random",
                "random2",
                "random-noisy",
                "trajectory",
                "trajectory2",
                "trajectory-noisy",
                "trajectory-staticPG-noisy",
                "trajectory-egoPG-noisy",
                "random_sampling",
                "trajectory_sampling"
            ]:
                ego_head_layers.append(nn.Tanh())

            self.regressor = nn.Sequential(*ego_head_layers)
        
        self.layer_init()
    
    def layer_init(self):
        if self.is_action:
            for layer in self.classifier:
                nn.init.orthogonal_(layer.weight, 0.01)
                nn.init.constant_(layer.bias, val=0)
        
        if self.is_egomotion:
            for layer in self.regressor:
                if "weight" in layer._parameters:
                    nn.init.orthogonal_(layer.weight, 0.01)
                if "bias" in layer._parameters:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, batch):
        x = []
        features = self.cnn(batch)
        x += [features]

        if self.is_action_input:
            action_embds = self.action_embedding(batch["targets"]["action"])
            x += [action_embds]

        task_head_input = torch.cat(x, dim=1)

        logits, egomotion_preds = None, None
        if self.is_action:
            logits = self.classifier(task_head_input)
        if self.is_egomotion:
            egomotion_preds = self.regressor(task_head_input)
        
        return {
            "logits": logits,
            "egomotion_preds": egomotion_preds
        }


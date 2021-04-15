import numpy as np

import torch
import torch.nn as nn

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

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        self.cnn_dims = (256,256)

        # computing the spatial dim of the final feature volume
        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                self.cnn_dims = self._conv_output_dim(
                    dimension=self.cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )
        
        self.cnn = self.create_base_network()

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
                "trajectory-noisy"
            ]:
                ego_head_layers.append(nn.Tanh())

            self.regressor = nn.Sequential(*ego_head_layers)
        
        self.layer_init()
  
    def create_base_network(self):
        
        return nn.Sequential(
            nn.Conv2d(
                in_channels=(2 * self.cnn_in_channels),
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            Flatten(),
            nn.Linear(32 * self.cnn_dims[0] * self.cnn_dims[1], self.repr_size),
            nn.ReLU(),
        )

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)
    
    def layer_init(self):
        
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                nn.init.constant_(layer.bias, val=0)
        
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
        
        source_input, target_input = [], []
        if self.is_rgb:
            source_images = batch["source_images"]
            target_images = batch["target_images"]
            source_input += [source_images]
            target_input += [target_images]
        if self.is_depth:
            source_depth_maps = batch["source_depth_maps"]
            target_depth_maps = batch["target_depth_maps"]
            source_input += [source_depth_maps]
            target_input += [target_depth_maps]

        concat_source_input = torch.cat(source_input, 1)
        concat_target_input = torch.cat(target_input, 1)
        input = torch.cat(
            [
                concat_source_input,
                concat_target_input
            ],
            1
        )

        x = []
        features = self.cnn(input)
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
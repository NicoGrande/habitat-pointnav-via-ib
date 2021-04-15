import random

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.aux_losses import AuxLosses
from habitat_baselines.rl.models.transformer_xl import GELU, Swish, TransformerXL


class TransformerStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_layer: int = 6,
        d_head: int = 64,
        d_model: int = 256,
        d_inner: int = 256,
        layer_drop: float = 0.0,
        dropout: float = 0.0,
        dropatt: float = 0.0,
        self_sup: bool = False,
        max_self_sup_K: int = 30,
    ):

        super().__init__()
        assert input_size == d_model
        n_head = d_model // d_head

        self.transformer = jit.script(
            TransformerXL(
                n_layer=n_layer,
                n_head=n_head,
                d_model=d_model,
                d_head=d_head,
                d_inner=d_inner,
                dropout=dropout,
                dropatt=dropatt,
                layer_drop=layer_drop,
            )
        )

        self.max_self_sup_K = max_self_sup_K
        self.self_sup = self_sup
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2, bias=False),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(True),
            nn.Linear(d_model // 2, 1),
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.transformer.named_parameters():
            if "weight" in name and len(param.size()) >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def single_forward(self, x, context, mems, masks):
        r"""Forward for a non-sequence input
        """
        mems = mems * masks.view(1, -1, 1, 1)
        x, mems, _ = self.transformer(x.unsqueeze(0), context.unsqueeze(0), mems)
        x = x.squeeze(0)

        return x, mems

    def seq_forward(self, x, context, mems, masks):
        r"""Forward for a sequence of length T
        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = mems.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1))
        context = context.view(t, n, context.size(1))
        masks = masks.view(t, n)

        if self.self_sup:
            ep_lens = []
            for i in range(n):
                last_zero = 0
                has_zeros = (
                    (masks[1:-1, i] == 0.0).nonzero().squeeze(-1).cpu().unbind(0)
                )
                for z in has_zeros:
                    z = z.item() + 1
                    ep_lens.append(z - last_zero)
                    last_zero = z

                ep_lens.append(t - last_zero)

            k = random.randint(
                1,
                max(min(self.max_self_sup_K, int(0.8 * np.mean(np.array(ep_lens)))), 2),
            )
        else:
            k = None

        content, mems, query = self.transformer.transformer_seq_forward(
            x, context, mems, masks, two_stream_k=k
        )

        if self.self_sup:
            positives = x
            negatives = []
            for _ in range(3):
                negative_inds = torch.randperm(t * n, device=x.device)
                negatives.append(
                    torch.gather(
                        x.view(t * n, -1),
                        dim=0,
                        index=negative_inds.view(t * n, 1).expand(t * n, x.size(-1)),
                    ).view(t, n, -1)
                )

            negatives = torch.stack(negatives, dim=-1)

            positives = torch.einsum("...i, ...i -> ...", positives, query)
            negatives = torch.einsum("...ik, ...i -> ...k", negatives, query)
            cpc_logits = torch.stack([positives.unsqueeze(-1), negatives], dim=-1)

            valid_modeling_queries = torch.ones(
                t, n, device=query.device, dtype=torch.bool
            )
            valid_modeling_queries[0:k] = 0
            for i in range(n):
                has_zeros_batch = (
                    (masks[:, i] == 0.0).nonzero().squeeze(-1).cpu().unbind(0)
                )
                for z in has_zeros_batch:
                    valid_modeling_queries[n : n + k, i] = 0

            cpc_loss = torch.masked_select(
                F.cross_entropy(
                    cpc_logits,
                    torch.zeros(t, n, dtype=torch.long, device=cpc_logits.device),
                    reduction="none",
                ),
                valid_modeling_queries,
            ).mean()

            AuxLosses.register_loss("CPC|A", cpc_loss, 0.2)

        content = content.view(t * n, -1)

        return content, mems

    def forward(self, x, context, mems, masks):
        if x.size(0) == mems.size(1):
            return self.single_forward(x, context, mems, masks)
        else:
            return self.seq_forward(x, context, mems, masks)

    def initial_hidden(self, bsz):
        return self.transformer.init_mems(bsz)

    @property
    def num_recurrent_layers(self):
        return 1
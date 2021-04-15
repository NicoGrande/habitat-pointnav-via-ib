from typing import List, Optional

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class IdentityResidualGate(nn.Module):
    def __init__(self, d_model=None):
        super().__init__()

    def forward(self, x, y):
        return x + F.relu(y)


class OutputResidualGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.zb = nn.Parameter(torch.full((d_model,), -1.0))
        self.W = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, y):
        z = self.W(x) - self.zb
        z = torch.sigmoid(z)

        return x + z * F.relu(y)


class GRUResidualGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.zb = nn.Parameter(torch.full((d_model,), 2.0))
        self.W = nn.Linear(d_model, 3 * d_model, bias=False)
        self.U = nn.Linear(d_model, 3 * d_model, bias=False)

    def forward(self, x, y):
        y = F.relu(y)

        rx, zx, gx = torch.chunk(self.U(x), 3, x.dim() - 1)
        ry, zy, gy = torch.chunk(self.W(y), 3, y.dim() - 1)

        r = torch.sigmoid(rx + ry)
        z = torch.sigmoid(zx + zy - self.zb)
        h = torch.tanh(gy + r * gx)

        return (1 - z) * x + h * y


GateType = IdentityResidualGate


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.einsum("t,d->td", pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)

        return pos_emb


class PositionwiseFF(nn.Module):
    d_model: Final[int]
    d_inner: Final[int]
    dropout: Final[float]

    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.residual_gate = GateType(d_model=d_model)

    def forward(self, inp):
        ##### layer normalization + positionwise feed-forward
        core_out = self.CoreNet(self.layer_norm(inp))

        output = self.residual_gate(inp, core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    n_head: Final[int]
    d_model: Final[int]
    d_head: Final[int]
    dropout: Final[float]
    scale: Final[float]

    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len=None, ext_len=None
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=True)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=True)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.residual_gate = GateType(d_model=d_model)

    def _rel_shift(self, x, zero_triu: bool = False):
        pad_size = list(x.size())
        pad_size[1] = 1
        zero_pad = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        view_size = list(x.size())
        view_size[0] = x.size(1) + 1
        view_size[1] = x.size(0)
        x_padded = x_padded.view(view_size)

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def attn_forward(self, x, r, q, k, v, mask):
        AC = torch.einsum("ibnd,jbnd->ijbn", q, k)  # qlen x klen x bsz x n_head
        BD = torch.einsum("ibnd,jnd->ijbn", q, r)
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        attn_score = attn_score.float().masked_fill(mask.unsqueeze(-1), -float("inf"))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1).type_as(x)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        output = self.residual_gate(x, attn_out)

        return output

    def forward(
        self,
        content,
        r,
        q_bias,
        content_attn_mask,
        mems: Optional[torch.Tensor],
        query: Optional[torch.Tensor],
        query_attn_mask: Optional[torch.Tensor],
    ):
        qlen, bsz = content.size(0), content.size(1)

        content = self.layer_norm(content)

        if mems is not None:
            mems = self.layer_norm(mems)
            cat = torch.cat([mems, content], 0)
        else:
            cat = content

        content_k, content_v = torch.chunk(self.kv_net(cat), 2, dim=cat.dim() - 1)

        if query is not None:
            assert query_attn_mask is not None
            query = self.layer_norm(query)

            x = torch.cat([content, query], 0)
            mask = torch.cat([content_attn_mask, query_attn_mask], 0)

            qlen = x.size(0)
        else:
            x = content
            mask = content_attn_mask

        q = self.q_net(x)
        r = self.r_net(r)

        klen = content_k.size(0)

        q = q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        content_k = content_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # klen x bsz x n_head x d_head
        content_v = content_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # klen x bsz x n_head x d_head
        r = r.view(klen, self.n_head, self.d_head)

        q = q + q_bias
        attn_out = self.attn_forward(x, r, q, content_k, content_v, mask)

        if query is not None:
            content, query = torch.chunk(attn_out, 2, 0)
        else:
            content = attn_out

        return content, query


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(
        self,
        content,
        x_pos_seq_emb,
        q_bias,
        content_attn_mask,
        mems: Optional[torch.Tensor],
        query: Optional[torch.Tensor],
        query_attn_mask: Optional[torch.Tensor],
    ):
        content, query = self.dec_attn(
            content,
            x_pos_seq_emb,
            q_bias,
            content_attn_mask,
            mems,
            query,
            query_attn_mask,
        )
        if query is not None:
            content_query = torch.cat([content, query], 0)
            content_query = self.pos_ff(content_query)
            content, query = torch.chunk(content_query, 2, 0)
        else:
            content = self.pos_ff(content)

        return content, query


class TransformerBase(nn.Module):
    d_model: Final[int]
    n_head: Final[int]
    d_head: Final[int]
    n_layer: Final[int]
    layers: Final[nn.ModuleList]

    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        layer_drop: float = 0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.register_buffer("layer_drop", torch.tensor([layer_drop]))

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.layers = nn.ModuleList(
            [
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout, dropatt=dropatt
                )
                for i in range(n_layer)
            ]
        )

        self._create_params()

        self._post_constructor()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def _post_constructor(self):
        pass


class QueryEmbTransformer(TransformerBase):
    __constants__ = ["d_model", "n_head", "d_head", "n_layer", "layers"]

    def _build_attn_mask(self, x):
        t = x.size(0)
        return torch.zeros(t, t, device=x.device, dtype=torch.bool).view(t, t, 1)

    def _inner_forward(self, content, x_pos_seq_emb, content_attn_mask):
        for layer in self.layers:
            if (
                not self.training
                or (self.layer_drop == 0)
                or (
                    torch.rand(1, device=content.device, dtype=content.dtype)
                    > self.layer_drop
                )
            ):
                content, _ = layer(
                    content,
                    x_pos_seq_emb,
                    self.bias,
                    content_attn_mask=content_attn_mask,
                    mems=None,
                    query=None,
                    query_attn_mask=None,
                )
                content = self.drop(content)

        return content

    def forward(self, x):
        t, n, _ = x.size()

        content_attn_mask = self._build_attn_mask(x)

        x_pos_seq = torch.arange(t - 1, -1.0, -1.0, device=x.device, dtype=x.dtype)
        x_pos_seq_emb = self.pos_emb(x_pos_seq)

        x_pos_seq_emb = self.drop(x_pos_seq_emb)
        content = self.drop(x)

        content = self._inner_forward(content, x_pos_seq_emb, content_attn_mask)

        return content


class TransformerXL(TransformerBase):
    def _post_constructor(self):
        self.default_vector = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.01)
        self.default_pos_emb = nn.Parameter(torch.randn(1, self.d_model) * 0.01)
        self.query_transformer = QueryEmbTransformer(
            n_layer=2,
            d_model=self.d_model,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_model,
            dropout=0.0,
            dropatt=0.0,
            layer_drop=0.0,
        )

    def init_mems(self, bsz):
        mems = []
        param = next(self.parameters())
        for i in range(self.n_layer + 1):
            empty = torch.zeros(
                (1, bsz, self.d_model), dtype=param.dtype, device=param.device
            )
            mems.append(empty)

        return torch.stack(mems, 3)

    def _build_attn_mask(self, x, mems, two_stream_k: Optional[int] = None):
        t, n, _ = x.size()

        mlen = mems.size(0)
        klen = mlen + t
        attn_mask = torch.triu(
            torch.ones(t, klen, device=x.device, dtype=torch.bool),
            diagonal=1 + mlen - (two_stream_k if two_stream_k is not None else 0),
        )
        # Block the default vector from the content stream.  The content stream will never have
        # (content_attn_mask[t] == 0.0).all() for some t
        # The query stream can see the default vector, however
        attn_mask = torch.cat(
            [
                torch.ones(t, 1, device=x.device, dtype=torch.bool)
                if two_stream_k is None
                else torch.zeros(t, 1, device=x.device, dtype=torch.bool),
                attn_mask,
            ],
            dim=1,
        )

        return attn_mask.view(t, klen + 1, 1)

    @jit.export
    def _inner_forward(
        self,
        content,
        x_pos_seq_emb,
        mems,
        content_attn_mask,
        query: Optional[torch.Tensor] = None,
        query_attn_mask: Optional[torch.Tensor] = None,
    ):
        n = content.size(1)
        hids = [content]

        x_pos_seq_emb = torch.cat([self.default_pos_emb, x_pos_seq_emb], dim=0)
        i = 0
        for layer in self.layers:
            if (
                self.training
                and (self.layer_drop > 0)
                and (
                    torch.rand(1, device=content.device, dtype=content.dtype)
                    < self.layer_drop
                )
            ):
                hids.append(torch.zeros_like(content))
            else:
                layer_mems = mems[..., i]
                # Add the default vector to the memories.  The default vector can always been seen.  It keeps
                # NaNs from happening when (attn_mask[t] == 0.0).all() for some t
                layer_mems = torch.cat(
                    [self.default_vector.expand(1, n, self.d_model), layer_mems], dim=0
                )

                content, query = layer(
                    content,
                    x_pos_seq_emb,
                    self.bias,
                    content_attn_mask=content_attn_mask,
                    mems=layer_mems,
                    query=query,
                    query_attn_mask=query_attn_mask,
                )
                hids.append(content)

            content = self.drop(content)
            if query is not None:
                query = self.drop(query)

            i = i + 1

        return content, hids, query

    def _build_query(self, x, two_stream_k: int):
        out = torch.ones_like(x)
        t, n, d = x.size()

        qx = torch.zeros(
            two_stream_k, t - two_stream_k + 1, n, d, device=x.device, dtype=x.dtype
        )
        for i in range(two_stream_k):
            qx[i] = x[i : t - two_stream_k + i + 1]

        qx = qx.view(two_stream_k, (t - two_stream_k + 1) * n, d)
        qx = self.query_transformer(qx)

        qx = qx.view(two_stream_k, t - two_stream_k + 1, n, d)
        qx = qx[two_stream_k - 1]

        out[two_stream_k - 1 :] = qx

        return out

    def forward(self, x, context, mems, two_stream_k: Optional[int] = None):
        t, n, _ = x.size()

        mlen = mems.size(0)
        content_attn_mask = self._build_attn_mask(x, mems, None)

        klen = mlen + t
        x_pos_seq = torch.arange(klen - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        x_pos_seq_emb = self.pos_emb(x_pos_seq)

        x_pos_seq_emb = self.drop(x_pos_seq_emb)
        content = self.drop(x + context)

        if two_stream_k is not None and two_stream_k > 0:
            query_attn_mask = self._build_attn_mask(x, mems, two_stream_k)
            query = self._build_query(context, two_stream_k)
        else:
            query_attn_mask = None
            query = None

        content, hids, query = self._inner_forward(
            content, x_pos_seq_emb, mems, content_attn_mask, query, query_attn_mask
        )

        new_mems = torch.stack(hids, 3)

        new_mems = torch.cat([mems, new_mems], 0)

        return content, new_mems, query

    def transformer_seq_forward(
        self, x, context, mems, masks, two_stream_k: Optional[int]
    ):
        t, n = masks.size()
        mlen = mems.size(0)

        content_attn_mask = self._build_attn_mask(x, mems, None).repeat(1, 1, n)

        if two_stream_k is not None:
            query_attn_mask = self._build_attn_mask(x, mems, two_stream_k).repeat(
                1, 1, n
            )
        else:
            query_attn_mask = None

        # Now mask out attention to properly respect episode boundaries!
        # This will be different for each batch, so gotta loop over batches
        # jit to the rescue!
        for i in range(n):
            has_zeros_batch = (masks[:, i] == 0.0).nonzero().squeeze(-1).cpu().unbind(0)
            for z in has_zeros_batch:
                # The 1: here handles the ability to always see the default vector
                # (okay, only the query can see that, but whatevs)
                content_attn_mask[z.item() :, 1 : z.item(), i] = 1

                if query_attn_mask is not None:
                    query_attn_mask[z.item() :, 1 : z.item(), i] = 1

        klen = mlen + t
        x_pos_seq = torch.arange(klen - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        x_pos_seq_emb = self.drop(self.pos_emb(x_pos_seq))

        content = self.drop(x + context)

        if two_stream_k is not None:
            query = self._build_query(context, two_stream_k)
        else:
            query = None

        content, hids, query = self._inner_forward(
            content, x_pos_seq_emb, mems, content_attn_mask, query, query_attn_mask
        )

        new_mems = torch.stack(hids, 3)
        new_mems = torch.cat([mems, new_mems], 0)

        return content, new_mems, query


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")

    parser.add_argument("--n_layer", type=int, default=1, help="")
    parser.add_argument("--n_head", type=int, default=1, help="")
    parser.add_argument("--d_head", type=int, default=1, help="")
    parser.add_argument("--d_model", type=int, default=4, help="")
    parser.add_argument("--d_inner", type=int, default=2, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--cuda", action="store_true", help="")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cpu")

    tgt_len = 16
    B = 4
    inp_data = torch.randn((tgt_len, B, args.d_model))
    context = torch.randn((tgt_len, B, args.d_model))

    model = jit.script(
        TransformerXL(
            args.n_layer,
            args.n_head,
            args.d_model,
            args.d_head,
            args.d_inner,
            args.dropout,
            dropatt=args.dropout,
        )
    ).to(device)

    mems = model.init_mems(B)

    print(sum(p.numel() for p in model.parameters()))
    for _ in range(2):
        out, mems, query_out = model.transformer_seq_forward(
            inp_data, context, mems, torch.randint(1, (tgt_len, B)), 1
        )
        print(out[0])
        print(query_out[0])

        print(torch.norm(out - query_out, dim=-1).mean())
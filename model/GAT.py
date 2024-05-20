import torch
import torch.nn as nn
import enum
from torch.nn import init
from typing import Callable, Optional


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats: torch.Tensor,
            out_feats: torch.Tensor,
            num_heads: int,
            dropout: float,
            thred: float,
            residual: bool = True,
            activation: Optional[Callable] = nn.ELU(),
            bias: bool = True
    ):
        super(GATConv, self).__init__()
        self.num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.thred = thred

        self.linear = nn.Linear(in_feats, num_heads * out_feats, False)

        self.scoring_fn_target = nn.Parameter(
            torch.Tensor(1, num_heads, out_feats))
        self.scoring_fn_source = nn.Parameter(
            torch.Tensor(1, num_heads, out_feats))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_parameter("bias", None)

        if residual:
            if self._in_feats != self.num_heads * self._out_feats:
                self.res_proj = nn.Linear(
                    in_feats, num_heads * out_feats, False)
            else:
                self.res_proj = nn.Identity()

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(-1)
        self._activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.linear.weight)
        init.xavier_uniform_(self.scoring_fn_target)
        init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, num_nodes = x.shape[:2]
        assert adj.shape == (b, num_nodes, num_nodes), \
            f"Expected connectivity matrix with" \
            f"shape=({num_nodes}, {num_nodes}), got shape={adj.shape[1:]}"

        x = self.dropout(x)
        input = x.clone()

        x = self.linear(x).view(-1, self.num_heads, self._out_feats)
        x = self.dropout(x)

        scores_src = torch.sum((x * self.scoring_fn_source), -1, True)
        scores_dst = torch.sum((x * self.scoring_fn_target), -1, True)
        scores_src = scores_src.view(b, num_nodes, self.num_heads, 1)
        scores_dst = scores_dst.view(b, num_nodes, self.num_heads, 1)
        scores_src = scores_src.transpose(1, 2)
        scores_dst = scores_dst.permute(0, 2, 3, 1)

        all_scores = self.leakyReLU(scores_src + scores_dst)
        connectivity_mask = torch.where(
            torch.lt(adj, self.thred), -torch.inf, 0.)
        att = self.softmax(all_scores + connectivity_mask.unsqueeze(1))

        x = x.view(b, num_nodes, self.num_heads, self._out_feats)
        x = torch.matmul(att, x.transpose(1, 2))
        x = x.permute(0, 2, 1, 3)

        if not x.is_contiguous():
            x = x.contiguous()

        x = x.view(b, num_nodes, self.num_heads * self._out_feats)
        x += self.res_proj(input)

        if self.bias is not None:
            x += self.bias

        if self._activation is not None:
            x = self._activation(x)
        return x

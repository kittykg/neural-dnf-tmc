from enum import Enum
from typing import List

import torch
from torch import nn, Tensor


class SemiSymbolicLayerType(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"


class SemiSymbolic(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super(SemiSymbolic, self).__init__()

        self.layer_type = layer_type

        self.in_features = in_features  # P
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if weight_init_type == "normal":
            nn.init.normal_(self.weights, mean=0.0, std=0.1)
        else:
            nn.init.uniform_(self.weights, a=-6, b=6)
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        abs_weight = torch.abs(self.weights)
        # abs_weight: Q x P
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # max_abs_w: Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: Q
        if self.layer_type == SemiSymbolicLayerType.CONJUNCTION:
            bias = max_abs_w - sum_abs_w
        else:
            bias = sum_abs_w - max_abs_w
        # bias: Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum


class DNF(nn.Module):
    conjunctions: SemiSymbolic
    disjunctions: SemiSymbolic

    def __init__(
        self,
        num_preds: int,
        num_conjuncts: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super(DNF, self).__init__()

        self.conjunctions = SemiSymbolic(
            in_features=num_preds,  # P
            out_features=num_conjuncts,  # Q
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight: Q x P

        self.disjunctions = SemiSymbolic(
            in_features=num_conjuncts,  # Q
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
        )  # weight R x Q

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x Q
        conj = nn.Tanh()(conj)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R

        return disj

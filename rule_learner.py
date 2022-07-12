from typing import List, OrderedDict

import torch
from torch import Tensor

from dnf_layer import DNF


class DNFClassifier:
    conj_weight_mask: Tensor
    disj_weight_mask: Tensor
    dnf: DNF

    def __init__(
        self,
        num_preds: int,  # P
        num_conjuncts: int,  # Q
        num_classes: int,  # R
        delta: float = 1.0,
        weight_init_type: str = "normal",
    ) -> None:
        super(DNFClassifier, self).__init__()

        self.dnf = DNF(
            num_preds, num_conjuncts, num_classes, delta, weight_init_type
        )

        self.conj_weight_mask = torch.ones(
            self.dnf.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = torch.ones(
            self.dnf.disjunctions.weights.data.shape
        )

    def set_delta_val(self, delta_val: float) -> None:
        self.dnf.conjunctions.delta = delta_val
        self.dnf.disjunctions.delta = delta_val

    def update_weight_wrt_mask(self) -> None:
        self.dnf.conjunctions.weights.data *= self.conj_weight_mask
        self.dnf.disjunctions.weights.data *= self.disj_weight_mask

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        out = self.dnf(input)
        # out: N x R
        return out
    

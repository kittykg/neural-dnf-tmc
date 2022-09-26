import torch
from torch import nn, Tensor

from dnf_layer import DNF


class DNFClassifier(nn.Module):
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


class DoubleDNFClassifier(nn.Module):
    conj_weight_mask1: Tensor
    disj_weight_mask1: Tensor
    conj_weight_mask2: Tensor
    disj_weight_mask2: Tensor
    dnf1: DNF
    dnf2: DNF

    def __init__(
        self,
        num_preds: int,  # P
        num_conjuncts1: int,  # Q
        num_disjuncts1: int,
        num_conjuncts2: int,
        num_classes: int,  # R
        delta: float = 1.0,
        weight_init_type: str = "normal",
    ) -> None:
        super(DoubleDNFClassifier, self).__init__()

        self.dnf1 = DNF(
            num_preds, num_conjuncts1, num_disjuncts1, delta, weight_init_type
        )
        self.dnf2 = DNF(
            num_disjuncts1, num_conjuncts2, num_classes, delta, weight_init_type
        )

        self.conj_weight_mask1 = torch.ones(
            self.dnf1.conjunctions.weights.data.shape
        )
        self.disj_weight_mask1 = torch.ones(
            self.dnf1.disjunctions.weights.data.shape
        )
        self.conj_weight_mask2 = torch.ones(
            self.dnf2.conjunctions.weights.data.shape
        )
        self.disj_weight_mask2 = torch.ones(
            self.dnf2.disjunctions.weights.data.shape
        )

    def set_delta_val(self, delta_val: float) -> None:
        self.dnf1.conjunctions.delta = delta_val
        self.dnf1.disjunctions.delta = delta_val
        self.dnf2.conjunctions.delta = delta_val
        self.dnf2.disjunctions.delta = delta_val

    def update_weight_wrt_mask(self) -> None:
        self.dnf1.conjunctions.weights.data *= self.conj_weight_mask1
        self.dnf1.disjunctions.weights.data *= self.disj_weight_mask1
        self.dnf2.conjunctions.weights.data *= self.conj_weight_mask2
        self.dnf2.disjunctions.weights.data *= self.disj_weight_mask2

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        out = self.dnf1(input)
        out = nn.Tanh()(out)
        out = self.dnf2(out)
        # out: N x R
        return out

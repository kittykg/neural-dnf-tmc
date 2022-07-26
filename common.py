from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

TMC_NUM_ATTRIBUTES = 500
TMC_NUM_CLASSES = 22


@dataclass
class TmcDatasetSample:
    sample_id: int
    label_encoding: Tensor
    attribute_encoding: Tensor


@dataclass
class TmcRawSample:
    sample_id: int
    labels: List[int]
    present_attributes: List[int]

    def to_tmc_dataset_sample(self) -> TmcDatasetSample:
        def to_encoding(indices: List[int], dim: int) -> Tensor:
            encoding = torch.zeros(dim)
            encoding[indices] = 1
            return encoding

        return TmcDatasetSample(
            sample_id=self.sample_id,
            label_encoding=to_encoding(self.labels, TMC_NUM_CLASSES),
            attribute_encoding=to_encoding(
                self.present_attributes, TMC_NUM_ATTRIBUTES
            ),
        )

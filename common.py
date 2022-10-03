from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor


@dataclass
class MultiLabelDatasetSample:
    sample_id: int
    label_encoding: Tensor
    attribute_encoding: Tensor

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, MultiLabelDatasetSample):
            return False
        else:
            return (
                self.sample_id == __o.sample_id
                and bool(torch.all(self.label_encoding == __o.label_encoding))
                and bool(
                    torch.all(self.attribute_encoding == __o.attribute_encoding)
                )
            )


@dataclass
class MultiLabelRawSample:
    sample_id: int
    labels: List[int]
    present_attributes: List[int]

    def to_dataset_sample(
        self, num_attributes: int, num_labels: int
    ) -> MultiLabelDatasetSample:
        def to_encoding(indices: List[int], dim: int) -> Tensor:
            encoding = torch.zeros(dim)
            encoding[indices] = 1
            return encoding

        return MultiLabelDatasetSample(
            sample_id=self.sample_id,
            label_encoding=to_encoding(self.labels, num_labels),
            attribute_encoding=to_encoding(
                self.present_attributes, num_attributes
            ),
        )

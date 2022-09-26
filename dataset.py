from dataclasses import asdict
from typing import Dict, List

from torch.utils.data import Dataset

from common import MultiLabelDatasetSample


class TmcDataset(Dataset):
    dataset: List[MultiLabelDatasetSample]

    def __init__(self, dataset: List[MultiLabelDatasetSample]) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        data = self.dataset[idx]
        return asdict(data)

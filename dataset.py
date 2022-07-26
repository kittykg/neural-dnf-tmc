from dataclasses import asdict
from typing import Dict, List

from torch.utils.data import Dataset

from common import TmcRawSample


class TmcDataset(Dataset):
    dataset: List[TmcRawSample]

    def __init__(self, dataset: List[TmcRawSample]) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        data = self.dataset[idx].to_tmc_dataset_sample()
        return asdict(data)

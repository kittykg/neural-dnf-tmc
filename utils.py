from enum import Enum
from typing import Dict, Tuple
import pickle

from torch import Tensor
from torch.utils.data import DataLoader

from dataset import TmcDataset
from rule_learner import DNFClassifier


class DataloaderMode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


DATA_PATH_DICT_KEY = ["train", "val", "test"]


def load_multi_label_data(
    is_training: bool, batch_size: int, data_path_dict: Dict[str, str]
) -> Tuple[DataLoader, DataLoader] | DataLoader:
    def _get_dataloader(
        dataset_path: str,
        dataloader_mode: DataloaderMode,
    ) -> DataLoader:
        with open(dataset_path, "rb") as f:
            dataset = TmcDataset(pickle.load(f))

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=dataloader_mode == DataloaderMode.TRAIN,
        )

    if is_training:
        train_loader = _get_dataloader(
            dataset_path=data_path_dict[DATA_PATH_DICT_KEY[0]],
            dataloader_mode=DataloaderMode.TRAIN,
        )
        val_loader = _get_dataloader(
            dataset_path=data_path_dict[DATA_PATH_DICT_KEY[1]],
            dataloader_mode=DataloaderMode.VAL,
        )
        return train_loader, val_loader
    else:
        test_loader = _get_dataloader(
            dataset_path=data_path_dict[DATA_PATH_DICT_KEY[2]],
            dataloader_mode=DataloaderMode.TEST,
        )
        return test_loader


def get_dnf_classifier_x_and_y(
    data: dict, use_cuda: bool
) -> Tuple[Tensor, Tensor]:
    """
    Get ground truth x and y for DNF Classifier
    x: attribute encoding
    y: label encoding
    """
    x = 2 * data["attribute_encoding"] - 1
    y = data["label_encoding"]

    if use_cuda:
        x = x.to("cuda")
        y = y.to("cuda")

    return x, y


class DeltaDelayedDecayScheduler:
    initial_delta: float
    delta_decay_delay: int
    delta_decay_steps: int
    delta_decay_rate: float

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
    ):
        self.initial_delta = initial_delta
        self.delta_decay_delay = delta_decay_delay
        self.delta_decay_steps = delta_decay_steps
        self.delta_decay_rate = delta_decay_rate

    def step(self, model: DNFClassifier, step: int) -> float:
        if step < self.delta_decay_delay:
            new_delta_val = self.initial_delta
        else:
            new_delta_val = self.calculate_new_delta(step)
        new_delta_val = 1 if new_delta_val > 1 else new_delta_val
        model.set_delta_val(new_delta_val)
        return new_delta_val

    def calculate_new_delta(self, step: int) -> float:
        raise NotImplementedError


class DeltaDelayedExponentialDecayScheduler(DeltaDelayedDecayScheduler):
    def calculate_new_delta(self, step: int) -> float:
        delta_step = step - self.delta_decay_delay
        new_delta_val = self.initial_delta * (
            self.delta_decay_rate ** (delta_step // self.delta_decay_steps)
        )
        return new_delta_val


class DeltaDelayedArithmeticSequenceScheduler(DeltaDelayedDecayScheduler):
    # Delta decay rate is used as the difference as in an arithmetic sequence

    def calculate_new_delta(self, step: int) -> float:
        delta_step = step - self.delta_decay_delay
        new_delta_val = self.initial_delta + self.delta_decay_rate * (
            delta_step // self.delta_decay_steps
        )
        return new_delta_val

from enum import Enum
from typing import List

from sklearn.metrics import classification_report, jaccard_score
import torch
from torch import Tensor


class ClassificationMetric(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1-score"


class Meter:
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_average(self) -> float:
        raise NotImplementedError


class MacroMetricMeter(Meter):
    metric_choice: ClassificationMetric

    # Accumulate output and target until get_average is called, such that
    # precision and f1 would be defined (avoid division by 0)
    acc_output: List[int] = []
    acc_target: List[int] = []

    def __init__(
        self,
        metric_choice: ClassificationMetric = ClassificationMetric.PRECISION,
    ):
        super(MacroMetricMeter, self).__init__()
        self.metric_choice = metric_choice

    def update(self, output: Tensor, target: Tensor) -> None:
        # Output should be in range [0, 1]
        output = output.cpu().detach().numpy()
        output = (output > 0.5).astype(int).tolist()
        target = target.cpu().detach().numpy().astype(int).tolist()

        self.acc_output += output
        self.acc_target += target

    def get_average(self) -> float:
        cr = classification_report(
            self.acc_target, self.acc_output, output_dict=True, zero_division=0
        )
        macro_avg = cr["macro avg"][self.metric_choice.value]
        return macro_avg


class MetricValueMeter(Meter):
    metric_name: str
    vals: List[float]

    def __init__(self, metric_name: str):
        super(MetricValueMeter, self).__init__()
        self.metric_name = metric_name
        self.vals = []

    def update(self, val: float) -> None:
        self.vals.append(val)

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.vals)).item()

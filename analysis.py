from enum import Enum
from typing import Callable, Dict, List

from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import Tensor


class ClassificationMetric(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1-score"

    def get_metric_function(self) -> Callable:
        metric_func_dict: Dict[str, Callable] = {
            "precision": precision_score,
            "recall": recall_score,
            "f1-score": f1_score,
        }

        return metric_func_dict[self.value]


class Meter:
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_average(self) -> float:
        raise NotImplementedError


class MacroMetricMeter(Meter):
    metric_choice: ClassificationMetric

    # Accumulate output and target until get_average is called, such that
    # precision and f1 would be defined (avoid division by 0)
    acc_output: List[List[int]]
    acc_target: List[List[int]]

    def __init__(
        self,
        metric_choice: ClassificationMetric = ClassificationMetric.F1_SCORE,
    ) -> None:
        super(MacroMetricMeter, self).__init__()
        self.metric_choice = metric_choice
        self.acc_output = []
        self.acc_target = []

    def update(self, output: Tensor, target: Tensor) -> None:
        # Output should be in range [0, 1]
        output_list = (output.cpu().detach() > 0.5).int().tolist()
        target_list = target.cpu().detach().int().tolist()

        self.acc_output += output_list
        self.acc_target += target_list

    def get_average(self) -> float:
        return self.metric_choice.get_metric_function()(
            self.acc_target, self.acc_output, average="macro", zero_division=0
        )


class MetricValueMeter(Meter):
    metric_name: str
    vals: List[float]

    def __init__(self, metric_name: str) -> None:
        super(MetricValueMeter, self).__init__()
        self.metric_name = metric_name
        self.vals = []

    def update(self, val: float) -> None:
        self.vals.append(val)

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.vals)).item()

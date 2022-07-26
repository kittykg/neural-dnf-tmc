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
    macro_metric_vals: List[float]

    def __init__(
        self,
        metric_choice: ClassificationMetric = ClassificationMetric.PRECISION,
    ):
        super(MacroMetricMeter, self).__init__()
        self.metric_choice = metric_choice
        self.macro_metric_vals = []

    def update(self, output: Tensor, target: Tensor) -> None:
        # Output should be in range [0, 1]
        output = output.cpu().detach().numpy()
        output = (output > 0.5).astype(int)
        target = target.cpu().detach().numpy().astype(int)

        cr = classification_report(
            target, output, output_dict=True, zero_division=0
        )

        macro_avg = cr["macro avg"][self.metric_choice.value]
        self.macro_metric_vals.append(macro_avg)

    def get_average(self) -> float:
        return torch.mean(torch.Tensor(self.macro_metric_vals)).item()


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


class JaccardScoreMeter(Meter):
    jacc_scores: List[float]

    def __init__(self) -> None:
        super(JaccardScoreMeter, self).__init__()
        self.jacc_scores = []

    def update(self, output: Tensor, target: Tensor) -> None:
        y = torch.zeros(output.shape)
        y[range(output.shape[0]), target.long()] = 1
        y_np = y.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()
        avg = jaccard_score(y_np, output_np, average="samples")
        self.jacc_scores.append(avg)

    def get_average(self) -> float:
        return sum(self.jacc_scores) / len(self.jacc_scores)

import logging
import re
from typing import Any, Dict, List, Optional

import clingo
from clingo import SolveHandle, Model
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader

from analysis import MacroMetricMeter, ClassificationMetric
from common import MultiLabelDatasetSample
from rule_learner import DNFClassifier
from utils import get_dnf_classifier_x_and_y


log = logging.getLogger()


def dnf_eval(
    model: DNFClassifier,
    use_cuda: bool,
    data_loader: DataLoader,
    metric_choice: ClassificationMetric,
    do_logging: bool = False,
):
    model.eval()
    performance_meter = MacroMetricMeter(metric_choice)
    for i, data in enumerate(data_loader):
        iter_perf_meter = MacroMetricMeter(metric_choice)
        with torch.no_grad():
            x, y = get_dnf_classifier_x_and_y(data, use_cuda)
            y_hat = (torch.tanh(model(x)) + 1) / 2

            performance_meter.update(y_hat, y)

        if do_logging:
            log.info(
                "[%3d] Test     avg perf: %.3f"
                % (i + 1, iter_perf_meter.get_average())
            )

    if do_logging:
        log.info(
            "Overall Test   avg perf: %.3f" % performance_meter.get_average()
        )

    return performance_meter.get_average()


def asp_eval(
    test_data: List[MultiLabelDatasetSample], rules: List[str]
) -> Dict[str, Any]:
    all_prediction_one_hot: List[List[int]] = []
    all_target: List[List[int]] = []
    fully_correct_count = 0

    for d in test_data:
        all_target.append(d.label_encoding.int().tolist())

        # Get rule prediction
        asp_base = []
        for i, a in enumerate(d.attribute_encoding):
            if a == 1:
                asp_base.append(f"a{i}.")

        asp_base += rules
        asp_base.append("#show label/1.")

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], " ".join(asp_base))
        ctl.ground([("base", [])])

        sh = ctl.solve(yield_=True)
        assert isinstance(sh, SolveHandle)
        asp_model: Optional[Model] = sh.model()

        if not asp_model or str(asp_model) == "":
            prediction_one_hot = torch.zeros(d.label_encoding.shape)
        else:
            # Find predicted all label
            p = re.compile(r"\d+")
            predict_labels = [int(l) for l in p.findall(str(asp_model))]
            prediction_one_hot = torch.zeros(d.label_encoding.shape)
            prediction_one_hot[predict_labels] = 1

        all_prediction_one_hot.append(prediction_one_hot.int().tolist())

        if torch.all(d.label_encoding == prediction_one_hot):
            fully_correct_count += 1

    total_sample_count = len(test_data)
    out_metrics = precision_recall_fscore_support(
        all_target, all_prediction_one_hot, average="macro"
    )

    return {
        "rule_precision": out_metrics[0],
        "rule_recall": out_metrics[1],
        "rule_f1": out_metrics[2],
        "total_fully_correct_count": fully_correct_count,
        "total_count": total_sample_count,
    }

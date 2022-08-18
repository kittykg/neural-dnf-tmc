import logging

import torch
from torch.utils.data import DataLoader

from analysis import MacroMetricMeter, ClassificationMetric
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

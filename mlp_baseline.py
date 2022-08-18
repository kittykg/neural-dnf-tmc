import random
import numpy as np
import torch
from torch import nn, Tensor

from analysis import ClassificationMetric, MetricValueMeter, MacroMetricMeter
from utils import load_tmc_data, get_dnf_classifier_x_and_y

TMC_IN = 500
TMC_OUT = 22

USE_CUDA: bool = True
BATCH_SIZE: int = 256
DATA_PATH_DICT = {
    "train": "data/train.pkl",
    "val": "data/val.pkl",
    "test": "data/test.pkl",
}
LR: float = 0.001
WEIGHT_DECAY: float = 0.00004
EPOCHS: int = 100
MACRO_METRIC: ClassificationMetric = ClassificationMetric.PRECISION
RANDOM_SEED: int = 73


class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.l1 = nn.Linear(TMC_IN, 1000,  bias=False)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(1000, TMC_OUT, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        y = self.l1(input)
        y = self.a1(y)
        return self.l2(y)


def train(model: MLP) -> None:
    if USE_CUDA:
        model.to("cuda")

    train_loader, val_loader = load_tmc_data(
        is_training=True,
        batch_size=BATCH_SIZE,
        data_path_dict=DATA_PATH_DICT,
    )
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), LR, weight_decay=WEIGHT_DECAY
    )

    for e in range(EPOCHS):
        # TRAIN
        epoch_loss_meter = MetricValueMeter("epoch_loss_meter")
        epoch_perf_score_meter = MacroMetricMeter(MACRO_METRIC)
        model.train()
        for data in train_loader:
            optimiser.zero_grad()
            x, y = get_dnf_classifier_x_and_y(data, USE_CUDA)
            # y_hat = model(x)
            y_hat = (torch.tanh(model(x)) + 1) / 2
            loss = criterion(y_hat, y)
            loss.backward()
            optimiser.step()
            epoch_loss_meter.update(loss.item())
            # epoch_perf_score_meter.update(torch.sigmoid(y_hat), y)
            epoch_perf_score_meter.update(y_hat, y)
        avg_loss = epoch_loss_meter.get_average()
        avg_perf = epoch_perf_score_meter.get_average()
        print(
            "[%3d] Train   avg loss: %.3f  avg perf: %.3f"
            % (e + 1, avg_loss, avg_perf)
        )

        # VAL
        epoch_val_loss_meter = MetricValueMeter("epoch_val_loss_meter")
        epoch_val_perf_score_meter = MacroMetricMeter(MACRO_METRIC)
        model.eval()
        for data in val_loader:
            with torch.no_grad():
                # Get model output and compute loss
                x, y = get_dnf_classifier_x_and_y(data, USE_CUDA)
                # y_hat = model(x)
                y_hat = (torch.tanh(model(x)) + 1) / 2
                loss = criterion(y_hat, y)
                epoch_val_loss_meter.update(loss.item())
                # epoch_val_perf_score_meter.update(torch.sigmoid(y_hat), y)
                epoch_val_perf_score_meter.update(y_hat, y)
        avg_loss = epoch_val_loss_meter.get_average()
        avg_perf = epoch_val_perf_score_meter.get_average()
        print(
            "[%3d] Val     avg loss: %.3f  avg perf: %.3f"
            % (e + 1, avg_loss, avg_perf)
        )


def eval(model: MLP):
    model.eval()
    test_loader = load_tmc_data(
        is_training=False,
        batch_size=BATCH_SIZE,
        data_path_dict=DATA_PATH_DICT,
    )
    performance_meter = MacroMetricMeter(MACRO_METRIC)
    for data in test_loader:
        with torch.no_grad():
            x, y = get_dnf_classifier_x_and_y(data, USE_CUDA)
            # y_hat = model(x)
            y_hat = (torch.tanh(model(x)) + 1) / 2
            # performance_meter.update(torch.sigmoid(y_hat), y)
            performance_meter.update(y_hat, y)

    print(
        "Overall Test   avg perf: %.3f" % performance_meter.get_average()
    )


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model = MLP()
    train(model)
    torch.save(model.state_dict(), 'mlp_baseline.pth')
    eval(model)

import logging
from typing import Callable, Dict, Iterable, OrderedDict

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb

from analysis import ClassificationMetric, MacroMetricMeter, MetricValueMeter
from rule_learner import DNFClassifier
from utils import (
    DeltaDelayedArithmeticSequenceScheduler,
    DeltaDelayedDecayScheduler,
    DeltaDelayedExponentialDecayScheduler,
    get_dnf_classifier_x_and_y,
    load_multi_label_data,
    DATA_PATH_DICT_KEY,
)


log = logging.getLogger()


class DnfClassifierTrainer:
    # Data loaders
    train_loader: DataLoader
    val_loader: DataLoader

    # Training parameters
    use_cuda: bool
    experiment_name: str
    optimiser_key: str
    optimiser_fn: Callable[[Iterable], Optimizer]
    scheduler_fn: Callable
    loss_func_key: str
    criterion: Callable[[Tensor, Tensor], Tensor]
    epochs: int
    reg_fn: str
    reg_lambda: float
    macro_metric: ClassificationMetric = ClassificationMetric.F1_SCORE

    # Delta decay scheduler
    delta_decay_scheduler: DeltaDelayedDecayScheduler
    delta_one_counter: int = -1

    # Configs
    cfg: DictConfig
    model_train_cfg: DictConfig

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        # Configs
        self.cfg = cfg
        self.model_train_cfg = cfg["training"][model_name]

        # Training parameters
        self.use_cuda = (
            cfg["training"]["use_cuda"] and torch.cuda.is_available()
        )
        self.experiment_name = cfg["training"]["experiment_name"]

        # Data loaders
        env_cfg = cfg["environment"]
        batch_size = self.model_train_cfg["batch_size"]

        for k in DATA_PATH_DICT_KEY:
            assert k + "_pkl" in env_cfg
        data_path_dict = {}
        for k in DATA_PATH_DICT_KEY:
            data_path_dict[k] = env_cfg[k + "_pkl"]

        self.train_loader, self.val_loader = load_multi_label_data(
            is_training=True,
            batch_size=batch_size,
            data_path_dict=data_path_dict,
        )

        # Optimiser
        lr = self.model_train_cfg["optimiser_lr"]
        weight_decay = self.model_train_cfg["optimiser_weight_decay"]
        self.optimiser_key = self.model_train_cfg["optimiser"]
        if self.optimiser_key == "sgd":
            self.optimiser_fn = lambda params: torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            self.optimiser_fn = lambda params: torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay
            )

        # Scheduler
        scheduler_step = self.model_train_cfg["scheduler_step"]
        self.scheduler_fn = lambda optimiser: torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=scheduler_step, gamma=0.1
        )

        # Loss function
        self.criterion = torch.nn.BCELoss()

        # Other training parameters
        self.epochs = self.model_train_cfg["epochs"]
        self.reg_fn = self.model_train_cfg["reg_fn"]
        self.reg_lambda = self.model_train_cfg["reg_lambda"]

        self.delta_decay_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=self.model_train_cfg["initial_delta"],
            delta_decay_delay=self.model_train_cfg["delta_decay_delay"],
            delta_decay_steps=self.model_train_cfg["delta_decay_steps"],
            delta_decay_rate=self.model_train_cfg["delta_decay_rate"],
        )

        if "macro_metric" in self.model_train_cfg:
            macro_metric_str_val = self.model_train_cfg["macro_metric"]
            assert macro_metric_str_val in [
                e.value for e in ClassificationMetric
            ]
            self.macro_metric = ClassificationMetric(macro_metric_str_val)

    def train(self, model: DNFClassifier) -> dict:
        seed = torch.get_rng_state()[0].item()
        log.info(f"{self.experiment_name} starts, seed: {seed}")

        if self.use_cuda:
            model.to("cuda")

        optimiser = self.optimiser_fn(model.parameters())
        scheduler = self.scheduler_fn(optimiser)

        for epoch in range(self.epochs):
            # 1. Training
            self._epoch_train(epoch, model, optimiser)

            # 2. Evaluate performance on val
            self._epoch_val(epoch, model)

            # 3. Let scheduler update optimiser at end of epoch
            scheduler.step()

        return model.state_dict()

    def _epoch_train(
        self, epoch: int, model: DNFClassifier, optimiser: Optimizer
    ) -> None:
        epoch_loss_meter = MetricValueMeter("epoch_loss_meter")
        epoch_perf_score_meter = MacroMetricMeter(self.macro_metric)

        model.train()

        for i, data in enumerate(self.train_loader):
            optimiser.zero_grad()

            x, y = get_dnf_classifier_x_and_y(data, self.use_cuda)
            y_hat = (torch.tanh(model(x)) + 1) / 2

            loss = self._loss_calculation(y_hat, y, model.parameters())

            loss.backward()
            optimiser.step()

            # Update meters
            epoch_loss_meter.update(loss.item())
            epoch_perf_score_meter.update(y_hat, y)

        # Update delta value
        new_delta_val = self.delta_decay_scheduler.step(model, epoch)

        if new_delta_val == 1.0:
            # The first time where new_delta_val becomes 1, the network isn't
            # train with delta being 1 for that epoch. So delta_one_counter
            # starts with -1, and when new_delta_val is first time being 1,
            # the delta_one_counter becomes 0.
            self.delta_one_counter += 1

        # Log average performance for train
        avg_loss = epoch_loss_meter.get_average()
        avg_perf = epoch_perf_score_meter.get_average()
        log.info(
            "[%3d] Train  Delta: %.3f  avg loss: %.3f  avg perf: %.3f"
            % (epoch + 1, new_delta_val, avg_loss, avg_perf)
        )

        # Generate weight histogram
        # sd = model.state_dict()
        # conj_w = sd["dnf.conjunctions.weights"].flatten().detach().cpu().numpy()
        # disj_w = sd["dnf.disjunctions.weights"].flatten().detach().cpu().numpy()

        # f1 = plt.figure(figsize=(20, 15))
        # plt.title("Conjunction weight distribution")
        # arr = plt.hist(conj_w, bins=20)
        # for i in range(20):
        #     plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))

        # f2 = plt.figure(figsize=(20, 15))
        # plt.title("Disjunction weight distribution")
        # arr = plt.hist(disj_w, bins=20)
        # for i in range(20):
        #     plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))

        # WandB logging
        wandb.log(
            {
                "train/epoch": epoch + 1,
                "delta": new_delta_val,
                "train/loss": avg_loss,
                "train/macro_avg": avg_perf,
                # "conj_w_hist": f1,
                # "disj_w_hist": f2,
            }
        )

        # with open("train_macro", "a") as f:
        #     f.write(f"{avg_perf:.3f}, {new_delta_val:.3f}\n")

        # plt.close(f1)
        # plt.close(f2)

    def _epoch_val(self, epoch: int, model: DNFClassifier) -> float:
        epoch_val_loss_meter = MetricValueMeter("epoch_val_loss_meter")
        epoch_val_perf_score_meter = MacroMetricMeter(self.macro_metric)

        model.eval()

        for data in self.val_loader:
            with torch.no_grad():
                # Get model output and compute loss
                x, y = get_dnf_classifier_x_and_y(data, self.use_cuda)
                y_hat = (torch.tanh(model(x)) + 1) / 2
                loss = self._loss_calculation(y_hat, y, model.parameters())

                # Update meters
                epoch_val_loss_meter.update(loss.item())
                epoch_val_perf_score_meter.update(y_hat, y)

        avg_loss = epoch_val_loss_meter.get_average()
        avg_perf = epoch_val_perf_score_meter.get_average()
        log.info(
            "[%3d] Val                  avg loss: %.3f  avg perf: %.3f"
            % (epoch + 1, avg_loss, avg_perf)
        )

        wandb.log(
            {
                "val/epoch": epoch + 1,
                "val/loss": avg_loss,
                "val/macro_avg": avg_perf,
            }
        )

        return avg_perf

    def _loss_calculation(
        self,
        y_hat: Tensor,
        y: Tensor,
        parameters: Iterable[nn.parameter.Parameter],
    ) -> Tensor:
        loss = self.criterion(y_hat, y)

        if self.delta_one_counter >= 10:
            # Extra regularisation when delta has been 1 more than for 10.
            # Pushes weights towards 0, -6 or 6.
            def modified_l1_regulariser(w: Tensor):
                return torch.abs(w * (6 - torch.abs(w))).sum()

            def l1_regulariser(w: Tensor):
                return torch.abs(w).sum()

            weight_regulariser = (
                modified_l1_regulariser
                if self.reg_fn == "l1_mod"
                else l1_regulariser
            )
            reg = self.reg_lambda * sum(
                [weight_regulariser(p.data) for p in parameters]
            )
            loss += reg

        return loss

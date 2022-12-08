from datetime import datetime
import os
import random
import requests
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from rule_learner import DNFClassifier, DoubleDNFClassifier
from train import DnfClassifierTrainer


WANDB_DISABLED = True


def _run_experiment_helper(cfg: DictConfig):
    experiment_name = cfg["training"]["experiment_name"]
    model_name = cfg["training"]["model_name"]

    # Set up wandb
    run = wandb.init(
        project="tmc-dnf",
        entity="kittykg",
        config=OmegaConf.to_container(cfg["training"][model_name]),  # type: ignore
        dir=HydraConfig.get().run.dir,
    )

    # Set random seed
    random_seed = cfg["training"]["random_seed"]
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Set up model
    base_cfg = OmegaConf.to_container(cfg["model"]["base_dnf"])
    model = DNFClassifier(**base_cfg)  # type: ignore
    model.set_delta_val(cfg["training"][model_name]["initial_delta"])

    torch.autograd.set_detect_anomaly(True)  # type: ignore

    trainer = DnfClassifierTrainer(model_name, cfg)
    state_dict = trainer.train(model)

    # Save model
    torch.save(state_dict, f"{experiment_name}_{random_seed}.pth")
    model_artifact = wandb.Artifact(
        f"{experiment_name}",
        type="model",
        description=f"{experiment_name} model",
        metadata=dict(wandb.config),
    )
    model_artifact.add_file(f"{experiment_name}_{random_seed}.pth")
    wandb.save(f"{experiment_name}_{random_seed}.pth")
    run.log_artifact(model_artifact)  # type: ignore


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    experiment_name = cfg["training"]["experiment_name"]
    random_seed = cfg["training"]["random_seed"]
    webhook_url = cfg["webhook"]["discord_webhook_url"]
    nodename = os.uname().nodename

    def post_to_discord_webhook(webhook_url: str, message: str) -> None:
        requests.post(webhook_url, json={"content": message})

    try:
        _run_experiment_helper(cfg)
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        webhook_msg = (
            f"[{dt}]\nExperiment {experiment_name} (seed {random_seed}) "
            f"on Machine {nodename} FINISHED!! Check out the log for result :P"
        )
    except BaseException as e:
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        webhook_msg = (
            f"[{dt}]\nExperiment {experiment_name} (seed {random_seed}) "
            f"on Machine {nodename} got an error!! Check that out!!"
        )
        print(traceback.format_exc())
    finally:
        post_to_discord_webhook(webhook_url, webhook_msg)  # type: ignore


if __name__ == "__main__":
    if WANDB_DISABLED:
        os.environ["WANDB_MODE"] = "disabled"
    run_experiment()

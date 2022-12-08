from datetime import datetime
import os
import pickle
import random
import requests
import traceback
from typing import Dict

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from analysis import ClassificationMetric
from rule_learner import DNFClassifier
from train import DnfClassifierTrainer
from dnf_post_train import DNFPostTrainingProcessor


def post_to_discord_webhook(webhook_url: str, message: str) -> None:
    requests.post(webhook_url, json={"content": message})


def convert_result_dict_to_discord_message(
    experiment_name: str,
    random_seed: int,
    metric_choice: ClassificationMetric,
    rd: Dict[str, float],
) -> str:
    nodename = os.uname().nodename
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    s = (
        f"[{dt}]\n"
        f"Post training process {experiment_name} (seed {random_seed}) "
        f"on Machine {nodename} has finished!\n"
    )
    s += f"Results (on test set):"
    s += f"""```
                        Macro {metric_choice.value}
        After train     {rd['after_train_perf']}
        After prune     {rd['after_prune_test']}
        After tune      {rd['after_tune']}
        After thresh    {rd['after_threshold_test']}

        Rule extract
           Macro precision  {rd['rule_precision']}
           Macro recall     {rd['rule_recall']}
           Macro f1         {rd['rule_f1']}
           Fully correct %  {rd['rule_fc_percentage']}```
    """
    return s


def run_train(
    cfg: DictConfig, experiment_name: str, random_seed: int
) -> DNFClassifier:
    experiment_name = cfg["training"]["experiment_name"]
    model_name = cfg["training"]["model_name"]

    # Set up wandb
    run = wandb.init(
        project="tmc-dnf",
        entity="kittykg",
        config=OmegaConf.to_container(cfg["training"][model_name]),  # type: ignore
        dir=HydraConfig.get().run.dir,
    )

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

    return model


def run_post_training_processing(
    model_name: str, cfg: DictConfig, model: DNFClassifier
) -> Dict[str, float]:
    pt_processor = DNFPostTrainingProcessor(model_name, cfg)
    result_dict = pt_processor.post_processing(model)
    return result_dict


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_pipeline(cfg: DictConfig) -> None:
    # Config parameters
    random_seed = cfg["training"]["random_seed"]
    experiment_name = cfg["training"]["experiment_name"]
    model_name = cfg["training"]["model_name"]
    if "macro_metric" in cfg["training"]:
        macro_metric_str_val = cfg["training"]["macro_metric"]
        assert macro_metric_str_val in [e.value for e in ClassificationMetric]
        metric_choice = ClassificationMetric(macro_metric_str_val)
    else:
        metric_choice = ClassificationMetric.F1_SCORE
    webhook_url = cfg["webhook"]["discord_webhook_url"]

    # Set random seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Train and post-training process
    try:
        model = run_train(cfg, experiment_name, random_seed)
        result_dict = run_post_training_processing(model_name, cfg, model)
        with open(
            f"{experiment_name}_full_pipeline_result_dict.pkl", "wb"
        ) as f:
            pickle.dump(result_dict, f)
        webhook_msg = convert_result_dict_to_discord_message(
            experiment_name, random_seed, metric_choice, result_dict
        )
    except BaseException as e:
        nodename = os.uname().nodename
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        webhook_msg = (
            f"[{dt}]\nExperiment {experiment_name} (seed {random_seed}) "
            f"on Machine {nodename} got an error!! Check that out!!"
        )
        print(traceback.format_exc())
    finally:
        post_to_discord_webhook(webhook_url, webhook_msg)  # type: ignore


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled"
    run_pipeline()

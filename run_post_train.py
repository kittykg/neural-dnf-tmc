from datetime import datetime
import os
import pickle
import random
import requests
import traceback
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from analysis import ClassificationMetric

from rule_learner import DNFClassifier
from dnf_post_train import DNFPostTrainingProcessor


WANDB_DISABLED = True


def _run_pt_process_helper(cfg: DictConfig) -> Dict[str, float]:
    model_name = cfg["training"]["model_name"]
    use_cuda = cfg["training"]["use_cuda"]

    # Set random seed
    random_seed = cfg["training"]["random_seed"]
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Set up model
    base_cfg = OmegaConf.to_container(cfg["model"]["base_dnf"])
    model = DNFClassifier(**base_cfg)  # type: ignore
    if use_cuda:
        model.to("cuda")

    # The state dict pth path is not defined in any hydra conf dir, this sholud
    # be passed as an extra arg of '+pt.pth_path=...'
    sd_path = cfg["pt"]["pth_path"]
    model.load_state_dict(torch.load(sd_path))
    # Trained model should has delta being 1 at the end
    model.set_delta_val(1)

    pt_processor = DNFPostTrainingProcessor(model_name, cfg)
    result_dict = pt_processor.post_processing(model)
    return result_dict


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_post_training_process(cfg: DictConfig) -> None:
    experiment_name = cfg["training"]["experiment_name"]
    random_seed = cfg["training"]["random_seed"]
    webhook_url = cfg["webhook"]["discord_webhook_url"]
    if "macro_metric" in cfg["training"]:
        macro_metric_str_val = cfg["training"]["macro_metric"]
        assert macro_metric_str_val in [e.value for e in ClassificationMetric]
        metric_choice = ClassificationMetric(macro_metric_str_val)
    else:
        metric_choice = ClassificationMetric.PRECISION

    def post_to_discord_webhook(webhook_url: str, message: str) -> None:
        requests.post(webhook_url, json={"content": message})

    try:
        result_dict = _run_pt_process_helper(cfg)

        with open(f"{experiment_name}_pt_result_dict.pkl", "wb") as f:
            pickle.dump(result_dict, f)
        webhook_msg = convert_result_dict_to_discord_message(
            experiment_name, random_seed, metric_choice, result_dict
        )
    except BaseException as e:
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        nodename = os.uname().nodename
        webhook_msg = (
            f"[{dt}]\n"
            f"Post training process {experiment_name} (seed {random_seed}) "
            f"on Machine {nodename} got an error!! Check that out!!"
        )
        print(traceback.format_exc())
    finally:
        post_to_discord_webhook(
            webhook_url, webhook_msg
        )  # pyright: reportUnboundVariable=false


if __name__ == "__main__":
    if WANDB_DISABLED:
        os.environ["WANDB_MODE"] = "disabled"
    run_post_training_process()

from argparse import ArgumentParser

from shutil import rmtree, copytree
from math import ceil
import os
import json
import subprocess
from omegaconf import OmegaConf


def fine_tune_and_save_metrics(project_name: str):
    data_path = os.path.join("data", "stage2")
    rmtree(data_path)
    copytree(os.path.join("preprocessed", project_name, "stage2"), data_path)

    part_sizes = json.load(open(os.path.join(data_path, "stats.json"), "r"))

    config_path = os.path.join("fine-tuning-experiments", "from_scratch_config.yaml")
    from_scratch_config = OmegaConf.load(open(config_path, "r"))
    from_scratch_config["training"]["simulated_batch_size_valid"] = part_sizes["val"]
    from_scratch_config["training"]["persistent_snapshot_every"] = ceil(
        part_sizes["train"] / from_scratch_config["training"]["batch_size"])
    OmegaConf.save(from_scratch_config, config_path)

    cmd = f"python -m scripts.run-experiment {config_path}"
    subprocess.check_call(cmd, shell=True)

    # print(from_scratch_config)
    # print(part_sizes)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_names", type=str, help="Path to file with project names")
    args = arg_parser.parse_args()

    with open(args.project_names, "r") as f:
        for name in f:
            fine_tune_and_save_metrics(name.strip())

from argparse import ArgumentParser
from shutil import rmtree, copytree
import os
import json
from .test_model import calculate_metrics

PRETRAINED = "CT-20"


def snapshot_by_id(model_id: str) -> str:
    snapshot = sorted(os.listdir(os.path.join("models", "ct_code_summarization", model_id)))[-1]
    snapshot = "".join(x for x in snapshot if x in "0123456789")
    return snapshot


def save_metrics(project: str, from_scratch_id: str, fine_tuned_id: str) -> None:
    data_path = os.path.join("data", "stage2")
    rmtree(data_path)
    copytree(os.path.join("preprocessed", project, "stage2"), data_path)
    result_path = os.path.join("results", project)
    os.mkdir(result_path)

    metrics, names = calculate_metrics(from_scratch_id, snapshot_by_id(from_scratch_id), save_predictions=True)
    with open(os.path.join(result_path, "new_after.json"), "w") as file:
        json.dump(metrics, file)
    with open(os.path.join(result_path, "new_after_names.txt"), "w") as file:
        file.writelines(names)

    metrics, names = calculate_metrics(PRETRAINED, snapshot_by_id(PRETRAINED), save_predictions=True)
    with open(os.path.join(result_path, "trained_before.json"), "w") as file:
        json.dump(metrics, file)
    with open(os.path.join(result_path, "trained_before_names.txt"), "w") as file:
        file.writelines(names)

    metrics, names = calculate_metrics(fine_tuned_id, snapshot_by_id(fine_tuned_id), save_predictions=True)
    with open(os.path.join(result_path, "trained_after.json"), "w") as file:
        json.dump(metrics, file)
    with open(os.path.join(result_path, "trained_after_names.txt"), "w") as file:
        file.writelines(names)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("projects", type=str, help="Path to file with project names, and models ids")
    args = arg_parser.parse_args()

    with open(args.projects, "r") as f:
        for name in f:
            save_metrics(*name.strip().split(","))

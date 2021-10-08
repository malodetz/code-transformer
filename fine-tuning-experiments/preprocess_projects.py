import os
from shutil import rmtree, copytree
import subprocess
from argparse import ArgumentParser


def process_single(project_name: str):
    print(f"Processing {project_name}...")
    rmtree("data")
    raw_dataset = os.path.join("data", "raw", "code2seq", "java-small")

    os.makedirs(raw_dataset)
    os.mkdir(os.path.join("data", "stage1"))
    os.mkdir(os.path.join("data", "stage2"))

    project_path = os.path.join("raw_java", project_name)
    for part in zip(["train", "val", "test"], ["training", "validation", "test"]):
        copytree(os.path.join(project_path, part[0]), os.path.join(os.path.join(raw_dataset, part[1])))

    cmd = "python -m scripts.extract-java-methods java-small"
    subprocess.check_call(cmd, shell=True)

    for part in ["train", "valid", "test"]:
        cmd = (
            f"python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-code2seq"
            f".yaml java-small {part} "
        )
        subprocess.check_call(cmd, shell=True)

    for part in ["train", "valid", "test"]:
        cmd = (
            f"python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2.yaml "
            f"java-small {part} "
        )
        subprocess.check_call(cmd, shell=True)

    preprocessed_path = os.path.join("preprocessed", project_name)
    copytree("data", preprocessed_path)

    print("Finished!")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_names", type=str, help="Path to file with project names")
    args = arg_parser.parse_args()

    with open(args.project_names, "r") as f:
        for name in f:
            process_single(name.strip())

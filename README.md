# Code Transformer

Fine-tuning specified PyTorch implementation of the `CodeTransformer` model proposed in:
> D. Zügner, T. Kirschstein, M. Catasta, J. Leskovec, and S. Günnemann, *“Language-agnostic representation learning of source code from structure and context”*

## Before experiments

1. Create conda environment and activate it:

```shell
conda env create --name <your-env-name> -f ct-env.yml
conda activate <your-env-name>
```

2. Put `.env` file from root of this project to `${HOME}/.config/code_transformer/.env`

## Preprocessing

1. Place folders with Java projects split into train/val/test subdirectories into `raw_java` directory
2. Write names of this projects separated by `'\n'` into some text file
3. Pass name of this file as argument to `fine-tuning-experiments/preprocess_projects.py` script
4. Preprocessed projects can be found in `preprocessed directory`

Example run:

```shell
python -m fine-tuning-experiments.preprocess_projects names.txt
```

## Fine-tuning

1. After preprocessing, write names of projects from `preprocessed` folder to some text file
2. Pass name of this file as argument to `fine-tuning-experiments/fine_tune_on_projects.py` script
3. Results for each project can be found in separate folders in `results` folder. Metrics and method names (target and
   predicted) are saved for three models: trained from scratch, original and fine-tuned
4. Fine-tuned and trained from scratch models can be found at `models/ct-code-summarization`

Example run:

```shell
python -m fine-tuning-experiments.fine_tune_on_projects names.txt
```

## Training reduced model from scratch

1. Run `fine-tuning-experiments/train_reduced_from_scratch.py` script, passing names on preprocessed projects to in a
   text file

Example run:

```shell
python -m fine-tuning-experiments.fine_tune_on_projects names.txt
```
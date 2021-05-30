import os
import re
from dataclasses import dataclass, field
from os.path import join

from transformers import HfArgumentParser, is_wandb_available
from transformers.models.auto import AutoModelForSequenceClassification

import bias_probing.config as project_config

WANDB_PROJECT_NAME = 'retraining'


@dataclass
class Arguments:
    tag: str
    model_name_or_path: str
    seed: int = field(default=42)
    max_seq_length: int = field(default=128)
    batch_size: int = field(default=64)
    wandb_project_name: str = field(default=WANDB_PROJECT_NAME)
    config_dir: str = field(default='configs')
    logging_dir: str = field(default='outputs')
    overwrite_cache: bool = False


def real_model_path(model_name_or_path):
    if os.path.isdir(join(project_config.MODELS_DIR, model_name_or_path)):
        return join(project_config.MODELS_DIR, model_name_or_path)
    return model_name_or_path


def _prepare_model(args: Arguments):
    model_name_or_path = real_model_path(args.model_name_or_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return model


def _init_wandb(args: Arguments):
    tag = args.tag

    if is_wandb_available():
        import wandb
        wandb.init(project=args.wandb_project_name, name=re.sub('seed:[0-9]+/', '', tag))
        wandb.config.update({
            k: v for k, v in args.__dict__.items()
        })


def main(args: Arguments):
    pass


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    # noinspection PyTypeChecker
    _args: Arguments = parser.parse_args()
    main(_args)
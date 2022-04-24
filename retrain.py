import os
import re
from dataclasses import dataclass, field
from os.path import join, abspath
from torch import nn
from typing import Dict, List

from transformers import (
    HfArgumentParser,
    is_wandb_available,
    set_seed,
    TrainingArguments,
    IntervalStrategy
)
from transformers.models.auto import AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.bert import BertTokenizerFast, BertForSequenceClassification
from datasets import DatasetDict

import bias_probing.config as project_config
from bias_probing.data.datasets import load_dataset_aux
from bias_probing.training import MultiPredictionDatasetTrainer
from bias_probing.modelling import BertWithWeakLearnerLegacy


# Faster tokenizer for optimization
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


WANDB_PROJECT_NAME = 'retraining'


@dataclass
class Arguments:
    model_name_or_path: str
    seed: int = field(default=42)
    max_seq_length: int = field(default=128)
    num_labels: int = 3
    batch_size: int = field(default=32)
    num_train_epochs: int = field(default=1)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.0)
    learning_rate: float = field(default=5e-5)
    save_steps: int = field(default=5000)
    # TODO Check
    logging_steps: int = field(default=100)
    eval_steps: int = field(default=200)
    init_classifier: bool = False
    freeze_encoder: bool = False
    train_dataset: str = field(default='snli')
    dev_datasets: List[str] = field(default_factory=lambda: ['snli', 'hans'])
    test_datasets: List[str] = field(default_factory=lambda: ['snli', 'hans'])
    overwrite_cache: bool = False
    config_dir: str = field(default='configs')
    logging_dir: str = field(default='outputs')
    do_train: bool = True
    do_eval: bool = True
    wandb_project_name: str = field(default=WANDB_PROJECT_NAME)


def real_model_path(model_name_or_path):
    target = abspath(join(project_config.MODELS_DIR, model_name_or_path))
    if os.path.isdir(target):
        return target
    return model_name_or_path


def _prepare_model(args: Arguments):
    model_name_or_path = real_model_path(args.model_name_or_path)
    num_labels = args.num_labels

    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    if 'architectures' in config.__dict__ and 'BertWithWeakLearner' in config.architectures:
        base_model = BertWithWeakLearnerLegacy.from_pretrained(model_name_or_path, config=config)
        model = base_model.bert
        assert isinstance(model, BertForSequenceClassification)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    return model, config


def _preprocess_func(args: Arguments, info, **tokenizer_kwargs):
    max_seq_length = args.max_seq_length
    premise_key = info.sentence1_key
    hypothesis_key = info.sentence2_key
    
    def _preprocess(examples):
        # Tokenize the texts
        args = (examples[premise_key], examples[hypothesis_key])
        result = tokenizer(*args, max_length=max_seq_length, truncation=True, return_length=True, **tokenizer_kwargs)
        return result
    return _preprocess


def _prepare_datasets(args: Arguments):
    train_dataset_name = args.train_dataset
    overwrite_cache = args.overwrite_cache
    dev_dataset_names = args.dev_datasets
    test_dataset_names = args.test_datasets

    ds, info = load_dataset_aux(train_dataset_name)

    # Train
    ds: DatasetDict = ds.map(_preprocess_func(args, info, ), 
                        batched=True, load_from_cache_file=not overwrite_cache).filter(lambda x: x['label'] != -1)

    train_dataset = ds[info.train_dataset_name]

    # Validation
    dev_datasets = []
    for dev_dataset_name in dev_dataset_names:
        dev_ds, dev_info = load_dataset_aux(dev_dataset_name)
        dev_ds = dev_ds.map(_preprocess_func(args, dev_info), 
                        batched=True, load_from_cache_file=not overwrite_cache).filter(lambda x: x['label'] != -1)
        dev_datasets.append((dev_dataset_name, dev_ds[dev_info.dev_dataset_name], dev_info))

    # Test
    test_datasets = []
    for test_dataset_name in test_dataset_names:
        test_ds, test_info = load_dataset_aux(test_dataset_name)
        test_ds = test_ds.map(_preprocess_func(args, test_info), 
                        batched=True, load_from_cache_file=not overwrite_cache).filter(lambda x: x['label'] != -1)
        test_datasets.append((test_dataset_name, test_ds[test_info.test_dataset_name], test_info))

    return train_dataset, dev_datasets, test_datasets


def _training_args(tag, args: Arguments):
    report_to_wandb = (args.wandb_project_name != None)
    return TrainingArguments(
        output_dir=args.logging_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        evaluation_strategy=IntervalStrategy.STEPS,
        run_name=tag,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        skip_memory_metrics=True,
        remove_unused_columns=True,
        group_by_length=True,
        seed=args.seed,
        report_to=['wandb'] if report_to_wandb else []
    )


def _init_wandb(args: Arguments):
    tag = args.model_name_or_path
    tag = re.sub('seed:[0-9]+/', '', tag)
    if args.train_dataset != 'snli':
        tag += f'/{args.train_dataset}'
    tag += '/' + '_'.join(
        (['freeze'] if args.freeze_encoder else []) +
        (['init'] if args.init_classifier else [])
    )

    if is_wandb_available():
        import wandb
        wandb.init(project=args.wandb_project_name, name=tag)
        wandb.config.update({
            k: v for k, v in args.__dict__.items()
        })
    return tag


def main(args: Arguments):
    
    tag = _init_wandb(args)

    init_classifier = args.init_classifier
    freeze_encoder = args.freeze_encoder
    train_dataset_name = args.train_dataset
    seed = args.seed

    set_seed(seed)

    model, config = _prepare_model(args)
    print(f'Loaded model {model.__class__.__name__}')

    if init_classifier:
        print('INIT CLASSIFIER')
        assert isinstance(model.classifier, nn.Linear)
        model.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if model.classifier.bias is not None:
            model.classifier.bias.data.zero_()
        print('Initialized classifier weights')

    assert model.bert is not None
    if freeze_encoder:
        print('FREEZE ENCODER')
        for param in model.bert.parameters():
            param.requires_grad = False
    
    train_dataset, dev_datasets, test_datasets = _prepare_datasets(args)

    training_args = _training_args(tag, args)

    trainer = MultiPredictionDatasetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_datasets=dev_datasets,
        tokenizer=tokenizer
    )

    if args.do_train:
        model_dir = join(project_config.MODELS_DIR, 
                        f'seed:{seed}',
                        f'retrain_{train_dataset_name}',
                        tag
                        )
        print(f'Model will be saved at {os.path.abspath(model_dir)}')

        train_output = trainer.train()
        print(f'Train outputs: {train_output}')
        print(f'Saving model to {os.path.abspath(model_dir)}')
        trainer.save_model(model_dir)
    else:
        print('Skipping training')

    if args.do_eval:
        eval_output = trainer.evaluate(eval_datasets=test_datasets)
        print(f'Eval Outputs: {eval_output}')
    else:
        print('Skipping evaluation')
    

if __name__ == '__main__':
    # Disable gradient watch
    os.environ['WANDB_WATCH'] = 'false'
    
    parser = HfArgumentParser(Arguments)
    # noinspection PyTypeChecker
    (_args, ) = parser.parse_args_into_dataclasses()[:1]
    main(_args)
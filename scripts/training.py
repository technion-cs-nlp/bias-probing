import os
import re
from collections import defaultdict
from os.path import join
from pprint import pprint
from typing import Type, Dict, Tuple, List

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from transformers import (
    PreTrainedModel,
    BertConfig,
    AutoConfig,
    AutoModelForSequenceClassification,
    set_seed,
    TrainingArguments,
    EvalPrediction,
    IntervalStrategy
)
from transformers.integrations import is_wandb_available
from transformers.models.bert import BertTokenizerFast, BertForSequenceClassification
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput

import bias_probing.config as project_config
from bias_probing.data.datasets import load_dataset_aux, DatasetInfo
from bias_probing.heuristics import is_constituent, is_subsequence, have_lexical_overlap
from bias_probing.modelling import (
    BertWithWeakLearner,
    BertWithWeakLearnerConfig,
    BertDistillConfig,
    BertDistill,
    BertWithExplicitBiasConfig,
    BertWithExplicitBias,
    BertWithLexicalBiasConfig,
    BertWithLexicalBiasModel
)

# Faster tokenizer for optimization
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


MODEL_TRAINING_PROJECT_NAME = 'bias-probing'


def per_class_accuracy_with_names(id_to_label: Dict = None):
    def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        classes = np.unique(y_true)
        acc_dict = {}
        for c in classes:
            indices = (y_true == c)
            y_true_c = y_true[indices]
            y_pred_c = y_pred[indices]
            class_name = id_to_label[c] if id_to_label is not None else c
            acc_dict[f'accuracy_{class_name}'] = accuracy_score(y_true=y_true_c, y_pred=y_pred_c)
        return acc_dict

    return _per_class_accuracy


per_class_accuracy = per_class_accuracy_with_names()


def compute_metrics_default(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        **per_class_accuracy(labels, preds)
    }


def compute_metrics_wrap(compute_metrics_fn, preprocess_fn):
    def wrapper(pred):
        new_pred = preprocess_fn(pred)
        return compute_metrics_fn(new_pred)

    return wrapper


def real_model_path(model_name_or_path):
    if os.path.isdir(join(project_config.MODELS_DIR, model_name_or_path)):
        return join(project_config.MODELS_DIR, model_name_or_path)
    return model_name_or_path


def _select_model_configuration(cfg: DictConfig) -> Tuple[PreTrainedModel, BertConfig, Type[Trainer]]:
    # Parameters
    run_type = cfg.get('type')
    model_name_or_path = real_model_path(cfg.get('model_name_or_path'))
    do_train = cfg.get('do_train', True)
    if not do_train:
        # Replace model name with tag
        seed = cfg.get('seed', 42)
        tag = cfg.get('tag')
        model_name_or_path = real_model_path(f'seed:{seed}/{tag}')

    if run_type == 'simple' or run_type == 'hypo_only':
        # Fine-tune a standard BERT (standard CE, no teacher)
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=3)
        if config.architectures and 'BertWithWeakLearner' in config.architectures:
            # Quickfix to extract BertForSequenceClassification from the model
            model = BertWithWeakLearner.from_pretrained(model_name_or_path, config=config).bert
            assert isinstance(model, BertForSequenceClassification)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

        freeze_encoder = cfg.get('freeze_encoder', False)
        if freeze_encoder:
            for param in model.bert.parameters():
                param.requires_grad = False
    elif run_type == 'debias':
        # Parameters
        weak_model_name_or_path = real_model_path(cfg.get('weak_model_name_or_path'))
        expert_policy = cfg.get('expert_policy')
        poe_alpha = cfg.get('poe_alpha')
        dfl_gamma = cfg.get('dfl_gamma')
        lambda_bias = cfg.get('lambda_bias')
        loss_fn: str = cfg.get('loss_fn')
        load_predictions_mode = cfg.get('load_predictions_mode')

        if loss_fn.startswith('cr/'):
            loss_fn = loss_fn.split('/')[-1]
            weak_model_name_or_path = join(project_config.ROOT_DIR, '..', cfg.get('weak_model_name_or_path'))
            teacher_model_name_or_path = join(project_config.ROOT_DIR, '..', cfg.get('teacher_model_name_or_path'))
            config = BertDistillConfig.from_pretrained(model_name_or_path,
                                                       num_labels=3,
                                                       expert_policy=expert_policy,
                                                       loss_fn=loss_fn,
                                                       weak_model_name_or_path=weak_model_name_or_path,
                                                       teacher_model_name_or_path=teacher_model_name_or_path,
                                                       load_predictions_mode=load_predictions_mode
                                                       )
            model = BertDistill.from_pretrained(model_name_or_path, config=config)
        elif loss_fn.startswith('explicit/'):
            loss_fn = loss_fn.split('/')[-1]
            bias_type = cfg.get('bias_type')
            dataset_name = cfg.get('train_dataset')
            base_hans_features = ['subsequence', 'lexical_overlap', 'overlap_rate']
            if dataset_name == 'multi_nli':
                base_hans_features = ['constituent'] + base_hans_features
            config = BertWithExplicitBiasConfig.from_pretrained(model_name_or_path,
                                                                num_labels=3,
                                                                bias_type=bias_type,
                                                                # SNLI supplies no parse trees
                                                                hans_features=base_hans_features,
                                                                expert_policy=expert_policy,
                                                                poe_alpha=poe_alpha,
                                                                dfl_gamma=dfl_gamma,
                                                                lambda_bias=lambda_bias,
                                                                loss_fn=loss_fn)
            model = BertWithExplicitBias.from_pretrained(model_name_or_path, config=config)
            print(f'Objective: {model.loss_fn}')
        else:
            config = BertWithWeakLearnerConfig.from_pretrained(model_name_or_path,
                                                               num_labels=3,
                                                               expert_policy=expert_policy,
                                                               poe_alpha=poe_alpha,
                                                               dfl_gamma=dfl_gamma,
                                                               lambda_bias=lambda_bias,
                                                               loss_fn=loss_fn,
                                                               weak_model_name_or_path=weak_model_name_or_path)
            model = BertWithWeakLearner.from_pretrained(model_name_or_path, config=config)
            print(f'Objective: {model.loss_fn}')
    elif run_type == 'hans_only':
        dataset_name = cfg.get('train_dataset')
        base_hans_features = ['subsequence', 'lexical_overlap', 'overlap_rate']
        if dataset_name == 'multi_nli':
            base_hans_features = ['constituent'] + base_hans_features
        config = BertWithLexicalBiasConfig.from_pretrained(model_name_or_path,
                                                           hans_features=base_hans_features,
                                                           num_labels=3
                                                           )

        model = BertWithLexicalBiasModel.from_pretrained(model_name_or_path, config=config)
    else:
        raise ValueError(f'Bad run_type ({run_type})')

    trainer_cls = Trainer
    return model, config, trainer_cls


def _prepare_datasets(cfg: DictConfig, remove_original_sentences=True) -> \
        Tuple[Dataset, Dataset, Dataset, List[Tuple[str, Dataset, DatasetInfo]]]:
    # Parameters
    train_dataset_name = cfg.get('train_dataset')
    overwrite_cache = cfg.get('overwrite_cache', False)
    run_type = cfg.get('type')
    datasets, dataset_info = load_dataset_aux(train_dataset_name)
    test_datasets = []

    ignored_keys = dataset_info.ignored_keys
    remove_columns = ignored_keys
    if remove_original_sentences:
        remove_columns += [dataset_info.sentence1_key, dataset_info.sentence2_key]
    if 'id' in remove_columns:
        remove_columns.remove('id')  # Include id in training data
    datasets: DatasetDict = datasets.map(
        _select_preprocess_func(cfg, premise_key=dataset_info.sentence1_key, hypothesis_key=dataset_info.sentence2_key),
        batched=True, load_from_cache_file=not overwrite_cache,
        remove_columns=remove_columns).filter(lambda e: e['label'] != -1)

    if run_type != 'hans_only':
        # Can't evaluate hans features for o.o.d datasets
        for eval_dataset_name in dataset_info.eval_datasets:
            dataset_tag = None
            if '::' in eval_dataset_name:
                eval_dataset_name, dataset_tag = eval_dataset_name.split('::')[:2]

            eval_datasets, eval_dataset_info = load_dataset_aux(eval_dataset_name)
            if eval_dataset_name == 'hans':
                # Quickfix to add HANS subsets
                for key in ['lexical_overlap', 'subsequence', 'constituent']:
                    eval_datasets[key] = eval_datasets[eval_dataset_info.test_dataset_name] \
                        .filter(lambda x: x['heuristic'] == key)

            if dataset_tag is None:
                dataset_tag = eval_dataset_info.test_dataset_name
            else:
                eval_dataset_name = f'{eval_dataset_name}_{dataset_tag}'

            remove_columns = eval_dataset_info.ignored_keys
            column_names = eval_datasets.column_names[dataset_tag]
            if remove_original_sentences and eval_dataset_info.sentence1_key in column_names:
                remove_columns += [eval_dataset_info.sentence1_key]
            if remove_original_sentences and eval_dataset_info.sentence2_key in column_names:
                remove_columns += [eval_dataset_info.sentence2_key]

            remove_columns = list(set(remove_columns))
            eval_ds = eval_datasets.map(_select_preprocess_func(cfg, premise_key=eval_dataset_info.sentence1_key,
                                                                hypothesis_key=eval_dataset_info.sentence2_key,
                                                                is_train=False), batched=True,
                                        remove_columns=remove_columns,
                                        load_from_cache_file=not overwrite_cache)

            test_datasets.append((eval_dataset_name, eval_ds[dataset_tag], eval_dataset_info))

    train_dataset = datasets[dataset_info.train_dataset_name]
    subset_size = cfg.get('subset_size', -1)
    if subset_size != -1:
        original_train_dataset = train_dataset
        all_idxes = np.arange(len(train_dataset)).astype(dtype=np.int)
        idxes = list(np.random.RandomState(26096781 + subset_size).choice(all_idxes, subset_size, replace=False))
        train_dataset = train_dataset.select(idxes)
    else:
        original_train_dataset = train_dataset

    if 'id' not in original_train_dataset.column_names:
        original_train_dataset = original_train_dataset.map(lambda x, idx: {**x, 'id': idx}, batched=True,
                                                            with_indices=True)
    if 'id' not in train_dataset.column_names and run_type == 'debias':
        train_dataset = train_dataset.map(lambda x, idx: {**x, 'id': idx}, batched=True,
                                          with_indices=True)
    if 'id' in train_dataset.column_names and run_type != 'debias':
        train_dataset = train_dataset.map(lambda x: x, remove_columns=['id'], batched=True)

    dev_dataset: Dataset = datasets[dataset_info.dev_dataset_name]
    test_dataset: Dataset = datasets[dataset_info.test_dataset_name]

    test_datasets.append((train_dataset_name, test_dataset, dataset_info))
    return train_dataset, original_train_dataset, dev_dataset, test_datasets


def _select_preprocess_func(cfg: DictConfig, is_train=True, premise_key='premise', hypothesis_key='hypothesis'):
    max_seq_length = cfg.get('max_seq_length')
    loss_fn: str = cfg.get('loss_fn')
    run_type = cfg.get('type')
    bias_type = cfg.get('bias_type')

    if ((is_train and loss_fn.startswith('explicit/'))
            or run_type == 'hans_only'):
        # Explicit Bias, extract bias-only features
        def get_hans_features_single(premise, hypothesis, parse):
            lexical_overlap, overlap_rate = have_lexical_overlap(premise, hypothesis)
            subsequence = is_subsequence(premise, hypothesis)
            result = {
                'subsequence': int(subsequence),
                'lexical_overlap': int(lexical_overlap),
                'overlap_rate': overlap_rate
            }
            if parse is not None:
                constituent = is_constituent(premise, hypothesis, parse)
                result['constituent'] = int(constituent)
            return result

        def get_hans_features(premise, hypothesis, parse):
            if isinstance(premise, list) and isinstance(hypothesis, list):
                assert len(premise) == len(hypothesis), f'Must have an equal number of premises and hypotheses,' \
                                                        f'got {len(premise)} and {len(hypothesis)}'
                if parse is None:
                    parse = [None for _ in premise]
                dicts = [get_hans_features_single(p, h, par) for p, h, par in zip(premise, hypothesis, parse)]
                result = defaultdict(list)
                for d in dicts:
                    for k, v in d.items():
                        result[k].append(v)
                return result
            return get_hans_features_single(premise, hypothesis, parse)

        def preprocess_function(examples):
            # Tokenize the texts
            premise = examples[premise_key]
            hypothesis = examples[hypothesis_key]
            result = tokenizer(premise, hypothesis, max_length=max_seq_length, truncation=True, return_length=True)
            result_premise = tokenizer(premise, max_length=max_seq_length // 2, truncation=True,
                                       padding='max_length')
            result_hypothesis = tokenizer(hypothesis, max_length=max_seq_length // 2, truncation=True,
                                          padding='max_length')

            features = {
                **result,
                'premise_ids': result_premise['input_ids'],
                'premise_attention_mask': result_premise['attention_mask'],
                'hypothesis_ids': result_hypothesis['input_ids'],
                'hypothesis_attention_mask': result_hypothesis['attention_mask'],
            }

            if bias_type == 'hans':
                parse = examples['premise_parse'] if 'premise_parse' in examples else None
                hans_features = get_hans_features(premise, hypothesis, parse)
                features.update(**hans_features)

            return features

        return preprocess_function

    if run_type == 'hypo_only':
        def preprocess_function(examples):
            # Tokenize the texts
            hypo = examples[hypothesis_key]
            result = tokenizer(hypo, max_length=max_seq_length, truncation=True)
            return result

        return preprocess_function

    def preprocess_function(examples):
        # Tokenize the texts
        args = (examples[premise_key], examples[hypothesis_key])
        result = tokenizer(*args, max_length=max_seq_length, truncation=True, return_length=is_train)
        return result

    return preprocess_function


@hydra.main(config_name='training')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    seed = cfg.get('seed', 42)
    set_seed(seed)

    do_train = cfg.get('do_train', True)
    do_eval = cfg.get('do_eval', True)
    tag: str = cfg.get('tag')
    output_dir = cfg.get('output_dir')
    output_model_logits = cfg.get('output_model_logits', False)
    report_to_wandb = do_train or do_eval
    output_dir = join(project_config.TEMP_DIR, output_dir, f'seed:{seed}', tag)

    if report_to_wandb and is_wandb_available():
        import wandb
        wandb.init(project=MODEL_TRAINING_PROJECT_NAME, name=re.sub('seed:[0-9]+/', '', tag))
        wandb.config.update({
            k: v for k, v in cfg.items()
        })

    if not do_eval and not do_train and not output_model_logits:
        raise ValueError('At least one of do_train or do_eval or output_model_logits must be true')

    model, config, trainer_cls = _select_model_configuration(cfg)
    train_dataset, original_train_dataset, dev_dataset, test_datasets = _prepare_datasets(cfg)
    train_ids = None
    if output_model_logits:
        # Note: This assumes that the IDs in the dataset are unique (which is not the case for original FEVER-NLI, so
        #   we modified it).
        train_ids = np.array(original_train_dataset['id'])
    print('Sample #0:')
    pprint(f'{train_dataset[0]}')
    train_parameters = cfg.get('training')
    # Parameters
    num_train_epochs = train_parameters['num_train_epochs']
    # warmup_steps = train_parameters['warmup_steps']
    warmup_ratio = train_parameters['warmup_ratio']
    # adam_epsilon = train_parameters['adam_epsilon']
    weight_decay = train_parameters['weight_decay']
    learning_rate = train_parameters['learning_rate']
    train_batch_size = train_parameters['train_batch_size']
    eval_batch_size = train_parameters['eval_batch_size']
    init_classifier = train_parameters['init_classifier']
    logging_steps = train_parameters['logging_steps']
    eval_steps = train_parameters['eval_steps']

    if init_classifier:
        assert isinstance(model.classifier, nn.Linear)
        model.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if model.classifier.bias is not None:
            model.classifier.bias.data.zero_()
        print('Initialized classifier weights')

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=num_train_epochs,  # total # of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
        # warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,  # strength of weight decay
        learning_rate=learning_rate,
        # adam_epsilon=adam_epsilon,
        # logging_dir=logging_dir,  # directory for storing logs
        evaluation_strategy=IntervalStrategy.STEPS,
        # dataloader_drop_last=True,
        run_name=tag,
        save_steps=5000,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        remove_unused_columns=isinstance(model, BertForSequenceClassification),
        skip_memory_metrics=True,
        group_by_length=True,
        seed=seed,
        report_to=['wandb'] if report_to_wandb else []
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        tokenizer=tokenizer
    )

    model_dir = join(project_config.MODELS_DIR, f'seed:{seed}', tag)
    print(f'Model will be saved at {os.path.abspath(model_dir)}')

    model_dir = join(project_config.MODELS_DIR, f"seed:{seed}", tag)
    if do_train:
        if 'model_path' in cfg.keys():
            print('Loading checkpoint...')
            train_output = trainer.train(model_path=cfg.get('model_path'))
        else:
            train_output = trainer.train()
        print(f'Train outputs: {train_output}')
        print(f'Saving model to {os.path.abspath(model_dir)}')
        trainer.save_model(model_dir)
    if do_eval:
        for eval_ds_name, eval_ds, info in test_datasets:
            print(f'Sample #0 ({eval_ds_name}):')
            pprint(f'{eval_ds[0]}')
            if info.binerize:
                # Binerization is needed because some datasets (like HANS, FEVER-Symmetric)
                # have 2 classes, while the model is trained on standard NLI (3 classes)
                def binerize_fn(pred: EvalPrediction):
                    print(f'Binerizing dataset {eval_ds_name}')
                    preds = pred.predictions.argmax(-1)
                    # (Entailment, Neutral, Contradiction)

                    # Neutral => Contradiction
                    preds[preds == 1] = 2
                    # Contradiction (2) => Contradiction (1)
                    preds[preds == 2] = 1

                    return EvalPrediction(predictions=preds, label_ids=pred.label_ids)

                def compute_metrics_binerized(pred):
                    y_true = pred.label_ids
                    y_pred = pred.predictions
                    labels = np.unique(pred.label_ids)
                    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, digits=3,
                                                   labels=labels)
                    return {
                        eval_ds_name: {
                            **per_class_accuracy(y_true, y_pred),
                            'accuracy': report['accuracy'],
                        }
                    }

                compute_metrics = compute_metrics_wrap(compute_metrics_binerized, binerize_fn)
            else:
                def compute_metrics_with_name(pred):
                    results = compute_metrics_default(pred)
                    return {
                        eval_ds_name: results
                    }

                compute_metrics = compute_metrics_with_name

            trainer.compute_metrics = compute_metrics
            if info.mapper is not None:
                original_compute_metrics = trainer.compute_metrics

                def mapped_compute_metrics(pred: EvalPrediction):
                    print(f'Mapping predictions of {eval_ds_name}')
                    pred = EvalPrediction(predictions=info.mapper(pred.predictions), label_ids=pred.label_ids)
                    return original_compute_metrics(pred)

                trainer.compute_metrics = mapped_compute_metrics

            print(f'[{eval_ds_name}] Evaluation outputs:')
            trainer.evaluate(eval_dataset=eval_ds)
    if output_model_logits:
        assert train_ids is not None
        output: PredictionOutput = trainer.predict(original_train_dataset)
        predictions = output.predictions
        train_ids = train_ids.reshape(-1, 1)  # Row => Column
        ids_predictions = np.hstack((train_ids, predictions))
        pd.DataFrame(ids_predictions, columns=['id'] + [f'label_{i}' for i in
                                                        range(train_dataset.features['label'].num_classes)]) \
            .set_index('id').to_csv(
            join(model_dir, 'train_predictions.csv'))


if __name__ == '__main__':
    # Disable gradient watch
    os.environ['WANDB_WATCH'] = 'false'
    main()

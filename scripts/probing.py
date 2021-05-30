import datetime
import json
import os
import pickle
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from os.path import join
from typing import List

import torch
import wandb
from nltk import word_tokenize
from torch import nn
from torch.utils.data import random_split
from transformers import set_seed, is_wandb_available, BertTokenizerFast, HfArgumentParser, BertConfig, BertModel, \
    WEIGHTS_NAME, BertForSequenceClassification

import config as project_config
from bias_probing.data.caching import get_cached_dataset, embed_and_cache
from bias_probing.data.datasets import load_dataset_aux, DatasetInfo
from bias_probing.heuristics import sentence_to_words, have_lexical_overlap, is_subsequence
from bias_probing.mdl import OnlineCodeMDLProbe
from bias_probing.modelling import BertWithWeakLearner, BertWithExplicitBias
from bias_probing.utils.ngrams import ngram_contained

# Faster tokenizer for optimization
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


ONLINE_CODE_PROJECT_NAME = 'online-code'


@dataclass
class ExperimentArguments:
    name: str
    task_config_file: str
    model_name_or_path: str
    seed: int = field(default=42)
    probe_type: str = field(default='linear', metadata={'choices': ['linear', 'mlp']})
    mdl_fractions: List[float] = field(default_factory=lambda: [2.0, 3.0, 4.4, 6.5, 9.5, 14.0,
                                                                21.0, 31.0, 45.7, 67.6, 100])
    embedding_size: int = field(default=768)
    max_seq_length: int = field(default=128)
    batch_size: int = field(default=64)
    hypothesis_only: bool = field(default=False)
    min_dataset_size: int = field(default=100)
    new_split_ratio: int = field(default=0.8)
    wandb_project_name: str = field(default=ONLINE_CODE_PROJECT_NAME)
    config_dir: str = field(default='configs')
    logging_dir: str = field(default='outputs')
    cache_only: bool = False
    overwrite_cache: bool = False


@dataclass
class TaskConfig:
    name: str
    type: str
    dataset_name: str
    mapper_kwargs: dict

    @staticmethod
    def from_dict(config: dict):
        return TaskConfig(config['name'], config['type'], config['dataset_name'], config['kwargs'])


@dataclass
class OnlineCodingExperimentResults:
    name: str
    uniform_cdl: float
    online_cdl: float
    compression: float
    report: dict
    fractions: List[float]


def build_probe(input_size, num_classes=2, probe_type='mlp'):
    probes = {
        'mlp': lambda: nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, num_classes)
        ),
        'linear': lambda: nn.Linear(input_size, num_classes)
    }
    return probes[probe_type]()


# Adapter function for fine-tuned (MNLI -> HANS) model from https://github.com/UKPLab/acl2020-confidence-regularization
def get_conf_reg_bert(model_name_or_path: str):
    config = BertConfig.from_pretrained(model_name_or_path)
    state_dict = torch.load(join(model_name_or_path, WEIGHTS_NAME))
    conf_reg_module_keys = [k[len('bert.'):] for k in state_dict.keys() if k.startswith('bert.')]

    bert_base = BertModel.from_pretrained('bert-base-uncased')
    bert_base_module_keys = list(bert_base.state_dict().keys())

    old_state_dict = state_dict
    new_state_dict = OrderedDict()
    for k in conf_reg_module_keys:
        new_state_dict[k] = old_state_dict[f'bert.{k}']

    # Version mismatch fix
    # https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertEmbeddings
    new_state_dict['embeddings.position_ids'] = torch.arange(config.max_position_embeddings).expand((1, -1))
    assert set() == set(new_state_dict.keys()).symmetric_difference(set(bert_base_module_keys))

    bert_conf_reg = BertModel(config)
    bert_conf_reg.load_state_dict(new_state_dict)
    return bert_conf_reg, config


def build_encoder(model_name_or_path, return_tokenizer=False, base_dir=None):
    if base_dir is None:
        base_dir = project_config.MODELS_DIR

    hypo_only = None
    if model_name_or_path == 'bert-random':
        config = BertConfig.from_pretrained('bert-base-uncased')
        encoder = BertModel(config)
        encoder.apply(encoder._init_weights)
    elif 'bert_confreg' in model_name_or_path:
        real_path = join(base_dir, model_name_or_path)
        encoder, config = get_conf_reg_bert(real_path)
    elif os.path.isdir(join(project_config.MODELS_DIR, model_name_or_path)):
        real_path = join(project_config.MODELS_DIR, model_name_or_path)
        config = BertConfig.from_pretrained(real_path)
        if config.architectures and 'BertWithWeakLearner' in config.architectures:
            bert_with_weak = BertWithWeakLearner.from_pretrained(real_path)
            # BertForSequenceClassification -> BertModel
            encoder = bert_with_weak.bert.bert
        elif config.architectures and 'BertWithExplicitBias' in config.architectures:
            bert_with_weak = BertWithExplicitBias.from_pretrained(real_path)
            encoder = bert_with_weak.bert
        else:
            model = BertForSequenceClassification.from_pretrained(real_path)
            encoder = model.bert
        assert isinstance(encoder, BertModel)
    else:
        config = BertConfig.from_pretrained(model_name_or_path)
        encoder = BertModel.from_pretrained(model_name_or_path, from_tf=False, config=config)

    if return_tokenizer:
        tokenizer = build_tokenizer(model_name_or_path)
        if hypo_only is not None:
            return encoder, config, hypo_only, tokenizer
        return encoder, config, tokenizer

    if hypo_only is not None:
        return encoder, config, hypo_only
    return encoder, config


def build_tokenizer(model_name_or_path, base_dir=None):
    if base_dir is None:
        base_dir = project_config.MODELS_DIR
    if model_name_or_path in ['bert-random']:
        return BertTokenizerFast.from_pretrained('bert-base-uncased')

    real_path = join(base_dir, model_name_or_path)
    if os.path.isdir(real_path):
        if os.path.isfile(join(real_path, 'vocab.txt')):
            return BertTokenizerFast.from_pretrained(real_path)
        return BertTokenizerFast.from_pretrained('bert-base-uncased')
    return BertTokenizerFast.from_pretrained(model_name_or_path)


class NegWordsMapper:
    # noinspection PyUnusedLocal
    def __init__(self, negative_vocab=None, n=1, hypothesis_key='hypothesis', **kwargs):
        self.negative_vocab = [tuple(tokenizer.tokenize(exp)) for exp in negative_vocab]
        print(self.negative_vocab)
        self.n = n
        self.hypothesis_key = hypothesis_key

    def __call__(self, examples):
        ngrams = self.negative_vocab
        n = self.n

        hypothesis = examples[self.hypothesis_key]
        hypo_words = [sentence_to_words(h) for h in hypothesis] if type(hypothesis) == list \
            else sentence_to_words(hypothesis)
        probing_label = [int(any(ngram_contained(words, bi, n) for bi in ngrams)) for words in hypo_words]
        return {
            **examples,
            'probing_label': probing_label
        }


class LexClassMapper:
    # noinspection PyUnusedLocal
    def __init__(self, heuristic='lex', premise_key='premise', hypothesis_key='hypothesis', **kwargs):
        self.heuristic = heuristic
        self.premise_key = premise_key
        self.hypothesis_key = hypothesis_key
        print(f'LexClassMapper(heuristic={heuristic})')

    def __call__(self, examples):
        prem_list = examples[self.premise_key]
        hypo_list = examples[self.hypothesis_key]
        if self.heuristic == 'lex':
            def label_fn(p, h):
                return int(have_lexical_overlap(p, h)[0])
        elif self.heuristic == 'sub':
            def label_fn(p, h):
                return int(is_subsequence(p, h))
        else:
            raise ValueError(f'Invalid heuristic {self.heuristic}')

        examples['probing_label'] = [label_fn(premise, hypothesis) for premise, hypothesis in zip(prem_list, hypo_list)]
        return examples


def get_mapper(config: TaskConfig, dataset_info: DatasetInfo):
    if config.type == 'lex-class':
        return LexClassMapper(**config.mapper_kwargs,
                              premise_key=dataset_info.sentence1_key,
                              hypothesis_key=dataset_info.sentence2_key)
    elif config.type == 'neg-words':
        return NegWordsMapper(**config.mapper_kwargs, hypothesis_key=dataset_info.sentence2_key)
    raise ValueError(f'Unsupported config type {config.type}')


def prepare_dataset(args: ExperimentArguments, config: TaskConfig, _embed_and_cache_flag=True):
    """Prepare and cache the target dataset for a probing task described by :config:
    :param args: Experiment arguments
    :param config: Task configuration
    :return: A tuple containing (train_dataset, dev_dataset, test_dataset), the split of which is determined by
    the `args` parameter
    """
    task_name = config.name
    model_name_or_path = args.model_name_or_path
    overwrite_cache = args.overwrite_cache
    output_dir = join(project_config.ENCODED_DATA_DIR, model_name_or_path)
    try:
        if args.overwrite_cache:
            raise IOError()  # To re-cache the datasets
        train_dataset = get_cached_dataset(task_name, 'train', output_dir, verbose=False)
        val_dataset = get_cached_dataset(task_name, 'val', output_dir, verbose=False)
        test_dataset = get_cached_dataset(task_name, 'test', output_dir, verbose=False)
        return train_dataset, val_dataset, test_dataset
    except IOError:
        pass

    # Parameters
    dataset_name = config.dataset_name
    task_name = config.name

    ds, dataset_info = load_dataset_aux(dataset_name)
    model_name_or_path = args.model_name_or_path
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    hypo_only = config.mapper_kwargs.pop('hypothesis_only', False)
    premise_key = dataset_info.sentence1_key
    hypothesis_key = dataset_info.sentence2_key
    task_mapper = get_mapper(config, dataset_info)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = build_encoder(model_name_or_path)[0]

    def preprocess_function(examples):
        # Tokenize the texts
        premise = examples[premise_key]  # List
        hypothesis = examples[hypothesis_key]  # List

        result = examples
        if hypo_only:
            result.update(
                tokenizer(hypothesis, padding='max_length', max_length=max_seq_length, truncation=True))
        else:
            result.update(
                tokenizer(premise, hypothesis, padding='max_length', max_length=max_seq_length, truncation=True))
        return task_mapper(result)

    mapped_ds = ds.map(preprocess_function, batched=True, load_from_cache_file=not overwrite_cache)

    train_dataset = mapped_ds[dataset_info.train_dataset_name]
    dev_dataset = mapped_ds[dataset_info.dev_dataset_name]
    test_dataset = mapped_ds[dataset_info.test_dataset_name]
    if not _embed_and_cache_flag:
        return train_dataset, dev_dataset, test_dataset

    def collate(samples):
        inputs = {
            'input_ids': torch.tensor([sample['input_ids'] for sample in samples], device=device),
            'attention_mask': torch.tensor([sample['attention_mask'] for sample in samples], device=device),
            'token_type_ids': torch.tensor([sample['token_type_ids'] for sample in samples], device=device)
        }
        probing_labels = torch.tensor([sample['probing_label'] for sample in samples])
        return (
            torch.zeros_like(probing_labels),  # Quickfix
            inputs,
            probing_labels
        )

    output_dir = join(project_config.CACHE_DIR, '_encodings')
    real_path = os.path.join(output_dir, model_name_or_path)
    train_dataset = embed_and_cache(task_name, train_dataset, 'train', encoder, real_path, collate_fn=collate,
                                    batch_size=batch_size, device=device, overwrite_cache=overwrite_cache)
    dev_dataset = embed_and_cache(task_name, dev_dataset, 'val', encoder, real_path, collate_fn=collate,
                                  batch_size=batch_size, device=device, overwrite_cache=overwrite_cache)
    test_dataset = embed_and_cache(task_name, test_dataset, 'test', encoder, real_path, collate_fn=collate,
                                   batch_size=batch_size, device=device, overwrite_cache=overwrite_cache)
    return train_dataset, dev_dataset, test_dataset


def run_online_coding_experiment(args: ExperimentArguments, config: TaskConfig, device=None):
    # Parameters
    min_dataset_size = args.min_dataset_size
    new_split_ratio = args.new_split_ratio
    seed = args.seed
    encoding_size = args.embedding_size
    probe_type = args.probe_type
    fractions = args.mdl_fractions
    name = args.name
    task_name = config.name

    # For reproducibility
    set_seed(seed)

    # Data
    train_dataset, val_dataset, test_dataset = prepare_dataset(args, config)

    if len(val_dataset) < min_dataset_size or len(test_dataset) < min_dataset_size:
        size_train = int(len(train_dataset) * new_split_ratio)
        size_val = (len(train_dataset) - size_train) // 2
        size_test = len(train_dataset) - size_train - size_val
        train_dataset, val_dataset, test_dataset = random_split(train_dataset, (size_train, size_val, size_test),
                                                                generator=torch.Generator().manual_seed(42))

    def create_probe():
        return build_probe(encoding_size, probe_type=probe_type)

    online_code_probe = OnlineCodeMDLProbe(create_probe, fractions)

    reporting_root = join(os.getcwd(), f'online_coding_{task_name}.pkl')
    uniform_cdl, online_cdl = online_code_probe.evaluate(train_dataset, test_dataset, val_dataset,
                                                         reporting_root=reporting_root, verbose=True, device=device)
    compression = round(uniform_cdl / online_cdl, 2)
    report = online_code_probe.load_report(reporting_root)
    exp_results = OnlineCodingExperimentResults(
        name=name,
        uniform_cdl=uniform_cdl,
        online_cdl=online_cdl,
        compression=compression,
        report=report,
        fractions=fractions
    )
    return exp_results


def task_config(args: ExperimentArguments) -> TaskConfig:
    task_config_file = args.task_config_file
    config_dir = args.config_dir

    config_file_path = join(TOP_DIR, config_dir, task_config_file)
    if not os.path.isfile(config_file_path):
        raise ValueError(f'No valid configuration file found at {config_file_path}')
    with open(config_file_path) as json_file:
        config = json.load(json_file)
    return TaskConfig.from_dict(config)


def _wandb_init(args: ExperimentArguments, config: TaskConfig):
    experiment_name = args.name
    task_name = config.name

    if is_wandb_available():
        dt = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        run_name = f'{experiment_name}/{dt}'
        group_name = f'{task_name}/{experiment_name}'
        wandb.init(project=args.wandb_project_name, name=run_name, group=group_name)
        wandb.config.update({
            k: v for k, v in args.__dict__.items()
        })
        wandb.config.update({
            f'task_{k}': v for k, v in asdict(config).items()
        })


def _working_dir_init(args: ExperimentArguments):
    logging_dir = args.logging_dir
    output_dir = join(logging_dir, datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)


def _dump_report(exp_results: OnlineCodingExperimentResults):
    report = exp_results.report
    print(report['eval']['classification_report'])

    results_dir = os.getcwd()
    p = join(results_dir, f'results.pkl')
    with open(p, 'wb') as f:
        pickle.dump(exp_results, f)

    p_json = join(results_dir, f'results.json')
    with open(p_json, 'w') as f:
        json.dump(asdict(exp_results), f)

    if is_wandb_available():
        # Save results (as JSON)
        wandb.save(p_json)

    # all_results_file = join(project_config.TEMP_DIR, cfg.get('results_output_file_name', 'results.csv'))
    #
    # ignored_keys = ['results_output_file_name']
    # cfg_keys = [k for k in list(cfg.keys()) if k not in ignored_keys]
    # if not os.path.exists(all_results_file):
    #     with open(all_results_file, 'w') as out_csv:
    #         writer = csv.writer(out_csv)
    #         writer.writerow(
    #             ['output_dir'] + cfg_keys + ['name', 'online_cdl', 'uniform_cdl', 'compression',
    #                                          'eval_accuracy'])
    #
    # with open(all_results_file, 'a') as out_csv:
    #     writer = csv.writer(out_csv)
    #     writer.writerow(
    #         [results_dir] + [cfg[k] for k in cfg_keys] + [results.name, results.online_cdl, results.uniform_cdl,
    #                                                       results.compression, report['training']['accuracy']])


def main(args: ExperimentArguments):
    config = task_config(args)

    _working_dir_init(args)
    print(f'Working directory : {os.getcwd()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    _wandb_init(args, config)

    results = run_online_coding_experiment(args, config, device=device)
    _dump_report(results)


def test():
    print('TEST')
    args = ExperimentArguments('test', 'mnli_lex_class.json', 'bert-base-uncased', cache_only=True,
                               overwrite_cache=True)
    config = task_config(args)
    train, _, _ = prepare_dataset(args, config, _embed_and_cache_flag=False)
    count = [0, 0, 0]
    samples = []
    for s in train:
        if s['probing_label'] == 1:
            count[s['label']] += 1
            if s['label'] == 2:
                samples.append((sentence_to_words(s['premise']), sentence_to_words(s['hypothesis']), s['label']))
    print('Counts:', count)
    orig_samples = []
    with open(join(project_config.TEMP_DIR, 'mnli_contradicting_examples.txt'), 'r') as f:
        for line in f:
            orig_samples.append(tuple(line.strip().split('\t')))

    intersect = []
    disjunct = []
    label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    for premise, hypothesis, label in orig_samples:
        premise = sentence_to_words(word_tokenize(premise))
        hypothesis = sentence_to_words(word_tokenize(hypothesis))
        label = label_to_id[label]
        appended = False
        for p2, h2, l2 in samples:
            if p2 == premise and h2 == hypothesis and l2 == label:
                intersect.append((premise, hypothesis, label))
                appended = True
                break
        if appended:
            continue
        disjunct.append((premise, hypothesis, label))

    print('Intersect:', len(intersect))
    print(intersect)
    print('Disjunct:', len(disjunct))
    print(disjunct)
    return args, config


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArguments)
    # noinspection PyTypeChecker
    _args: ExperimentArguments = parser.parse_args()
    if os.environ.pop('TEST', None) == '1':
        test()
    elif _args.cache_only:
        prepare_dataset(_args, task_config(_args))
    else:
        main(_args)

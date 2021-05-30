import os
import sys
from dataclasses import field, dataclass
from os.path import join

import matplotlib as mpl

# MPL settings, do not modify
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForSequenceClassification, BertTokenizerFast, \
    default_data_collator

sys.path.append(join(os.getcwd(), 'src'))
from data.datasets import load_dataset_aux
import config as project_config


@dataclass
class Arguments:
    model_name_or_path: str
    dataset_name: str
    hypothesis_only: bool = field(default=False)
    max_seq_length: int = field(default=128)
    batch_size: int = field(default=64)
    overwrite_cache: bool = field(default=False)
    dataset_types: list = field(default_factory=lambda: ['dev', 'test'])


def real_model_path(model_name_or_path):
    if os.path.isdir(join(project_config.MODELS_DIR, model_name_or_path)):
        return join(project_config.MODELS_DIR, model_name_or_path)
    return model_name_or_path


def main():
    parser = HfArgumentParser(Arguments)
    # noinspection PyTypeChecker
    args: Arguments = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    dataset_name = args.dataset_name
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    hypo_only = args.hypothesis_only
    overwrite_cache = args.overwrite_cache

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(real_model_path(model_name_or_path))
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    datasets, dataset_info = load_dataset_aux(dataset_name)

    if hypo_only:
        def preprocess_function(examples):
            # Tokenize the texts
            result = tokenizer(examples[dataset_info.sentence2_key],
                               padding='max_length', max_length=max_seq_length, truncation=True)
            return result
    else:
        def preprocess_function(examples):
            # Tokenize the texts
            result = tokenizer(examples[dataset_info.sentence1_key], examples[dataset_info.sentence2_key],
                               padding='max_length', max_length=max_seq_length, truncation=True)
            return result

    datasets = datasets.map(preprocess_function, batched=True, remove_columns=dataset_info.ignored_keys)

    train_dataset = datasets[dataset_info.train_dataset_name]
    dev_dataset = datasets[dataset_info.dev_dataset_name]
    test_dataset = datasets[dataset_info.test_dataset_name]

    print('Sample #1:')
    print(dev_dataset[0])

    mapping = {
        'train': train_dataset,
        'dev': dev_dataset,
        'test': test_dataset
    }

    model.eval()
    for name in args.dataset_types:
        ds = mapping[name]
        temp_dir = join(project_config.TEMP_DIR, 'confidence_bins')
        temp_file = join(temp_dir, f'confidence_bins_{dataset_name}_{name}_{model_name_or_path.replace("/", "_")}.pkl')
        if os.path.exists(temp_file) and not overwrite_cache:
            obj = torch.load(temp_file)
            preds = obj['preds']
            labels = obj['labels']
        else:
            with torch.no_grad():
                print(f'Generating predictions for {name}...')
                dl = DataLoader(ds, batch_size=batch_size, collate_fn=default_data_collator)
                labels = torch.tensor([])
                preds = torch.tensor([])
                for batch in tqdm(dl, desc='Predicting'):
                    y_true = batch.pop('labels')
                    inputs = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device)

                    outputs = model(**inputs)
                    logits = softmax(outputs[0], dim=-1)
                    preds = torch.cat((preds.cpu(), logits.cpu()), dim=0)
                    labels = torch.cat((labels.cpu(), y_true.cpu()), dim=0)

            temp_dir = join(project_config.TEMP_DIR, 'confidence_bins')
            os.makedirs(temp_dir, exist_ok=True)
            torch.save({
                'labels': labels,
                'preds': preds,
            }, join(temp_dir, temp_file))

        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y_pred = preds.argmax(dim=-1)
        y_true = labels
        plt.hist(preds.max(dim=-1).values, bins, color='red', alpha=0.5, label='Number of predictions')
        plt.hist(preds[y_true == y_pred].max(dim=-1).values, bins, color='blue', alpha=0.5, label='Correctly predicted')
        plt.title(name)
        plt.legend()
        output_dir = join(project_config.FIGURES_DIR, 'confidence_bins')
        os.makedirs(output_dir, exist_ok=True)
        output_file = join(output_dir,
                           f'confidence_bins_{dataset_name}_{name}_{model_name_or_path.replace("/", "_")}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()


if __name__ == '__main__':
    main()

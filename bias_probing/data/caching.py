import os

import numpy as np
import torch
from datasets import concatenate_datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm


def get_cached_dataset(task_name: str, set_type, output_dir, verbose=True) -> torch.utils.data.Dataset:
    assert set_type in {'train', 'val', 'test'}
    file_name = f'{task_name}_{set_type}_cached.pkl'
    output_file = os.path.abspath(os.path.join(output_dir, file_name))
    if os.path.isfile(output_file):
        if verbose:
            print(f'Found cached dataset in {output_file}')
        data = torch.load(output_file)
        all_guid = data['guid']
        all_embeddings = data['examples']
        all_labels = data['labels']
        return TensorDataset(all_guid, all_embeddings, all_labels)
    raise IOError(f'Database not found at {output_file}')


def embed_and_cache(cache_tag: str, dataset, set_type, encoder, output_dir, balance=True, collate_fn=None,
                    batch_size=32, device=None, overwrite_cache=False):
    assert set_type in {'train', 'val', 'test'}
    file_name = f'{cache_tag}_{set_type}_cached.pkl'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.abspath(os.path.join(output_dir, file_name))
    if overwrite_cache:
        print(f'Overwriting cached files...')
    elif os.path.isfile(output_file):
        print(f'Found cached dataset in {output_file}')
        data = torch.load(output_file)
        all_guid, all_embeddings, all_labels = data['guid'], data['examples'], data['labels']
        return TensorDataset(all_guid, all_embeddings, all_labels)
    print(f'Caching into {output_file}')

    all_labels = torch.tensor(dataset['probing_label'])
    # Only supports binary classification (currently)
    num_positive_samples = (all_labels == 1).sum().item()
    print('total samples: ', len(all_labels))
    print('positive samples: ', num_positive_samples)

    if balance:
        ds_seed = 42
        positive_dataset = dataset.filter(lambda x: x['probing_label'] == 1, load_from_cache_file=not overwrite_cache)
        negative_dataset = dataset.filter(lambda x: x['probing_label'] == 0, load_from_cache_file=not overwrite_cache) \
            .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache) \
            .select(np.arange(positive_dataset.num_rows))

        dataset = concatenate_datasets([positive_dataset, negative_dataset])\
            .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache)
        num_samples = dataset.num_rows
    else:
        num_samples = len(all_labels)
        # sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_guid = None
    all_embeddings = None
    all_labels = None
    if num_samples > 0:
        encoder = encoder.to(device)
        t = tqdm(dataloader, desc='Embedding')
        with torch.no_grad():
            for batch in t:
                guid = batch[0]
                inputs = batch[1]
                label = batch[2]
                _, embedding = encoder(**inputs, return_dict=False)
                if all_guid is None:
                    all_guid = guid.numpy()
                    all_embeddings = embedding.cpu().numpy()
                    all_labels = label.numpy()
                all_guid = np.append(all_guid, guid, axis=0)
                all_embeddings = np.append(all_embeddings, embedding.cpu().numpy(), axis=0)
                all_labels = np.append(all_labels, label.numpy(), axis=0)

        all_guid = torch.tensor(all_guid)
        all_embeddings = torch.tensor(all_embeddings)
        all_labels = torch.tensor(all_labels)
    else:
        all_guid = all_embeddings = all_labels = torch.tensor([])

    torch.save({
        'guid': all_guid,
        'examples': all_embeddings,
        'labels': all_labels
    }, output_file)
    dataset = TensorDataset(all_guid, all_embeddings, all_labels)
    print('new dataset size: ', len(dataset))
    return dataset

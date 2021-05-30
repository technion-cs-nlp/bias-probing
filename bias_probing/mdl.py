import abc
import pickle
from abc import ABC
from typing import List, Callable, Union, Sized

import numpy as np
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset, Subset, IterableDataset
from tqdm import tqdm
import wandb
from transformers.integrations import is_wandb_available

from probes import train_probe, evaluate_probe, ProbeTrainingArgs


class MDLProbe(ABC):
    r"""This class represents a generic probing model which is evaluated using MDL.

    For a concrete implementation, take a look at OnlineCodeMDLProbe
    """

    def __init__(self, model_class: Callable):
        r"""Initialize the probe with a given model class.

        Parameters:
            model_class (Callable): A Callable for building the probe. Assumed to expect no parameters such that a
            probing classifier can be constructed with a call to ``model_class()``

        """
        super().__init__()
        self.model_class = model_class

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the probe given the MDL metric"""
        raise NotImplementedError()


class OnlineCodeMDLProbe(MDLProbe):
    r"""An MDL probe with Online Coding evaluation.
    Implementation is adapted from Voita and Titov 2020 (https://arxiv.org/pdf/2003.12298.pdf)
    """

    def __init__(self, model_class, fractions: List[float], device=None):
        super().__init__(model_class)
        self.fractions = fractions
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)

    @staticmethod
    def split_datasets(dataset: Union[Dataset, Sized], fractions: List[float]):
        r"""Split a dataset into portions, given by :fractions:

        Returns a tuple containing 2 lists of size len(fractions) and len(fractions) - 1,
        the first being the train datasets and the latter being the corresponding evaluation (test) datasets.
        The last fraction is always assumed to be 100 (The full dataset).

        Parameters:
            dataset (Dataset): The dataset to split into fractions
            fractions (List[int]): The list of fractions. This should be a monotonically increasing list of integers
            with values between 0 and 100. The last item is assumed to be 100.

        Returns:
            train_portions: A list of Subsets, of size len(fractions)
            eval_portions: A list of Subsets, of size len(fractions) - 1
        """
        if isinstance(Dataset, IterableDataset):
            raise ValueError('dataset must not be of type InstanceDataset, and must implement a __getitem__ method')
        # Normalize to [0, 1]
        fractions = [i / 100 for i in fractions]
        total_len = len(dataset)

        train_portions = []
        eval_portions = []
        for i in range(len(fractions)):
            train_subset = Subset(dataset, range(0, int(fractions[i] * total_len)))
            train_portions.append(train_subset)
            if i != len(fractions) - 1:
                # Last dataset does not have a corresponding evaluation set
                eval_subset = Subset(dataset, range(int(fractions[i] * total_len), int(fractions[i + 1] * total_len)))
                eval_portions.append(eval_subset)

        return train_portions, eval_portions

    @staticmethod
    def save_report(reporting_root, results: dict):
        r"""Save a report for this probe, after training.

        The report is saved as online_coding.pkl in the given directory.
        """
        pickle.dump(results, open(reporting_root, 'wb'))

    @staticmethod
    def load_report(reporting_root: str):
        r"""Load a saved report for this probe, after it was trained.

        Parameters:
            reporting_root (str): A directory in which the report will be saved.

        Returns:
            report (dict): The saved report, containing two variables: online_coding_list and accuracy.
        """
        return pickle.load(open(reporting_root, 'rb'))

    @staticmethod
    def uniform_code_length(num_classes: int, train_dataset_size: int):
        r"""Calculate the uniform code length for a given training task

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            train_dataset_size (int): The size of the full training dataset which the probe was trained on.

        Returns:
            uniform_code_length (float): The uniform code length for the given training/evaluation parameters of
            the probe.
        """
        return train_dataset_size * np.log2(num_classes)

    @staticmethod
    def online_code_length(num_classes: int, t1: int, losses: List[float]):
        r"""Calculate the online code length.

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            t1 (int): The size of the first training block (fraction) dataset.
            losses (List[float]): The list of (test) losses for each evaluation block (fraction)
            dataset, of size len(fractions).

        Returns:
            online_code_length (float): The online code length for the given training/evaluation parameters of
            the probe.
        """
        return t1 * np.log2(num_classes) + sum(losses)

    def _training_step(self, args: ProbeTrainingArgs,
                       train_ds: Union[Sized, Dataset],
                       dev_ds: Union[Sized, Dataset],
                       eval_ds: Union[Sized, Dataset],
                       collate_fn):
        # Fresh probe instance
        probe = self.model_class().to(self.device)
        loss_fn = self.loss_fn

        acc_list, loss_list = train_probe(args, train_ds, probe,
                                          dev_dataset=dev_ds,
                                          loss_fn=loss_fn,
                                          collate_fn=collate_fn)
        eval_loss, y_true, preds = evaluate_probe(args, eval_ds, probe,
                                                  device=self.device,
                                                  loss_fn=loss_fn,
                                                  collate_fn=collate_fn)
        return {
            'train': (acc_list, loss_list),
            'eval': (eval_loss, y_true, preds)
        }

    def evaluate(self, train_dataset: Dataset, test_dataset: Union[Dataset, Sized], dev_dataset: Dataset = None,
                 train_batch_size=16, learning_rate=1e-3, num_train_epochs=50,
                 checkpoint_steps=10, early_stopping=4, early_stopping_tolerance=1e-3,
                 reporting_root=None,
                 verbose=False, device=None, collate_fn=None):
        r"""Evaluate the probe and return the online and uniform code lengths."""

        train_datasets_list, eval_datasets_list = self.split_datasets(train_dataset, self.fractions)
        assert len(train_datasets_list) == len(eval_datasets_list) + 1
        # Real training dataset
        train_dataset = train_datasets_list[-1]
        # Online code training fractions
        train_datasets_list = train_datasets_list[:-1]

        online_coding_list = []
        args = ProbeTrainingArgs(
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            checkpoint_steps=checkpoint_steps,
            early_stopping=early_stopping,
            early_stopping_tolerance=early_stopping_tolerance
        )

        if is_wandb_available():
            wandb.config.update(args.__dict__)

        for index, (train_ds, eval_ds) in tqdm(enumerate(zip(train_datasets_list, eval_datasets_list)),
                                               desc='Online Code training',
                                               total=len(train_datasets_list),
                                               disable=not verbose):
            res = self._training_step(args,
                                      train_ds=train_ds,
                                      dev_ds=dev_dataset,
                                      eval_ds=eval_ds,
                                      collate_fn=collate_fn)
            fraction = self.fractions[index]
            eval_loss = res['eval'][0]
            online_coding_list.append({
                'fraction': fraction,
                'eval_loss': eval_loss,
                'train_acc_list': res['train'][0],
                'train_loss_list': res['train'][1]
            })
            if is_wandb_available():
                wandb.log({
                    'loss': eval_loss,
                    'fraction': fraction
                })

        res = self._training_step(args, train_dataset, dev_dataset, test_dataset, collate_fn)

        eval_loss, y_true, preds = res['eval']
        y_pred = np.argmax(preds, axis=1)
        correct = (y_true == y_pred).sum()
        accuracy = correct / len(test_dataset)

        # save results
        self.save_report(reporting_root, {
            'online_coding_list': online_coding_list,
            'training': {
                'eval_loss': eval_loss,
                'train_acc_list': res['train'][0],
                'train_loss_list': res['train'][1],
                'accuracy': accuracy
            },
            'eval': {
                'classification_report': metrics.classification_report(y_true, y_pred)
            }
        })

        num_classes = len(np.unique(y_pred))
        train_dataset_size = len(train_dataset)
        uniform_cdl = self.uniform_code_length(num_classes, train_dataset_size)
        online_cdl = self.online_code_length(num_classes, len(train_datasets_list[0]),
                                             list(map(lambda obj: obj['eval_loss'], online_coding_list)))

        if is_wandb_available():
            wandb.run.summary['eval_loss'] = eval_loss
            wandb.run.summary['uniform_cdl'] = uniform_cdl
            wandb.run.summary['online_cdl'] = online_cdl
            wandb.run.summary['compression'] = round(uniform_cdl / online_cdl, 2)
            wandb.run.summary['eval_accuracy'] = accuracy
        return uniform_cdl, online_cdl

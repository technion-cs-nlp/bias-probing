from transformers.trainer import Trainer
from transformers import (
    EvalPrediction
)
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


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


class MultiPredictionDatasetTrainer(Trainer):
    def __init__(
        self,
        eval_datasets = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_datasets = eval_datasets

    def evaluate(
        self,
        eval_datasets=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_datasets is None:
            eval_datasets = self.eval_datasets
        assert eval_datasets is not None

        all_metrics = {}
        for eval_dataset_name, eval_dataset, eval_info in eval_datasets:
            if eval_info.binerize:
                # Binerization is needed because some datasets (like HANS, FEVER-Symmetric)
                # have 2 classes, while the model is trained on standard NLI (3 classes)
                def binerize_fn(pred: EvalPrediction):
                    print(f'Binerizing dataset {eval_dataset_name}')
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
                        eval_dataset_name: {
                            **per_class_accuracy(y_true, y_pred),
                            'accuracy': report['accuracy'],
                        }
                    }

                compute_metrics = compute_metrics_wrap(compute_metrics_binerized, binerize_fn)
            else:
                def compute_metrics_with_name(pred):
                    results = compute_metrics_default(pred)
                    return {
                        eval_dataset_name: results
                    }

                compute_metrics = compute_metrics_with_name

            self.compute_metrics = compute_metrics
            metrics = super().evaluate(eval_dataset, ignore_keys,
                metric_key_prefix=f'{eval_dataset_name}_{metric_key_prefix}')
            all_metrics.update(metrics)
        return all_metrics
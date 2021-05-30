from os.path import abspath

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from transformers.models.bert import BertPreTrainedModel, BertModel, BertConfig, BertForSequenceClassification
import pandas as pd
import numpy as np

from loss import DistillLoss, SmoothedDistillLoss, BiasProductBaseline, SmoothedReweightLoss


class BertDistillConfig(BertConfig):
    EXPERT_POLICIES = {'freeze', 'e2e'}
    LOSS_FUNCTIONS = {'smoothed'}

    def __init__(self,
                 teacher_model_name_or_path='bert-base-uncased',
                 weak_model_name_or_path='google/bert_uncased_L-2_H-128_A-2',
                 expert_policy='freeze',
                 loss_fn='smoothed',
                 load_predictions_mode='new',
                 *args, **kwargs
                 ):
        assert expert_policy in BertDistillConfig.EXPERT_POLICIES, \
            f'The expert policy must be one of {", ".join(BertDistillConfig.EXPERT_POLICIES)},' \
            f'but got {expert_policy}'

        super().__init__(*args, **kwargs)
        self.teacher_model_name_or_path = teacher_model_name_or_path
        self.weak_model_name_or_path = weak_model_name_or_path
        self.expert_policy = expert_policy
        self.loss_fn = loss_fn
        self.load_predictions_mode = load_predictions_mode


def _select_loss(config: BertDistillConfig) -> DistillLoss:
    name = config.loss_fn
    if name == 'smoothed':
        return SmoothedDistillLoss()
    if name == 'poe':
        return BiasProductBaseline()
    if name == 'reweight':
        return SmoothedReweightLoss()

    raise ValueError(f'Unknown loss function ${name}')


class BertDistill(BertPreTrainedModel):
    """Pre-trained BERT model that uses our loss functions"""
    config_class = BertDistillConfig
    _keys_to_ignore_on_save = ['teacher_model', 'weak_model']

    def __init__(self, config: BertDistillConfig, teacher_model, weak_model,
                 bert: BertModel, loss_fn: DistillLoss):
        super(BertDistill, self).__init__(config)
        assert isinstance(loss_fn, DistillLoss), f'loss_fn must be an instance of ClfDistillLossFunction, ' \
                                                 f'but got {loss_fn.__class__.__name__}'
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bert = bert
        self.loss_fn = loss_fn
        self.load_predictions_mode = config.load_predictions_mode
        # Must be set after weight initialization
        if isinstance(teacher_model, str):
            if self.load_predictions_mode in ['legacy', 'legacy_teacher']:
                self.teacher_logits = self._load_df_legacy(teacher_model)
            elif self.load_predictions_mode in ['new', 'legacy_weak']:
                self.teacher_logits = self._load_df_new(teacher_model)
        else:
            self.teacher_logits = None
        self.teacher_model = teacher_model

        if isinstance(weak_model, str):
            if self.load_predictions_mode in ['legacy', 'legacy_weak']:
                self.weak_logits = self._load_df_legacy(weak_model)
            elif self.load_predictions_mode in ['new', 'legacy_teacher']:
                self.weak_logits = self._load_df_new(weak_model)
        else:
            self.weak_logits = None
        self.weak_model = weak_model

    @staticmethod
    def _load_df_legacy(path):
        # Dictionary structure:
        #  { <id> : [p1, p2, p3], ... }
        print(f'Loading probabilities from JSON file ({abspath(path)})...')
        df = pd.read_json(path, orient='index')
        c = df.columns
        # Legacy fix (logits from Utama et. al 2020)
        #   cont, ent, neut => ent, neut, cont
        #   0 <= 1
        #   1 <= 2
        #   2 <= 0
        df[[c[0], c[1], c[2]]] = df[[c[1], c[2], c[0]]]
        return df

    @staticmethod
    def _load_df_new(path):
        # CSV Structure:
        #   id,   label_0, label_1, label_2
        #   <id>, p1     , p2     , p3
        #   ...
        print(f'Loading probabilities from CSV file ({abspath(path)})...')
        return pd.read_csv(path).set_index('id')

    def _forward_teacher(self, inputs_with_labels, ids):
        if self.teacher_logits is not None:
            # Return logits from loaded file
            # (N, C)
            inferred_device = inputs_with_labels['input_ids'].device
            if self.load_predictions_mode in ['legacy', 'legacy_teacher']:
                return torch.from_numpy(np.array(self.teacher_logits.loc[ids.cpu() - 1])).to(inferred_device), True

            return torch.from_numpy(np.array(self.teacher_logits.loc[ids.cpu()])).to(inferred_device), False

        with torch.no_grad():
            # (N, C)
            _, teacher_logits = self.teacher_model(**inputs_with_labels, return_dict=False)
        return teacher_logits, False

    def _forward_weak(self, inputs_with_labels, ids):
        if self.weak_logits is not None:
            # Return logits from loaded file
            # (N, C)
            inferred_device = inputs_with_labels['input_ids'].device
            if self.load_predictions_mode in ['legacy', 'legacy_weak']:
                return torch.from_numpy(np.array(self.weak_logits.loc[ids.cpu() - 1])).to(inferred_device), True

            return torch.from_numpy(np.array(self.weak_logits.loc[ids.cpu()])).to(inferred_device), False

        with torch.no_grad():
            _, weak_logits = self.weak_model(**inputs_with_labels, return_dict=False)
        return weak_logits, False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, id=None, **kwargs):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        inputs_with_labels = {**inputs, 'labels': labels}
        _, pooled_output = self.bert(**inputs, return_dict=False)
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None:
            return logits

        if not self.training:
            loss = self.loss_fn(pooled_output, logits, None, None, labels)
            return loss, logits

        teacher_logits, is_teacher_probs = self._forward_teacher(inputs_with_labels, id)
        weak_logits, is_weak_probs = self._forward_weak(inputs_with_labels, id)
        loss = self.loss_fn(pooled_output, logits, weak_logits, teacher_logits, labels, not is_teacher_probs,
                            not is_weak_probs)
        return loss, logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config: BertDistillConfig = kwargs.pop('config', None)
        assert isinstance(config, cls.config_class), f'config must be an instance of {cls.config_class.__name__}' \
                                                     f', but got {config.__class__.__name__}'
        bert_base = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config,
                                                                  *model_args, **kwargs)

        # weak_model = AutoModelForSequenceClassification.from_pretrained(
        #     config.weak_model_name_or_path,
        #     num_labels=config.num_labels
        # )
        # teacher_model = AutoModelForSequenceClassification.from_pretrained(
        #     config.teacher_model_name_or_path,
        #     num_labels=config.num_labels
        # )
        loss_fn = _select_loss(config)
        # model = cls(config=config, bert=bert_base.bert, weak_model=weak_model, teacher_model=teacher_model,
        #             loss_fn=loss_fn)
        model = cls(config=config,
                    bert=bert_base.bert,
                    weak_model=config.weak_model_name_or_path,
                    teacher_model=config.teacher_model_name_or_path,
                    loss_fn=loss_fn)
        # Load classifier weights from the saved model (if available)
        model.classifier = bert_base.classifier
        return model

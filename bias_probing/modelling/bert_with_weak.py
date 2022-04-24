import os
import re
from os.path import abspath
from typing import Iterable, Union
import pandas as pd
import numpy as np

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, WEIGHTS_NAME, is_wandb_available
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertConfig, BertForSequenceClassification, BertPreTrainedModel, BertModel
from transformers.utils.logging import get_logger

from ..loss import ProductOfExpertsLoss, EnsembleLoss, DebiasedFocalLoss
from .bias import LexicalBiasModel, HypothesisOnlyModel

if is_wandb_available():
    import wandb

logger = get_logger()


class BertWithWeakLearnerConfig(BertConfig):
    EXPERT_POLICIES = {'freeze', 'e2e'}
    LOSS_FUNCTIONS = {'poe', 'dfl'}

    def __init__(self,
                 weak_model_name_or_path='google/bert_uncased_L-2_H-128_A-2',
                 expert_policy='freeze',
                 loss_fn='poe',
                 poe_alpha=1.0,
                 dfl_gamma=2.0,
                 lambda_bias=1.0,
                 *args, **kwargs
                 ):
        assert expert_policy in BertWithWeakLearnerConfig.EXPERT_POLICIES, \
            f'The expert policy must be one of {", ".join(BertWithWeakLearnerConfig.EXPERT_POLICIES)},' \
            f'but got {expert_policy}'
        assert loss_fn in BertWithWeakLearnerConfig.LOSS_FUNCTIONS, \
            f'The loss functions must be one of {", ".join(BertWithWeakLearnerConfig.LOSS_FUNCTIONS)},' \
            f'but got {loss_fn}'

        super().__init__(*args, **kwargs)
        self.weak_model_name_or_path = weak_model_name_or_path
        self.expert_policy = expert_policy
        self.loss_fn = loss_fn
        self.poe_alpha = poe_alpha
        self.dfl_gamma = dfl_gamma
        self.lambda_bias = lambda_bias


def _select_loss(config: BertWithWeakLearnerConfig) -> EnsembleLoss:
    name = config.loss_fn
    if name == 'poe':
        return ProductOfExpertsLoss(config.poe_alpha)
    elif name == 'dfl':
        return DebiasedFocalLoss(gamma=config.dfl_gamma)

    raise ValueError(f'Unknown loss function ${name}')


class BertWithWeakLearner(BertPreTrainedModel):
    config_class = BertWithWeakLearnerConfig

    def __init__(self, config: BertWithWeakLearnerConfig):
        super().__init__(config)
        # self.weak_learner = weak_learner
        # self.bert = bert
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bert = BertModel(config)
        self.loss_fn = _select_loss(config)
        self.lambda_bias = self.config.lambda_bias
        self.bias_loss_fn = nn.CrossEntropyLoss()
        if config.expert_policy == 'freeze':
            self.weak_logits = self._load_df(config.weak_model_name_or_path)
        elif config.expert_policy == 'e2e':
            self.weak_logits = None
            self.weak_model = AutoModelForSequenceClassification.from_pretrained(
                config.weak_model_name_or_path,
                num_labels=config.num_labels
            )

    @staticmethod
    def _load_df(path):
        print(f'Loading probabilities from CSV file ({abspath(path)})...')
        return pd.read_csv(path).set_index('id')

    def _forward_weak(self, inputs_with_labels, ids):
        if self.weak_logits is not None:
            # Return logits from loaded file
            # (N, C)
            inferred_device = inputs_with_labels['input_ids'].device

            return torch.from_numpy(np.array(self.weak_logits.loc[ids.cpu()])).to(inferred_device)

        with torch.no_grad():
            _, weak_logits = self.weak_model(**inputs_with_labels, return_dict=False)
        return weak_logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, id=None, **kwargs):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }
        _, pooled_output = self.bert(**inputs, return_dict=False)
        main_logits = self.classifier(self.dropout(pooled_output))

        weak_logits = self._forward_weak(inputs, id)
        # TODO Remove
        # expert_policy = self.config.expert_policy
        # if expert_policy == 'freeze':
        #     with torch.no_grad():
        #         weak_outputs: SequenceClassifierOutput = self.weak_learner(**inputs, return_dict=True)
        #         weak_logits = weak_outputs.logits
        # elif expert_policy == 'e2e':
        #     _, logits = self.weak_learner(**inputs, return_dict=False)
        #     weak_logits = logits
        # else:
        #     raise ValueError(f'Incorrect expert_policy detected ({expert_policy})')
        # assert weak_logits.size() == main_logits.size(), f'{weak_logits.size()} != {main_logits.size()}'
        # assert labels.size()[0] == weak_logits.size()[0]

        loss, logits = self.loss_fn(main_logits, weak_logits, labels)
        expert_policy = self.config.expert_policy
        if expert_policy == 'e2e' and self.training:
            loss += self.lambda_bias * self.bias_loss_fn(weak_logits, labels)
        return loss, logits

    # TODO Remove
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     config: BertWithWeakLearnerConfig = kwargs.pop('config', None)
    #     assert isinstance(config, cls.config_class), f'config must be an instance of {cls.config_class.__name__}' \
    #                                                  f', but got {config.__class__.__name__}'
    #     bert_base = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config,
    #                                                               *model_args, **kwargs)
    #     loss_fn = _select_loss(config)
    #     model = cls(config=config, bert=bert_base.bert, loss_fn=loss_fn)
    #     model.classifier = bert_base.classifier
    #     return model


class BertWithWeakLearnerLegacy(BertPreTrainedModel):
    config_class = BertWithWeakLearnerConfig

    def __init__(self, config: BertWithWeakLearnerConfig):
        super().__init__(config)
        # self.weak_learner = weak_learner
        # self.bert = bert
        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertForSequenceClassification(config)
        weak_config = AutoConfig.from_pretrained(config.weak_model_name_or_path)
        weak_config.num_labels = config.num_labels
        self.weak_learner = BertForSequenceClassification(weak_config)
        self.loss_fn = _select_loss(config)
        self.lambda_bias = self.config.lambda_bias


class BertWithExplicitBiasConfig(BertWithWeakLearnerConfig):
    BIAS_TYPES = {'hypo', 'hans'}

    def __init__(self, bias_type='hans', similarity_types=None, hans_features=None, lambda_bias=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert bias_type in BertWithExplicitBiasConfig.BIAS_TYPES, \
            f'The bias type must be one of {", ".join(BertWithExplicitBiasConfig.BIAS_TYPES)},' \
            f'but got {bias_type}'
        self.bias_type = bias_type
        if similarity_types is None:
            # Similarity Types: max, mean, min (top 2)
            similarity_types = ['max', 'mean', 'min', 'second_min']
        self.similarity_types = similarity_types
        if hans_features is None:
            # HANS Features (4):
            # - Whether all words in the hypothesis are included in the premise
            # - If the hypothesis is the contiguous subsequence of the premise
            # - If the hypothesis is a subtree in the premise’s parse tree
            # - The number of tokens shared between premise and hypothesis
            #   normalized by the number of tokens in the premise
            #
            hans_features = ['constituent', 'subsequence', 'lexical_overlap', 'overlap_rate']
        self.hans_features = hans_features
        self.lambda_bias = lambda_bias


class BertWithExplicitBias(BertPreTrainedModel):
    config_class = BertWithExplicitBiasConfig

    def __init__(self, config: BertWithExplicitBiasConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.config = config
        self.loss_fn = _select_loss(config)
        self.bias_type = self.config.bias_type
        self.similarity_types = self.config.similarity_types
        self.hans_features = self.config.hans_features
        self.lambda_bias = self.config.lambda_bias

        print(f'Bias type: {self.bias_type}')
        if config.expert_policy == 'freeze':
            self.bias_logits = self._load_df(config.weak_model_name_or_path)
        elif config.expert_policy == 'e2e':
            self.bias_logits = None
            self.bias_model, self.bias_loss_fn = self._init_bias_model(config)

    def _init_bias_model(self, config: BertWithExplicitBiasConfig):
        if self.bias_type == 'hans':
            num_hans_features = len(self.hans_features)
            bias_model = LexicalBiasModel(num_hans_features, self.similarity_types, self.config.num_labels)
            # Try to guess device here. This could probably be better implemented from the outside but oh well.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # entailment, neutral, contradiction
            bias_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.5, 0.5]).to(device))
        else:
            bias_model = HypothesisOnlyModel(self._encode_hypothesis_only, config.hidden_size, config.num_labels)
            bias_loss_fn = nn.CrossEntropyLoss()
        return bias_model, bias_loss_fn

    @staticmethod
    def _load_df(path):
        print(f'Loading probabilities from CSV file ({abspath(path)})...')
        return pd.read_csv(path).set_index('id')

    def _forward_bias(self, input_ids, hypothesis_ids=None, hypothesis_attention_mask=None, id=None, **kwargs):
        if self.bias_logits is not None:
            # Return logits from loaded file
            # (N, C)
            inferred_device = input_ids.device
            return torch.from_numpy(np.array(self.bias_logits.loc[id.cpu()])).to(inferred_device)

        assert self.bias_model
        if self.bias_type == 'hypo':
            # Hypothesis-Only bias
            bias_logits = self._forward_hypothesis_only(hypothesis_ids, hypothesis_attention_mask)
        elif self.bias_type == 'hans':
            # Lexical Overlap Bias
            hans_features = [kwargs.pop(hans_feature, None) for hans_feature in self.hans_features]
            premise_attention_mask = kwargs.pop('premise_attention_mask', None)
            premise_ids = kwargs.pop('premise_ids', None)

            bias_logits = self._forward_hans_features(hans_features=hans_features,
                                                      premise_input_ids=premise_ids,
                                                      hypothesis_input_ids=hypothesis_ids,
                                                      premise_attention_mask=premise_attention_mask,
                                                      hypothesis_attention_mask=hypothesis_attention_mask)
        else:
            raise ValueError(f'Bad bias type detected: {self.bias_type}')
        return bias_logits

    def _encode_hypothesis_only(self, hypothesis_input_ids, hypothesis_attention_mask):
        with torch.no_grad():
            _, bias_pooled_output = self.bert(hypothesis_input_ids, token_type_ids=None,
                                              attention_mask=hypothesis_attention_mask, return_dict=False)
        return self.dropout(bias_pooled_output)

    def _forward_hypothesis_only(self, hypothesis_input_ids, hypothesis_attention_mask=None):
        return self.bias_model(hypothesis_input_ids, hypothesis_attention_mask)

    def _forward_hans_features(self, hans_features: Iterable[torch.Tensor],
                               premise_input_ids, hypothesis_input_ids,
                               premise_attention_mask=None, hypothesis_attention_mask=None):
        if premise_input_ids is not None and hypothesis_input_ids is not None and len(self.similarity_types) > 0:
            with torch.no_grad():
                h_outputs = self.bert(hypothesis_input_ids, token_type_ids=None,
                                      attention_mask=hypothesis_attention_mask, return_dict=False)
                h_matrix = self.dropout(h_outputs[0])
                # h_matrix = h_outputs[0]

                p_outputs = self.bert(premise_input_ids, token_type_ids=None, attention_mask=premise_attention_mask,
                                      return_dict=False)
                p_matrix = self.dropout(p_outputs[0])
            return self.bias_model(hans_features, p_matrix, h_matrix, premise_attention_mask, hypothesis_attention_mask)

        return self.bias_model(hans_features)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                hypothesis_ids=None, hypothesis_attention_mask=None, id=None, **kwargs):

        # Main model
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     return_dict=False)
        logits = self.classifier(self.dropout(pooled_output))
        if not self.training:
            loss, logits = self.loss_fn(logits, None, labels)
            return loss, logits

        bias_logits = self._forward_bias(input_ids, hypothesis_ids, hypothesis_attention_mask, id, **kwargs)
        loss, logits = self.loss_fn(logits, bias_logits, labels)

        # Combination loss
        if self.config.expert_policy == 'e2e' and self.training:
            bias_loss = self.bias_loss_fn(bias_logits, labels)
            loss += self.lambda_bias * bias_loss
        return loss, logits

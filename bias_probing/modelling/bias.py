from typing import Iterable, List, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.models.bert import BertPreTrainedModel, BertConfig, BertModel


def _get_word_similarity(hyp_matrix: Tensor, prem_matrix: Tensor,
                         h_mask: Tensor, p_mask: Tensor,
                         similarity_types: List[str]):
    # normalize the token embeddings.
    # (bsz, seq_len, hidden_dim)
    hyp_matrix = F.normalize(hyp_matrix, p=2, dim=2)
    prem_matrix = F.normalize(prem_matrix, p=2, dim=2)

    # (bsz, seq_len, hidden_dim)
    hyp_matrix = hyp_matrix * h_mask.view(hyp_matrix.shape[0], hyp_matrix.shape[1], 1).float()
    # (bsz, seq_len, hidden_dim)
    prem_matrix = prem_matrix * p_mask.view(prem_matrix.shape[0], prem_matrix.shape[1], 1).float()

    # (bsz, h_seq_len, hidden_dim) x (bsz, hidden_dim, p_seq_len) => (bsz, h_seq_len, p_seq_len)
    similarity_matrix = hyp_matrix.bmm(prem_matrix.transpose(2, 1))
    # (bsz, h_seq_len, p_seq_len) => (bsz, h_seq_len)
    similarity, _ = torch.max(similarity_matrix, 2)

    sim_score = []
    if 'min' in similarity_types or 'second_min' in similarity_types:
        # compute the min and second min in the similarities.
        similarity_replace = similarity.clone()
        # all the similarity values are smaller than 1 so 10 is a good number
        # so that the masked elements are not selected during the top minimum computations.
        similarity_replace[h_mask == 0] = 10
        y, i = torch.topk(similarity_replace, k=2, dim=1, largest=False, sorted=True)
        if 'min' in similarity_types:
            sim_score.append(y[:, 0].view(-1, 1))
        if 'second_min' in similarity_types:
            sim_score.append(y[:, 1].view(-1, 1))
    if 'mean' in similarity_types:
        h_lens = torch.sum(h_mask, dim=1)
        # note that to account for zero values, we have to consider the length not
        # getting mean.
        sum_similarity = torch.sum(similarity, dim=1)
        mean_similarity = sum_similarity / h_lens.float()
        sim_score.append(mean_similarity.view(-1, 1))
    if 'max' in similarity_types:
        max_similarity = torch.max(similarity, 1)[0]
        sim_score.append(max_similarity.view(-1, 1))

    similarity_score = torch.cat(sim_score, dim=1)
    return similarity_score


class LexicalBiasModel(nn.Module):
    def __init__(self, num_hans_features: int, similarity_types: List[str], num_labels: int = 3):
        super().__init__()
        self.num_labels = num_labels
        self.num_hans_features = num_hans_features
        self.similarity_types = similarity_types
        self.num_features = self.num_hans_features + len(self.similarity_types)
        self.classifier = self._hans_features_classifier(self.num_features)
        self.perform_check = True
        print(f'LexicalBiasModel::num_features = {self.num_features}')

    def _hans_features_classifier(self, num_features, non_linear=True):
        num_labels_bias_only = self.num_labels
        if not non_linear:
            return nn.Linear(num_features, num_labels_bias_only)
        else:
            return nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.Tanh(),
                nn.Linear(num_features, num_features),
                nn.Tanh(),
                nn.Linear(num_features, num_labels_bias_only))

    def forward(self, hans_features: Iterable[Tensor],
                premise_word_embeddings: Tensor = None, hypothesis_word_embeddings: Tensor = None,
                premise_attention_mask=None, hypothesis_attention_mask=None):
        # (N,) => 4 x (N, 1) => (N, 4)
        base_features = [feature.unsqueeze(-1) for feature in hans_features if feature is not None]
        if self.perform_check:
            self.perform_check = False
            assert len(base_features) == self.num_hans_features, f'Number of input HANS features ' \
                                                                 f'({len(base_features)}) ' \
                                                                 f'does not match expected ({self.num_hans_features})'

        if premise_word_embeddings is not None and hypothesis_word_embeddings is not None and len(
                self.similarity_types) > 0:
            # compute similarity features.
            similarity_score = _get_word_similarity(hypothesis_word_embeddings,
                                                    premise_word_embeddings,
                                                    hypothesis_attention_mask,
                                                    premise_attention_mask,
                                                    self.similarity_types)

            hans_features = torch.cat([similarity_score, *base_features], dim=1)
        else:
            hans_features = torch.cat(base_features, dim=1)

        bias_logits = self.classifier(hans_features)
        return bias_logits


class HypothesisOnlyModel(nn.Module):
    def __init__(self, encoder: Callable[[Tensor, Tensor], Tensor], input_size: int, num_labels: int = 3):
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.encoder = encoder
        self.classifier = self._hypothesis_only_classifier()

    def _hypothesis_only_classifier(self, non_linear=True):
        # (Hidden Size => Classes)
        input_size = self.input_size
        num_labels = self.num_labels
        if non_linear:
            return nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Tanh(),
                nn.Linear(input_size, int(input_size / 2)),
                nn.Tanh(),
                nn.Linear(int(input_size / 2), int(input_size / 4)),
                nn.Tanh(),
                nn.Linear(int(input_size / 4), num_labels),
            )
        return nn.Linear(input_size, num_labels)

    def forward(self, hypothesis_input_ids: Tensor, hypothesis_attention_mask: Tensor = None):
        bias_pooled_output = self.encoder(hypothesis_input_ids, hypothesis_attention_mask)
        # pooled_input (bsz, hidden_size)
        return self.classifier(bias_pooled_output)


class BertWithLexicalBiasConfig(BertConfig):

    def __init__(self, similarity_types=None, hans_features=None, **kwargs):
        super().__init__(**kwargs)
        if similarity_types is None:
            similarity_types = ['mean', 'max', 'min']
        self.similarity_types = similarity_types
        if hans_features is None:
            # HANS Features (4):
            # - Whether all words in the hypothesis are included in the premise
            # - If the hypothesis is the contiguous subsequence of the premise
            # - If the hypothesis is a subtree in the premise’s parse tree
            # - The number of tokens shared between premise and hypothesis
            #   normalized by the number of tokens in the premise
            #
            # Similarity Types: mean, min, max
            hans_features = ['constituent', 'subsequence', 'lexical_overlap', 'overlap_rate']
        self.hans_features = hans_features


class BertWithLexicalBiasModel(BertPreTrainedModel):
    config_class = BertWithLexicalBiasConfig

    def __init__(self, config: BertWithLexicalBiasConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.similarity_types = config.similarity_types
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hans_features = config.hans_features
        self.classifier = LexicalBiasModel(len(config.hans_features), config.similarity_types, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def _forward_hans_features(self, hans_features: Iterable[torch.Tensor],
                               premise_input_ids, hypothesis_input_ids,
                               premise_attention_mask=None, hypothesis_attention_mask=None):
        if premise_input_ids is not None and hypothesis_input_ids is not None and len(self.similarity_types) > 0:
            with torch.no_grad():
                h_matrix = self.dropout(
                    self.bert(hypothesis_input_ids, token_type_ids=None,
                              attention_mask=hypothesis_attention_mask, return_dict=False)[0]
                )
                p_matrix = self.dropout(
                    self.bert(premise_input_ids, token_type_ids=None, attention_mask=premise_attention_mask,
                              return_dict=False)[0]
                )
            return self.classifier(hans_features, p_matrix, h_matrix, premise_attention_mask, hypothesis_attention_mask)
        return self.classifier(hans_features)

    def forward(self, hypothesis_ids=None, hypothesis_attention_mask=None,
                premise_ids=None, premise_attention_mask=None, labels=None, **kwargs):
        hans_features = [kwargs.pop(hans_feature, None) for hans_feature in self.hans_features]
        bias_logits = self._forward_hans_features(hans_features=hans_features,
                                                  premise_input_ids=premise_ids,
                                                  hypothesis_input_ids=hypothesis_ids,
                                                  premise_attention_mask=premise_attention_mask,
                                                  hypothesis_attention_mask=hypothesis_attention_mask)
        if labels is None:
            return bias_logits

        loss = self.loss_fn(bias_logits, labels)
        return loss, bias_logits

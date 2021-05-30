# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Symmetric evaluation set based on the FEVER (fact verification) dataset"""

from __future__ import absolute_import, division, print_function

import json
import os
from os.path import join

import datasets

_CITATION = """\
TODO Add
"""

_DESCRIPTION = """\

"""


class MultiNliHardConfig(datasets.BuilderConfig):
    """BuilderConfig for FEVER-Symmetric."""

    def __init__(self, **kwargs):
        """BuilderConfig for FEVER-Symmetric.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(MultiNliHardConfig, self).__init__(version=datasets.Version("0.9.0", ""), **kwargs)


class MultiNliHard(datasets.GeneratorBasedBuilder):
    """Symmetric evaluation set based on the FEVER (fact verification) dataset. Version 0.2."""

    BUILDER_CONFIGS = [
        MultiNliHardConfig(
            name="plain_text",
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://github.com/rabeehk/robust-nli",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["premise"], ex["hypothesis"]])

    def _split_generators(self, dl_manager):
        downloaded_path_matched, downloaded_path_mismatched = dl_manager.download_and_extract([
            "https://www.dropbox.com/s/3aktzl4bhmqti9n/MNLIMatchedHardWithHardTest.zip?dl=1",
            "https://www.dropbox.com/s/bidxvrd8s2msyan/MNLIMismatchedHardWithHardTest.zip?dl=1"
        ])
        downloaded_path_matched = join(downloaded_path_matched, 'MNLIMatchedHardWithHardTest')
        downloaded_path_mismatched = join(downloaded_path_mismatched, 'MNLIMismatchedHardWithHardTest')
        return [
            datasets.SplitGenerator(name="test_matched", gen_kwargs={"filepath": downloaded_path_matched}),
            datasets.SplitGenerator(name="test_mismatched", gen_kwargs={"filepath": downloaded_path_mismatched}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "evidence", "claim" and "label" strings
        """
        premise_file = join(filepath, 's1.test')
        hypothesis_file = join(filepath, 's2.test')
        labels_file = join(filepath, 'labels.test')
        with open(premise_file, 'r') as pf, open(hypothesis_file, 'r') as hf, open(labels_file, 'r') as lf:
            for idx, (premise, hypothesis, label) in enumerate(zip(pf, hf, lf)):
                premise = premise.strip()
                hypothesis = hypothesis.strip()
                label = label.strip()

                # Works for both splits even though dev has some extra human labels.
                yield idx, {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label
                }

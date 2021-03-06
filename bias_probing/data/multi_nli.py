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
"""The Multi-Genre NLI Corpus."""

from __future__ import absolute_import, division, print_function

import os

import datasets


_CITATION = """\
@InProceedings{N18-1101,
  author = {Williams, Adina
            and Nangia, Nikita
            and Bowman, Samuel},
  title = {A Broad-Coverage Challenge Corpus for
           Sentence Understanding through Inference},
  booktitle = {Proceedings of the 2018 Conference of
               the North American Chapter of the
               Association for Computational Linguistics:
               Human Language Technologies, Volume 1 (Long
               Papers)},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  pages = {1112--1122},
  location = {New Orleans, Louisiana},
  url = {http://aclweb.org/anthology/N18-1101}
}
"""

_DESCRIPTION = """\
The Multi-Genre Natural Language Inference (MultiNLI) corpus is a
crowd-sourced collection of 433k sentence pairs annotated with textual
entailment information. The corpus is modeled on the SNLI corpus, but differs in
that covers a range of genres of spoken and written text, and supports a
distinctive cross-genre generalization evaluation. The corpus served as the
basis for the shared task of the RepEval 2017 Workshop at EMNLP in Copenhagen.
"""


class MultiNLIConfig(datasets.BuilderConfig):
    """BuilderConfig for MultiNLI."""

    def __init__(self, **kwargs):
        """BuilderConfig for MultiNLI.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(MultiNLIConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class MultiNli(datasets.GeneratorBasedBuilder):
    """MultiNLI: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        MultiNLIConfig(
            name="plain_text",
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "premise_parse": datasets.Value("string"),
                    "hypothesis_parse": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://www.nyu.edu/projects/bowman/multinli/",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["premise"], ex["hypothesis"]])

    def _split_generators(self, dl_manager):

        downloaded_dir = dl_manager.download_and_extract(
            "http://storage.googleapis.com/tfds-data/downloads/multi_nli/multinli_1.0.zip"
        )
        mnli_path = os.path.join(downloaded_dir, "multinli_1.0")
        train_path = os.path.join(mnli_path, "multinli_1.0_train.txt")
        matched_validation_path = os.path.join(mnli_path, "multinli_1.0_dev_matched.txt")
        mismatched_validation_path = os.path.join(mnli_path, "multinli_1.0_dev_mismatched.txt")

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name="validation_matched", gen_kwargs={"filepath": matched_validation_path}),
            datasets.SplitGenerator(name="validation_mismatched", gen_kwargs={"filepath": mismatched_validation_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate mnli examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "premise", "hypothesis" and "label" strings
        """
        for idx, line in enumerate(open(filepath, "rb")):
            if idx == 0:
                continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Examples not marked with a three out of five consensus are marked with
            # "-" and should not be used in standard evaluations.
            if split_line[0] == "-":
                continue
            # Works for both splits even though dev has some extra human labels.
            yield idx, {
                "id": idx,
                "premise": split_line[5],
                "hypothesis": split_line[6],
                "premise_parse": split_line[1],  # sentence1_binary_parse
                "hypothesis_parse": split_line[2],  # sentence2_binary_parse
                "label": split_line[0]
            }

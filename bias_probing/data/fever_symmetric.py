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

import datasets

_CITATION = """\
@InProceedings{schuster2019towards,
  author = 	"Schuster, Tal and
            Shah, Darsh J and
            Yeo, Yun Jie Serene and
            Filizzola, Daniel and
            Santus, Enrico and
            Barzilay, Regina", 			
  title = 	"Towards Debiasing Fact Verification Models",
  booktitle = 	"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  year = 	"2019",
  publisher = 	"Association for Computational Linguistics",
  url = 	"https://arxiv.org/abs/1908.05267"
}
"""

# TODO Add
_DESCRIPTION = """\

"""


class FeverSymmetricConfig(datasets.BuilderConfig):
    """BuilderConfig for FEVER-Symmetric."""

    def __init__(self, **kwargs):
        """BuilderConfig for FEVER-Symmetric.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(FeverSymmetricConfig, self).__init__(version=datasets.Version("0.1.0", ""), **kwargs)


class FeverSymmetric(datasets.GeneratorBasedBuilder):
    """Symmetric evaluation set based on the FEVER (fact verification) dataset. Version 0.2."""

    BUILDER_CONFIGS = [
        FeverSymmetricConfig(
            name="plain_text",
            description="Plain text",
            data_dir='src/data'
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "evidence": datasets.Value("string"),
                    "claim": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["SUPPORTS", "REFUTES"]),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://github.com/TalSchuster/FeverSymmetric",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["claim"], ex["evidence"]])

    def _split_generators(self, dl_manager):
        print(os.getcwd())
        test_path = dl_manager.download(
            "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl",
        )
        return [
            datasets.SplitGenerator(name="test", gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """Generate examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "evidence", "claim" and "label" strings
        """
        with open(filepath, 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip().decode('utf-8')
                # print(line)
                json_obj = json.loads(line)

                # Works for both splits even though dev has some extra human labels.
                yield idx, {
                    "id": json_obj["id"],
                    "claim": json_obj["claim"],
                    "evidence": json_obj["evidence_sentence"] if "evidence_sentence" in json_obj
                    else json_obj["evidence"],
                    "label": json_obj["label"]
                }

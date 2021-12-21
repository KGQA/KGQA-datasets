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
"""CWQ (Compositional Wikidata Questions) dataset."""


import json
import os
import re

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
    @ARTICLE{2021arXiv210803509C,
        author = {{Cui}, Ruixiang and {Aralikatte}, Rahul and {Lent}, Heather and {Hershcovich}, Daniel},
        title = "{Multilingual Compositional Wikidata Questions}",
        journal = {arXiv e-prints},
        keywords = {Computer Science - Computation and Language},
        year = 2021,
        month = aug,
        eid = {arXiv:2108.03509},
        pages = {arXiv:2108.03509},
        archivePrefix = {arXiv},
        eprint = {2108.03509},
        primaryClass = {cs.CL},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210803509C},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
"""

_DESCRIPTION = """
The CWQ dataset (and it's splits) for measuring compositional generalization.
See https://arxiv.org/abs/2108.03509.pdf for background.
Example usage:
data = datasets.load_dataset('cwq/mcd1')
"""

_DATA_URLS = {
    "dataset": "https://raw.githubusercontent.com/coastalcph/seq2sparql/master/cwq/dataset.json",
    "mcd1": "https://raw.githubusercontent.com/coastalcph/seq2sparql/master/cwq/split/mcd1.json",
    "mcd2": "https://raw.githubusercontent.com/coastalcph/seq2sparql/master/cwq/split/mcd2.json",
    "mcd3": "https://raw.githubusercontent.com/coastalcph/seq2sparql/master/cwq/split/mcd3.json",
    "random_split": "https://raw.githubusercontent.com/coastalcph/seq2sparql/master/cwq/split/random_split.json"
}


class CwqConfig(datasets.BuilderConfig):
    """BuilderConfig for CWQ splits."""

    def __init__(self, name, **kwargs):
        """BuilderConfig for CWQ.
        Args:
          name: Unique name of the split.
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        super(CwqConfig, self).__init__(
            name=name, description=_DESCRIPTION, **kwargs
        )
        self.split_file = os.path.join(name + ".json")


_QUESTION = "question"
_QUERY = "query"
_QUESTION_FIELD = "questionPatternModEntities"
_QUERY_FIELD = "sparqlPatternModEntities"


class CWQ(datasets.GeneratorBasedBuilder):
    """CWQ task / splits."""

    BUILDER_CONFIGS = [
        CwqConfig(name="mcd1"),
        CwqConfig(name="mcd2"),
        CwqConfig(name="mcd3"),
        CwqConfig(name="random_split"),
    ]

    DEFAULT_CONFIG_NAME = "random_split"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "CFQquestionIdx": datasets.Value("int64"),
                    "questionPatternModEntities": datasets.Value("string"),
                    "questionTemplate": datasets.Value("string"),
                    "questionWithBrackets": datasets.Value("string"),
                    "recursionDepth": datasets.Value("int32"),
                    "sparql": datasets.Value("string"),
                    "sparqlPattern": datasets.Value("string"),
                    "sparqlPatternModEntities": datasets.Value("string"),
                    "questionWithBrackets_kn": datasets.Value("string"),
                    "questionPatternModEntities_kn": datasets.Value("string"),
                    "questionWithBrackets_he": datasets.Value("string"),
                    "questionPatternModEntities_he": datasets.Value("string"),
                    "questionWithBrackets_zh": datasets.Value("string"),
                    "questionPatternModEntities_zh": datasets.Value("string"),
                    "expectedResponse": datasets.Value("bool")
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/coastalcph/seq2sparql",
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        cwq_files = dl_manager.download_and_extract(_DATA_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_files": cwq_files,
                    "split": "train",
                    "split_id": "trainIdxs"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_files": cwq_files,
                    "split": "dev",
                    "split_id": "devIdxs"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_files": cwq_files,
                    "split": "test",
                    "split_id": "testIdxs"
                },
            ),
        ]

    def _generate_examples(self, data_files, split, split_id, **kwargs):
        """Yields examples."""
        with open(os.path.join(data_files["dataset"]), encoding="utf-8") as f1:
            if self.config.name == "random_split":
                splits = json.load(open(data_files["random_split"], encoding="utf-8"))
            elif self.config.name == "mcd1":
                splits = json.load(open(data_files["mcd1"], encoding="utf-8"))
            elif self.config.name == "mcd2":
                splits = json.load(open(data_files["mcd2"], encoding="utf-8"))
            elif self.config.name == "mcd3":
                splits = json.load(open(data_files["mcd3"], encoding="utf-8"))

            cwq = json.load(f1)
            for idx in splits[split_id]:
                yield idx, cwq[idx]
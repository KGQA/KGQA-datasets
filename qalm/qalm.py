"""QALM."""

import json
import os
import ast

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{DBLP:conf/kcap/KaffeeESV19,
    title     = {Ranking Knowledge Graphs By Capturing Knowledge about Languages and
               Labels},
    author    = {Lucie{-}Aim{\'{e}}e Kaffee and
               Kemele M. Endris and
               Elena Simperl and
               Maria{-}Esther Vidal},
    booktitle = {Proceedings of the 10th International Conference on Knowledge Capture,
               {K-CAP} 2019, Marina Del Rey, CA, USA, November 19-21, 2019},
    publisher = {{ACM}},
    year      = {2019},
    url       = {https://doi.org/10.1145/3360901.3364443}
    """

_DESCRIPTION = """\
    QALM Dataset Description
"""

_URL = "https://github.com/luciekaffee/QALM"

_QALM_URLS = {
    "all": "https://raw.githubusercontent.com/luciekaffee/QALM/master/QALM.json"
}

class QALMConfig(datasets.BuilderConfig):
    """BuilderConfig for QALM"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for QALM.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QALMConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class QALMQuestions(datasets.GeneratorBasedBuilder):
    """QALD."""
    BUILDER_CONFIGS = [
        QALMConfig(
            name="qalm",
            description="QALM",
            data_url="",
            data_dir="QALM"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "questions": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                    "classes": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        qalm_files = dl_manager.download(
            {
                "all": _QALM_URLS["all"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", qalm_files["all"]),
                    "split": "train"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            qalm = json.load(f)
            question = {}
            for idx, item in qalm.items():
                question["id"] = str(idx)
                question["domain"] = item["domain"]
                question["questions"] = json.dumps(item["questions"])
                question["answers"] = json.dumps(item["answers"])
                question["classes"] = json.dumps(item["classes"])
                yield idx, question

"""QALD-10."""

import json
import os
import ast

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    
    """

_DESCRIPTION = """\
    QALD-10 Dataset Description
"""

_URL = "https://github.com/KGQA/QALD_10"

_QALD10_URLS = {
    "all": "https://raw.githubusercontent.com/KGQA/QALD_10/main/data/qald_10/qald_10.json"
}

class QALDConfig(datasets.BuilderConfig):
    """BuilderConfig for QALD"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for QALD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QALDConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class QALDQuestions(datasets.GeneratorBasedBuilder):
    """QALD."""
    BUILDER_CONFIGS = [
        QALDConfig(
            name="qald10",
            description="QALD",
            data_url="",
            data_dir="QALD"
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
                    "aggregation": datasets.Value("bool"),
                    "question": datasets.Value("string"),
                    "query": datasets.Features(
                        {
                            "sparql": datasets.Value("string")
                        }
                    ),
                    "answers": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        qald_files = dl_manager.download(
            {
                "all": _QALD10_URLS["all"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", qald_files["all"]),
                    "split": "train"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            qald = json.load(f)
            for idx, question in enumerate(qald["questions"]):
                question["question"] = json.dumps(question["question"])
                question["answers"] = json.dumps(question["answers"])

                yield idx, question

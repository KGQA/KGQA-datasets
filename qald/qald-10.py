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
    "train": "https://raw.githubusercontent.com/KGQA/QALD_10/main/data/qald_9_plus/qald_9_plus_train_wikidata.json",
    "test": "https://raw.githubusercontent.com/KGQA/QALD_10/main/data/qald_10/qald_10.json"
}

class QALDConfig(datasets.BuilderConfig):
    """BuilderConfig for QALD-10"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for QALD-10.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QALDConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class QALDQuestions(datasets.GeneratorBasedBuilder):
    """QALD-10."""
    BUILDER_CONFIGS = [
        QALDConfig(
            name="qald10",
            description="QALD-10",
            data_url="",
            data_dir="QALD-10"
        ),
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
                "train": _QALD10_URLS["train"],
                "test": _QALD10_URLS["test"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", qald_files["train"]),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", qald_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            qald = json.load(f)
            for idx, question in enumerate(qald["questions"]):
                question["question"] = json.dumps(question["question"])
                question["answers"] = json.dumps(question["answers"])

                if kwargs["split"]== "test":
                    del question["aggregation"]

                yield idx, question

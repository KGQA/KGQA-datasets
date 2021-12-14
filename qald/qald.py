"""QALD-9."""

import json
import os
import ast

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{Usbeck20189thCO,
        title={9th Challenge on Question Answering over Linked Data (QALD-9) (invited paper)},
        author={Ricardo Usbeck and Ria Hari Gusmita and Axel-Cyrille Ngonga Ngomo and Muhammad Saleem},
        booktitle={Semdeep/NLIWoD@ISWC},
        year={2018}
    }
    """

_DESCRIPTION = """\
    QALD-9 Dataset Description
"""

_URL = "https://github.com/ag-sc/QALD"

_QALD9_URLS = {
    "train": "https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json",
    "test": "https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-test-multilingual.json"
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

class RuBQQuestions(datasets.GeneratorBasedBuilder):
    """QALD."""
    BUILDER_CONFIGS = [
        QALDConfig(
            name="qald",
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
                    "answertype": datasets.Value("string"),
                    "aggregation": datasets.Value("bool"),
                    "onlydbo": datasets.Value("bool"),
                    "hybrid": datasets.Value("bool"),
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
                "train": _QALD9_URLS["train"],
                "test": _QALD9_URLS["test"]
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

                yield idx, question

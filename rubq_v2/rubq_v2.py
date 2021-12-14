"""RuBQ 2.0: An Innovated Russian Question Answering Dataset."""

import json
import os
import ast

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{
        rybin2021rubq,
        title={Ru{\{}BQ{\}} 2.0: An Innovated Russian Question Answering Dataset},
        author={Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
        booktitle={Eighteenth Extended Semantic Web Conference - Resources Track},
        year={2021},
        url={https://openreview.net/forum?id=P5UQFFoQ4PJ}
    }
    """

_DESCRIPTION = """\
    RuBQ 2.0: An Innovated Russian Question Answering Dataset
"""

_URL = "https://github.com/vladislavneon/RuBQ"

_RUBQ2_URLS = {
    "dev": "https://raw.githubusercontent.com/vladislavneon/RuBQ/master/RuBQ_2.0/RuBQ_2.0_dev.json",
    "test": "https://raw.githubusercontent.com/vladislavneon/RuBQ/master/RuBQ_2.0/RuBQ_2.0_test.json"
}

class RuBQ2Config(datasets.BuilderConfig):
    """BuilderConfig for RuBQ 2.0"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for RuBQ 2.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RuBQ2Config, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class RuBQ2Questions(datasets.GeneratorBasedBuilder):
    """RuBQ 2.0: An Innovated Russian Question Answering Dataset."""
    BUILDER_CONFIGS = [
        RuBQ2Config(
            name="rubq2",
            description="RuBQ2",
            data_url="",
            data_dir="RuBQ2"
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
                    "uid": datasets.Value("int32"),
                    "question_text": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "answer_text": datasets.Value("string"),
                    "question_uris": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "question_props": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "answers": datasets.Value("string"),
                    "paragraphs_uids": datasets.Value("string"),
                    "tags": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "RuBQ_version": datasets.Value("string"),
                    "question_eng": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        rubq2_files = dl_manager.download(
            {
                "dev": _RUBQ2_URLS["dev"],
                "test": _RUBQ2_URLS["test"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", rubq2_files["dev"]),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", rubq2_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            rubq2 = json.load(f)
            for idx, question in enumerate(rubq2):
                question["answers"] = json.dumps(question["answers"])
                question["paragraphs_uids"] = json.dumps(question["paragraphs_uids"])
                yield idx, question

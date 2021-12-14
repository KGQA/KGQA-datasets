"""RuBQ: A Russian Dataset for Question Answering over Wikidata."""

import json
import os
import ast

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @ARTICLE{2020arXiv200510659K,
        author = {{Korablinov}, Vladislav and {Braslavski}, Pavel},
        title = "{RuBQ: A Russian Dataset for Question Answering over Wikidata}",
        journal = {arXiv e-prints},
        keywords = {Computer Science - Computation and Language},
        year = 2020,
        month = may,
        eid = {arXiv:2005.10659},
        pages = {arXiv:2005.10659},
        archivePrefix = {arXiv},
        eprint = {2005.10659},
        primaryClass = {cs.CL},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200510659K},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """

_DESCRIPTION = """\
    RuBQ: A Russian Dataset for Question Answering over Wikidata.
"""

_URL = "https://github.com/vladislavneon/RuBQ"

_RUBQ_URLS = {
    "dev": "https://raw.githubusercontent.com/vladislavneon/RuBQ/master/RuBQ_1.0/RuBQ_1.0_dev.json",
    "test": "https://raw.githubusercontent.com/vladislavneon/RuBQ/master/RuBQ_1.0/RuBQ_1.0_test.json"
}

class RuBQConfig(datasets.BuilderConfig):
    """BuilderConfig for RuBQ"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for RuBQ.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RuBQConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class RuBQ(datasets.GeneratorBasedBuilder):
    """RuBQ: A Russian Dataset for Question Answering over Wikidata."""
    BUILDER_CONFIGS = [
        RuBQConfig(
            name="rubq",
            description="RuBQ",
            data_url="",
            data_dir="RuBQ"
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
                    "tags": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "question_eng": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        rubq_files = dl_manager.download(
            {
                "dev": _RUBQ_URLS["dev"],
                "test": _RUBQ_URLS["test"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", rubq_files["dev"]),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", rubq_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            rubq = json.load(f)
            for idx, question in enumerate(rubq):
                question["answers"] = json.dumps(question["answers"])
                yield idx, question

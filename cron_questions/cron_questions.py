"""CronKGQA."""

import os
import json
import pickle

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{saxena2021cronkgqa,
      title={Question Answering over Temporal Knowledge Graphs},
      author={Saxena, Apoorv and Chakrabarti, Soumen and Talukdar, Partha},
      booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
      year={2021}
    }
"""

_DESCRIPTION = """\
CronKGQA Dataset Description.
"""

_URL = "https://github.com/apoorvumang/CronKGQA"

_LICENSE = "MIT License"

class CronQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for CronQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for CronQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CronQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class CronQuestions(datasets.GeneratorBasedBuilder):
    """CronQuestions"""
    BUILDER_CONFIGS = [
        CronQuestionsConfig(
            name="cron_questions",
            description="CronQuestions",
            data_url="https://drive.google.com/u/0/uc?id=1wilPf3qohD-6156Daaz5M6GRuJiRP3P4&export=download",
            data_dir="CronQuestions"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            license=_LICENSE,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                    "answer_type": datasets.Value("string"),
                    "template": datasets.Value("string"),
                    "entities": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "times": datasets.Value("string"),
                    "relations": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "type": datasets.Value("string"),
                    "annotation": datasets.Value("string"),
                    "uniq_id": datasets.Value("int64"),
                    "paraphrases": datasets.Sequence(
                        datasets.Value("string")
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(self.config.data_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(download_dir, "data", "wikidata_big", "questions", "train.pickle"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(download_dir, "data", "wikidata_big", "questions", "valid.pickle"),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(download_dir, "data", "wikidata_big", "questions", "test.pickle"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, split, **kwargs):
        with open(data_file, "rb") as f:
            cronquestions = pickle.load(f)
            for idx, question in enumerate(cronquestions):
                question["annotation"] = json.dumps(question["annotation"])
                question["answers"] = json.dumps(list(question["answers"]))
                question["times"] = json.dumps(list(question["times"]))
                yield idx, question
"""ComplexQuestions."""

import json
import os
import ast

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{bao-etal-2016-constraint,
            title = "Constraint-Based Question Answering with Knowledge Graph",
            author = "Bao, Junwei and Duan, Nan and Yan, Zhao and Zhou, Ming and Zhao, Tiejun",
            booktitle = "Proceedings of {COLING} 2016, the 26th International Conference on Computational Linguistics: Technical Papers",
            month = dec,
            year = "2016",
            address = "Osaka, Japan",
            publisher = "The COLING 2016 Organizing Committee",
            url = "https://aclanthology.org/C16-1236",
            pages = "2503--2514"
        }
    """

_DESCRIPTION = """\
    ComplexQuestions Dataset Description
"""

_URL = ""

_COMPLEXQUESTIONS_URLS = {
    "train": "https://raw.githubusercontent.com/JunweiBao/MulCQA/ComplexQuestions/compQ.train.release",
    "test": "https://raw.githubusercontent.com/JunweiBao/MulCQA/ComplexQuestions/compQ.test.release"
}

class ComplexQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for ComplexQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for ComplexQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ComplexQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class ComplexQuestions(datasets.GeneratorBasedBuilder):
    """ComplexQuestions."""
    BUILDER_CONFIGS = [
        ComplexQuestionsConfig(
            name="complex_questions",
            description="ComplexQuestions",
            data_url="",
            data_dir="ComplexQuestions"
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
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        datasets.Value("string")
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        complexquestions_files = dl_manager.download(
            {
                "train": _COMPLEXQUESTIONS_URLS["train"],
                "test": _COMPLEXQUESTIONS_URLS["test"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", complexquestions_files["train"]),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", complexquestions_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            complexquestions = f.readlines()
            for idx, question in enumerate(complexquestions):
                yield idx, {
                    "question": question.strip().split("\t")[0],
                    "answers": ast.literal_eval(question.strip().split("\t")[1])
                }

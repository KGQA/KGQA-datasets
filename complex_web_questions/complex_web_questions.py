"""ComplexWebQuestions: A Dataset for Answering Complex Questions that Require Reasoning over Multiple Web Snippets."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{Talmor2018TheWA,
            title={The Web as a Knowledge-Base for Answering Complex Questions},
            author={Alon Talmor and Jonathan Berant},
            booktitle={NAACL},
            year={2018}
        }
    """

_DESCRIPTION = """\
    ComplexWebQuestions is a dataset for answering complex questions that require reasoning over multiple web snippets. It contains a large set of complex questions in natural language, and can be used in multiple ways: 1) By interacting with a search engine, which is the focus of our paper (Talmor and Berant, 2018); 2) As a reading comprehension task: we release 12,725,989 web snippets that are relevant for the questions, and were collected during the development of our model; 3) As a semantic parsing task: each question is paired with a SPARQL query that can be executed against Freebase to retrieve the answer.
"""

_URL = "https://allenai.org/data/complexwebquestions"
_COMPLEXWEBQUESTIONS_URLS = {
    "train": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AAAIHeWX0cPpbpwK6w06BCxva/ComplexWebQuestions_train.json?dl=1",
    "dev": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
    "test": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AABr4ysSy_Tg8Wfxww4i_UWda/ComplexWebQuestions_test.json?dl=1"
}

class ComplexWebQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for ComplexWebQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for ComplexWebQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ComplexWebQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class ComplexWebQuestions(datasets.GeneratorBasedBuilder):
    """ComplexWebQuestions: A Dataset for Answering Complex Questions that Require Reasoning over Multiple Web Snippets."""
    BUILDER_CONFIGS = [
        ComplexWebQuestionsConfig(
            name="complex_web_questions",
            description="ComplexWebQuestions",
            data_url="",
            data_dir="ComplexWebQuestions"
        ),
        ComplexWebQuestionsConfig(
            name="complexwebquestions_test",
            description="ComplexWebQuestions",
            data_url="",
            data_dir="ComplexWebQuestions"
        )
    ]

    def _info(self):
        features = datasets.Features(
                {
                    "ID": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        datasets.Features(
                            {
                                "aliases": datasets.features.Sequence(
                                    datasets.Value("string")
                                ),
                                "answer": datasets.Value("string"),
                                "answer_id": datasets.Value("string")
                            }
                        )
                    ),
                    "composition_answer": datasets.Value("string"),
                    "compositionality_type": datasets.Value("string"),
                    "created": datasets.Value("string"),
                    "machine_question": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "sparql": datasets.Value("string"),
                    "webqsp_ID": datasets.Value("string"),
                    "webqsp_question": datasets.Value("string")
                }
            )

        if self.config.name == "complexwebquestions_test":
            features.pop("answers", None)
            features.pop("composition_answer", None)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            features=features
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        if self.config.name == "complexwebquestions_test":
            complexwebquestions_test_files = dl_manager.download(
                {
                    "test": _COMPLEXWEBQUESTIONS_URLS["test"],
                }
            )
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", complexwebquestions_test_files["test"]),
                        "split": "test"
                    }
                )
            ]
        else:
            complexwebquestions_files = dl_manager.download(
                {
                    "train": _COMPLEXWEBQUESTIONS_URLS["train"],
                    "dev": _COMPLEXWEBQUESTIONS_URLS["dev"]
                }
            )
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", complexwebquestions_files["train"]),
                        "split": "train"
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", complexwebquestions_files["dev"]),
                        "split": "validation"
                    }
                )
            ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            complexwebquestions = json.load(f)
            for idx, question in enumerate(complexwebquestions):
                 yield idx, question

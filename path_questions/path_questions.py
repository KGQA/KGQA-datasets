"""PathQuestions."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{zhou-etal-2018-interpretable,
        title = "An Interpretable Reasoning Network for Multi-Relation Question Answering",
        author = "Zhou, Mantong and Huang, Minlie and Zhu, Xiaoyan",
        booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
        month = aug,
        year = "2018",
        address = "Santa Fe, New Mexico, USA",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/C18-1171",
        pages = "2010--2022",
        abstract = "Multi-relation Question Answering is a challenging task, due to the requirement of elaborated analysis on questions and reasoning over multiple fact triples in knowledge base. In this paper, we present a novel model called Interpretable Reasoning Network that employs an interpretable, hop-by-hop reasoning process for question answering. The model dynamically decides which part of an input question should be analyzed at each hop; predicts a relation that corresponds to the current parsed results; utilizes the predicted relation to update the question representation and the state of the reasoning process; and then drives the next-hop reasoning. Experiments show that our model yields state-of-the-art results on two datasets. More interestingly, the model can offer traceable and observable intermediate predictions for reasoning analysis and failure diagnosis, thereby allowing manual manipulation in predicting the final answer.",
    }
    """

_DESCRIPTION = """\
    PathQuestions Dataset Description
"""

_URL = "https://github.com/zmtkeke/IRN"

_PATHQUESTIONS_URLS = {
    "PQ2H": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQ-2H.txt",
    "PQ3H": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQ-3H.txt",
    "PQL2H": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQL-2H.txt",
    "PQL3H": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQL-3H.txt",
    "PQL3H_more": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQL-3H_more.txt",
    "PQ2H-KB": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/2H-kb.txt",
    "PQ3H-KB": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/3H-kb.txt",
    "PQL3H-KB": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQL3-KB.txt",
    "PQL2H-KB": "https://raw.githubusercontent.com/zmtkeke/IRN/master/PathQuestion/PQL2-KB.txt"
}

class PathQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for PathQuestions"""
    def __init__(self,
                 data_dir,
                 **kwargs):
        """BuilderConfig for PathQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PathQuestionsConfig, self).__init__(**kwargs)
        self.data_dir = data_dir

class PathQuestions(datasets.GeneratorBasedBuilder):
    """PathQuestions."""
    BUILDER_CONFIGS = [
        PathQuestionsConfig(
            name="PQ2H",
            description="PathQuestions 2H",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQ3H",
            description="PathQuestions 3H",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQL2H",
            description="PathQuestions Large 2H",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQL3H",
            description="PathQuestions Large 3H",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQL3H_more",
            description="PathQuestions Large 3H more",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQ2H-KB",
            description="PathQuestions 2H KG",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQ3H-KB",
            description="PathQuestions 3H KG",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQL2H-KB",
            description="PathQuestions Large 2H KG",
            data_dir="PathQuestions"
        ),
        PathQuestionsConfig(
            name="PQL3H-KB",
            description="PathQuestions Large 3H KG",
            data_dir="PathQuestions"
        )
    ]

    def _info(self):

        if self.config.name in ["PQ2H", "PQ3H", "PQL2H", "PQL3H"]:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                supervised_keys=None,
                homepage=_URL,
                citation=_CITATION,
                features=datasets.Features(
                    {
                        "question": datasets.Value("string"),
                        "answer": datasets.Value("string"),
                        "answer_set": datasets.Sequence(
                            datasets.Value("string")
                        ),
                        "path": datasets.Value("string")
                    }
                )
            )
        else:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                supervised_keys=None,
                homepage=_URL,
                citation=_CITATION,
                features=datasets.Features(
                    {
                        "subject": datasets.Value("string"),
                        "predicate": datasets.Value("string"),
                        "object": datasets.Value("string")
                    }
                )
            )

    def _split_generators(self, dl_manager):
        my_urls = _PATHQUESTIONS_URLS[self.config.name]
        pathquestions_file = dl_manager.download(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": pathquestions_file,
                    "split": "train"
                }
            )
        ]

    def _generate_examples(self, data_file, split, **kwargs):
        with open(data_file, encoding="utf-8") as f:
            if self.config.name in ["PQ2H", "PQ3H", "PQL2H", "PQL3H"]:
                pathquestions = f.readlines()
                for idx, line in enumerate(pathquestions):
                    question, answers, path = line.strip().split("\t")
                    answer = answers[:answers.find("(")]
                    answer_set_string = answers[answers.find("(")+1: answers.find(")")]
                    answer_set = [a for a in answer_set_string.split("/") if a != ""]
                    yield idx, {
                        "question": question,
                        "answer": answer,
                        "answer_set": answer_set,
                        "path": path
                    }
            else:
                kbs = f.readlines()
                for idx, line in enumerate(kbs):
                    subject, predicate, object = line.strip().split("\t")
                    yield idx, {
                        "subject": subject,
                        "predicate": predicate,
                        "object": object
                    }

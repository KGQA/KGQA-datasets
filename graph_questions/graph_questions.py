"""GraphQuestions: A Characteristic-Rich Dataset for Factoid Question Answering."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{su-etal-2016-generating,
            title = "On Generating Characteristic-rich Question Sets for {QA} Evaluation",
            author = {Su, Yu and Sun, Huan and Sadler, Brian and Srivatsa, Mudhakar and G{\"u}r, Izzeddin and Yan, Zenghui and Yan, Xifeng},
            booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
            month = nov,
            year = "2016",
            address = "Austin, Texas",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/D16-1054",
            doi = "10.18653/v1/D16-1054",
            pages = "562--572",
        }
    """

_DESCRIPTION = """\
    GraphQuestions Dataset Description
"""

_URL = "https://github.com/ysu1989/GraphQuestions"
_GRAPHQUESTIONS_URLS = {
    "train": "https://raw.githubusercontent.com/ysu1989/GraphQuestions/master/freebase13/graphquestions.training.json",
    "test": "https://raw.githubusercontent.com/ysu1989/GraphQuestions/master/freebase13/graphquestions.testing.json"
}

class GraphQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for GraphQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for GraphQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GraphQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class GraphQuestions(datasets.GeneratorBasedBuilder):
    """GraphQuestions: A Characteristic-Rich Dataset for Factoid Question Answering."""
    BUILDER_CONFIGS = [
        GraphQuestionsConfig(
            name="graph_questions",
            description="GraphQuestions",
            data_url="",
            data_dir="GraphQuestions"
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
                    "qid": datasets.Value("int64"),
                    "question": datasets.Value("string"),
                    "answer": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "function": datasets.Value("string"),
                    "commonness": datasets.Value("float64"),
                    "num_node": datasets.Value("int32"),
                    "num_edge": datasets.Value("int32"),
                    "graph_query": datasets.Features(
                        {
                            "nodes": datasets.features.Sequence(
                                datasets.Features(
                                    {
                                        "nid": datasets.Value("int32"),
                                        "node_type": datasets.Value("string"),
                                        "id": datasets.Value("string"),
                                        "class": datasets.Value("string"),
                                        "friendly_name": datasets.Value("string"),
                                        "question_node": datasets.Value("int32"),
                                        "function": datasets.Value("string")
                                    }
                                )
                            ),
                            "edges": datasets.features.Sequence(
                                datasets.Features(
                                    {
                                        "start": datasets.Value("int32"),
                                        "end": datasets.Value("int32"),
                                        "relation": datasets.Value("string"),
                                        "friendly_name": datasets.Value("string")
                                    }
                                )
                            )
                        }
                    ),
                    "sparql_query": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        graphquestions_files = dl_manager.download(
            {
                "train": _GRAPHQUESTIONS_URLS["train"],
                "test": _GRAPHQUESTIONS_URLS["test"]
            }
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", graphquestions_files["train"]),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", graphquestions_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            graphquestion = json.load(f)
            for idx, question in enumerate(graphquestion):
                yield idx, question

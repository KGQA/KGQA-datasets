"""GrailQA: The Strongly Generalizable Question Answering Dataset."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{gu2021beyond,
            title={Beyond IID: three levels of generalization for question answering on knowledge bases},
            author={Gu, Yu and Kase, Sue and Vanni, Michelle and Sadler, Brian and Liang, Percy and Yan, Xifeng and Su, Yu},
            booktitle={Proceedings of the Web Conference 2021},
            pages={3477--3488},
            organization={ACM}
        }
    """

_DESCRIPTION = """\
    GrailQA Dataset Description
"""

_URL = "https://dki-lab.github.io/GrailQA/"

class GrailQAConfig(datasets.BuilderConfig):
    """BuilderConfig for GrailQA"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for GrailQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GrailQAConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class GrailQA(datasets.GeneratorBasedBuilder):
    """GrailQA: The Strongly Generalizable Question Answering Dataset."""
    BUILDER_CONFIGS = [
        GrailQAConfig(
            name="grail_qa",
            description="GrailQA",
            data_url="https://dl.orangedox.com/WyaCpL?dl=1",
            data_dir="GrailQA_v1.0"
        ),
        GrailQAConfig(
            name="grailqa_test_public",
            description="GrailQA Public Test Dataset",
            data_url="https://dl.orangedox.com/WyaCpL?dl=1",
            data_dir="GrailQA_v1.0"
        )
    ]

    def _info(self):

        if self.config.name == "grailqa_test_public":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                supervised_keys=None,
                homepage=_URL,
                citation=_CITATION,
                features=datasets.Features(
                    {
                        "qid": datasets.Value("string"),
                        "question": datasets.Value("string")
                    }
                )
            )

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
                        datasets.Features(
                            {
                                "answer_type": datasets.Value("string"),
                                "answer_argument": datasets.Value("string"),
                                "entity_name": datasets.Value("string")
                            }
                        )
                    ),
                    "function": datasets.Value("string"),
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
                    "sparql_query": datasets.Value("string"),
                    "domains": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "level": datasets.Value("string"),
                    "s_expression": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(self.config.data_url)
        data_dir = os.path.join(download_dir, self.config.data_dir)

        if self.config.name == "grailqa_test_public":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "grailqa_v1.0_test_public.json"),
                        "split": "test"
                    }
                )
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "grailqa_v1.0_train.json"),
                        "split": "train"
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "grailqa_v1.0_dev.json"),
                        "split": "validation"
                    }
                )
            ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            grailqa = json.load(f)
            if self.config.name == "grailqa_test_public":
                for idx, question in enumerate(grailqa):
                    yield idx, question
            else:
                for idx, question in enumerate(grailqa):
                    if not question.get("level", None):
                        question["level"] = "null"
                    for answer in question["answer"]:
                        if not answer.get("entity_name", None):
                            answer["entity_name"] = "null"
                    yield idx, question

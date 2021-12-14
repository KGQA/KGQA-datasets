"""WebQuestionsSP: The WebQuestions Semantic Parses Dataset"""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{yih-etal-2016-value,
        title={The Value of Semantic Parse Labeling for Knowledge Base Question Answering},
        author={Yih, Wen-tau and Richardson, Matthew and Meek, Chris and Chang, Ming-Wei and Suh, Jina},
        booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
        year={2016},
        publisher={Association for Computational Linguistics},
        pages={201--206},
        }
    """

_DESCRIPTION = """\
    WebQuestionsSP Dataset Description
"""

_URL = "https://www.microsoft.com/en-us/download/details.aspx?id=52763"

class WebQSPConfig(datasets.BuilderConfig):
    """BuilderConfig for WebQuestionsSP"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for WebQuestionsSP.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WebQSPConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class WebQuestionsSP(datasets.GeneratorBasedBuilder):
    """WebQuestionsSP: The WebQuestions Semantic Parses Dataset"""
    BUILDER_CONFIGS = [
        WebQSPConfig(
            name="webqsp",
            description="WebQSP",
            data_dir="WebQSP",
            data_url="https://download.microsoft.com/download/F/5/0/F5012144-A4FB-4084-897F-CFDA99C60BDF/WebQSP.zip"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "QuestionId": datasets.Value("string"),
                "RawQuestion": datasets.Value("string"),
                "ProcessedQuestion": datasets.Value("string"),
                "Parses": datasets.Sequence(
                    datasets.Features({
                        "ParseId": datasets.Value("string"),
                        "AnnotatorId": datasets.Value("int32"),
                        "AnnotatorComment": datasets.Features({
                            "ParseQuality": datasets.Value("string"),
                            "QuestionQuality": datasets.Value("string"),
                            "Confidence": datasets.Value("string"),
                            "FreeFormComment": datasets.Value("string")
                        }),
                        "Sparql": datasets.Value("string"),
                        "PotentialTopicEntityMention": datasets.Value("string"),
                        "TopicEntityName": datasets.Value("string"),
                        "TopicEntityMid": datasets.Value("string"),
                        "InferentialChain": datasets.Sequence(
                            datasets.Value("string")
                        ),
                        "Constraints": datasets.Sequence(
                            datasets.Features({
                                "Operator": datasets.Value("string"),
                                "ArgumentType": datasets.Value("string"),
                                "Argument": datasets.Value("string"),
                                "EntityName": datasets.Value("string"),
                                "SourceNodeIndex": datasets.Value("int32"),
                                "NodePredicate": datasets.Value("string"),
                                "ValueType": datasets.Value("string")
                            })
                        ),
                        "Answers": datasets.Sequence(
                            datasets.Features({
                                "AnswerType": datasets.Value("string"),
                                "AnswerArgument": datasets.Value("string"),
                                "EntityName": datasets.Value("string")
                            })
                        )
                    })
                ),
            }),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(self.config.data_url)
        data_dir = os.path.join(download_dir, self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, "data", "WebQSP.train.json"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, "data", "WebQSP.test.json"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            webqsp = json.load(f)
            for idx, question in enumerate(webqsp["Questions"]):
                parses = []
                for parse in question["Parses"]:
                    parse.pop("Time", None)
                    parse.pop("Order", None)
                    parses.append(parse)
                yield idx, {
                  "QuestionId": question["QuestionId"],
                  "RawQuestion": question["RawQuestion"],
                  "ProcessedQuestion": question["ProcessedQuestion"],
                  "Parses": parses
                }
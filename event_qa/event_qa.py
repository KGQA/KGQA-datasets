"""Event-QA"""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{souza2019eventqa,
        title={Event-QA: A Dataset for Event-Centric Question Answering over Knowledge Graphs},
        author={Souza, Tarcisio, Gottschalk, Simon and Demidova, Elena},
        year={2019}
    }
    """

_DESCRIPTION = """\
    Event-QA is a Dataset for answering Event-Centric questions over Knowledge Graphs (KGs). We target EventKG, a recently proposed Event-Centric KG. In our paper we describe details about the dataset creation process. Current datasets are mostly entity-centric, Event-QA is the first Question Answering dataset that focus on events.
"""

_URL = "https://eventcqa.l3s.uni-hannover.de/"

class EventQAConfig(datasets.BuilderConfig):
    """BuilderConfig for Event-QA"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for Event-QA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(EventQAConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class EventQA(datasets.GeneratorBasedBuilder):
    """Event-QA"""
    BUILDER_CONFIGS = [
        EventQAConfig(
            name="eventkg",
            description="Event-QA EventKG",
            data_dir="EventQA",
            data_url="https://eventcqa.l3s.uni-hannover.de/dataset/data.tgz"
        ),
        EventQAConfig(
            name="dbpedia",
            description="Event-QA DBpedia",
            data_dir="EventQA",
            data_url="https://eventcqa.l3s.uni-hannover.de/dataset/data.tgz"
        ),
        EventQAConfig(
            name="predicates",
            description="Event-QA's predicates",
            data_dir="EventQA",
            data_url="https://eventcqa.l3s.uni-hannover.de/dataset/data.tgz"
        ),
        EventQAConfig(
            name="entities",
            description="Event-QA's entities",
            data_dir="EventQA",
            data_url="https://eventcqa.l3s.uni-hannover.de/dataset/data.tgz"
        ),
        EventQAConfig(
            name="events",
            description="Event-QA's events",
            data_dir="EventQA",
            data_url="https://eventcqa.l3s.uni-hannover.de/dataset/data.tgz"
        ),
    ]

    def _info(self):

        if self.config.name in ["predicates", "entities", "events"]:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                supervised_keys=None,
                homepage=_URL,
                citation=_CITATION,
                features=datasets.Features(
                    {
                        "name": datasets.Value("string")
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
                    "knowledgegraph": datasets.Value("string"),
                    "hybrid": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Features(
                        {
                            "sparql": datasets.Value("string")
                        }
                    ),
                    "onlydbo": datasets.Value("string"),
                    "answertype": datasets.Value("string"),
                    "aggregation": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "answers": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(self.config.data_url)
        data_dir = download_dir
        if self.config.name in ["predicates", "entities", "events"]:
            file_name = "{}.txt".format(self.config.name)
        else:
            file_name = "eventqa-train-multilingual-{}.json".format(self.config.name)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", file_name),
                    "split": "train"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        if self.config.name in ["predicates", "entities", "events"]:
            lines = open(data_file, encoding="utf-8").readlines()
            for idx, line in enumerate(lines):
                yield idx, {"name": line.strip()}
        else:
            all_data = json.load(open(data_file, encoding="utf8"))
            questions = all_data["questions"]
            for idx, question in enumerate(questions):
                if question.get("answerType"):
                    question["answertype"] = question.pop("answerType")
                if question.get("onlyDbo"):
                    question["onlydbo"] = question.pop("onlyDbo")
                question["question"] = json.dumps(question["question"])
                question["id"] = str(question["id"])
                question["answers"] = json.dumps(question["answers"])
                yield idx, question
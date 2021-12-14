"""LC-QuAD: A Large Scale Complex Question Answering Dataset."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{trivedi2017lc,
        title={Lc-quad: A corpus for complex question answering over knowledge graphs},
        author={Trivedi, Priyansh and Maheshwari, Gaurav and Dubey, Mohnish and Lehmann, Jens},
        booktitle={International Semantic Web Conference},
        pages={210--218},
        year={2017},
        organization={Springer}
    }
    """

_DESCRIPTION = """\
    LC-QuAD is a Question Answering dataset with 5000 pairs of question and its corresponding SPARQL query. The target knowledge base is DBpedia, specifically, the April, 2016 version. Please see our paper for details about the dataset creation process and framework.
"""

_URL = "http://lc-quad.sda.tech/lcquad1.0.html"
_LCQUAD_URLS = {
    "train": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/train-data.json",
    "test": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/test-data.json",
    "entities": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/resources/entities.txt",
    "predicates": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/resources/predicates.txt",
    "templates": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/resources/templates.json"
}

class LCQuADConfig(datasets.BuilderConfig):
    """BuilderConfig for LC-QuAD"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for LC-QuAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LCQuADConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class LCQuAD(datasets.GeneratorBasedBuilder):
    """LC-QuAD: A Large Scale Complex Question Answering Dataset."""
    BUILDER_CONFIGS = [
        LCQuADConfig(
            name="lcquad",
            description="LCQuAD",
            data_url="",
            data_dir="LCQuAD"
        ),
        LCQuADConfig(
            name="entities",
            description="LCQuAD's entities",
            data_url="",
            data_dir="LCQuAD"
        ),
        LCQuADConfig(
            name="predicates",
            description="LCQuAD's predicates",
            data_url="",
            data_dir="LCQuAD"
        ),
        LCQuADConfig(
            name="templates",
            description="LCQuAD's templates",
            data_url="",
            data_dir="LCQuAD"
        )
    ]

    def _info(self):

        if self.config.name == "entities":
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

        if self.config.name == "predicates":
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

        if self.config.name == "templates":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                supervised_keys=None,
                homepage=_URL,
                citation=_CITATION,
                features=datasets.Features(
                    {
                        "id": datasets.Value("int32"),
                        "n_entities": datasets.Value("int32"),
                        "template": datasets.Value("string"),
                        "type": datasets.Value("string")
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
                    "_id": datasets.Value("string"),
                    "corrected_question": datasets.Value("string"),
                    "intermediary_question": datasets.Value("string"),
                    "sparql_query": datasets.Value("string"),
                    "sparql_template_id": datasets.Value("int32")
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        lcquad_files = dl_manager.download(
            {
                "train": _LCQUAD_URLS["train"],
                "test": _LCQUAD_URLS["test"],
                "entities": _LCQUAD_URLS["entities"],
                "predicates": _LCQUAD_URLS["predicates"],
                "templates": _LCQUAD_URLS["templates"],
            }
        )

        if self.config.name == "entities":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", lcquad_files["entities"]),
                        "split": "train"
                    }
                )
            ]

        if self.config.name == "predicates":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", lcquad_files["predicates"]),
                        "split": "train"
                    }
                )
            ]

        if self.config.name == "templates":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir or "", lcquad_files["templates"]),
                        "split": "train"
                    }
                )
            ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", lcquad_files["train"]),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", lcquad_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        if self.config.name == "entities" or self.config.name == "predicates":
            lines = open(data_file, encoding="utf-8").readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if self.config.name == "predicates" and "," in line: line = line[0:-1]
                yield idx, {"name": line}
        elif self.config.name == "templates" or self.config.name == "lcquad":
            with open(data_file, encoding="utf8") as f:
                all_data = json.load(f)
                for idx, example in enumerate(all_data):
                    yield idx, example

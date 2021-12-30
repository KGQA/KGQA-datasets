"""LC-QuAD 2.0: A Large Scale Complex Question Answering Dataset."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{dubey2017lc2,
        title={LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia},
        author={Dubey, Mohnish and Banerjee, Debayan and Abdelkawi, Abdelrahman and Lehmann, Jens},
        booktitle={Proceedings of the 18th International Semantic Web Conference (ISWC)},
        year={2019},
        organization={Springer}
    }
    """

_DESCRIPTION = """\
    LC-QuAD 2.0 is a Large Question Answering dataset with 30,000 pairs of question and its corresponding SPARQL query. The target knowledge base is Wikidata and DBpedia, specifically the 2018 version. Please see our paper for details about the dataset creation process and framework.
"""

_URL = "http://lc-quad.sda.tech/index.html"
_LCQUAD2_URLS = {
    "train": "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15738824/train.json",
    "test": "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15738818/test.json"
}

class LCQuAD2Config(datasets.BuilderConfig):
    """BuilderConfig for LC-QuAD 2.0"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for LC-QuAD 2.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LCQuAD2Config, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class LCQuAD2(datasets.GeneratorBasedBuilder):
    """LC-QuAD 2.0: A Large Scale Complex Question Answering Dataset."""
    BUILDER_CONFIGS = [
        LCQuAD2Config(
            name="lcquad2-wikidata",
            description="LCQuAD2 Wikidata",
            data_url="",
            data_dir="LCQuAD2"
        ),
        LCQuAD2Config(
            name="lcquad2-dbpedia",
            description="LCQuAD2 DBpedia",
            data_url="",
            data_dir="LCQuAD2"
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
                    "NNQT_question": datasets.Value("string"),
                    "uid": datasets.Value("string"),
                    "subgraph": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "template_index": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "sparql": datasets.Value("string"),
                    "template": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "template_id": datasets.Value("string"),
                    "answer": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "paraphrased_question": datasets.Sequence(
                        datasets.Value("string")
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        lcquad2_files = dl_manager.download(
            {
                "train": _LCQUAD2_URLS["train"],
                "test": _LCQUAD2_URLS["test"]
            }
        )
        if self.config.name == "lcquad2-wikidata":
            kb = "wikidata"
        elif self.config.name == "lcquad2-dbpedia":
            kb = "dbpedia"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", lcquad2_files["train"]),
                    "split": "train",
                    "kb": kb
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", lcquad2_files["test"]),
                    "split": "test",
                    "kb": kb
                }
            )
        ]

    def _generate_examples(self, data_file, split, kb, **kwargs):
        with open(data_file, encoding="utf8") as f:
            lcquad2 = json.load(f)
            for idx, question in enumerate(lcquad2):
                paraphrases = []
                templates = []
                subgraphs = []
                if isinstance(question["paraphrased_question"], list):
                    paraphrases = question["paraphrased_question"]
                elif isinstance(question["paraphrased_question"], str):
                    paraphrases.append(question["paraphrased_question"])

                if isinstance(question["template"], list):
                    templates = question["template"]
                elif isinstance(question["template"], str):
                    templates.append(question["template"])

                if isinstance(question["subgraph"], list):
                    subgraphs = question["subgraph"]
                elif isinstance(question["subgraph"], str):
                    subgraphs.append(question["subgraph"])
                if kb == "wikidata":
                    sparql = question["sparql_wikidata"]
                elif kb == "dbpedia":
                    sparql = question["sparql_dbpedia18"]
                yield idx, {
                    "NNQT_question": question["NNQT_question"],
                    "uid": str(question["uid"]),
                    "subgraph": subgraphs,
                    "template_index": str(question["template_index"]),
                    "question": question["question"],
                    "sparql": sparql,
                    "template": templates,
                    "template_id": str(question["template_id"]),
                    "answer": question["answer"],
                    "paraphrased_question": paraphrases
                }

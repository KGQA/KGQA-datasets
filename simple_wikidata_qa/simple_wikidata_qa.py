"""SimpleWikidataQuestions: A Dataset for Simple QA over Knowledge Graphs on Wikidata."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{wikidata-benchmark,
      author    = {Dennis Diefenbach and Thomas Pellissier Tanon and Kamal Deep Singh and Pierre Maret},
      title     = {Question Answering Benchmarks for Wikidata},
      booktitle = {Proceedings of the {ISWC} 2017 Posters {\&} Demonstrations and
                   Industry Tracks co-located with 16th International Semantic Web Conference
                   {(ISWC} 2017), Vienna, Austria, October 23rd - to - 25th, 2017.},
      year      = {2017},
      url       = {http://ceur-ws.org/Vol-1963/paper555.pdf}
    }
    """

_DESCRIPTION = """\
    SimpleWikidataQuestions: A Dataset for Simple QA over Knowledge Graphs on Wikidata.
"""

_URL = "https://github.com/askplatypus/wikidata-simplequestions"
_DATA_URLS = {
    "train": "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/annotated_wd_data_train_answerable.txt",
    "dev": "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/annotated_wd_data_valid_answerable.txt",
    "test": "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/annotated_wd_data_test_answerable.txt"
}

class SimpleWikidataQAConfig(datasets.BuilderConfig):
    """BuilderConfig for SimpleWikidataQuestions"""
    def __init__(self,
                 data_dir,
                 **kwargs):
        """BuilderConfig for SimpleWikidataQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SimpleWikidataQAConfig, self).__init__(**kwargs)
        self.data_dir = data_dir

class SimpleWikidataQA(datasets.GeneratorBasedBuilder):
    """SimpleWikidataQuestions: A Dataset for Simple QA over Knowledge Graphs on Wikidata."""
    BUILDER_CONFIGS = [
        SimpleWikidataQAConfig(
            name="simple_wikidata_qa",
            description="SimpleWikidataQA",
            data_dir="SimpleWikidataQA"
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
                    "answer": datasets.Features(
                        {
                            "subject": datasets.Value("string"),
                            "predicate": datasets.Value("string"),
                            "object": datasets.Value("string")
                        }
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_dir = None
        data_files = dl_manager.download(
            {
                "train": _DATA_URLS["train"],
                "dev": _DATA_URLS["dev"],
                "test": _DATA_URLS["test"]
            }
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", data_files["train"]),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", data_files["dev"]),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir or "", data_files["test"]),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            qa_data = f.readlines()
            for idx, line in enumerate(qa_data):
                yield idx, {
                    "question": line.strip().split("\t")[3],
                    "answer": {
                        "subject": line.strip().split("\t")[0],
                        "predicate": line.strip().split("\t")[1],
                        "object": line.strip().split("\t")[2]
                    }
                }

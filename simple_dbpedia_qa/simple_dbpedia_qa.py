"""SimpleDBpediaQA: A Dataset for Simple QA over Knowledge Graphs on DBpedia."""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{azmy-etal-2018-farewell,
        title = "Farewell {F}reebase: Migrating the {S}imple{Q}uestions Dataset to {DB}pedia",
        author = "Azmy, Michael and Shi, Peng and Lin, Jimmy and Ilyas, Ihab",
        booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
        month = aug,
        year = "2018",
        address = "Santa Fe, New Mexico, USA",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/C18-1178",
        pages = "2093--2103",
        abstract = "Question answering over knowledge graphs is an important problem of interest both commercially and academically. There is substantial interest in the class of natural language questions that can be answered via the lookup of a single fact, driven by the availability of the popular SimpleQuestions dataset. The problem with this dataset, however, is that answer triples are provided from Freebase, which has been defunct for several years. As a result, it is difficult to build {``}real-world{''} question answering systems that are operationally deployable. Furthermore, a defunct knowledge graph means that much of the infrastructure for querying, browsing, and manipulating triples no longer exists. To address this problem, we present SimpleDBpediaQA, a new benchmark dataset for simple question answering over knowledge graphs that was created by mapping SimpleQuestions entities and predicates from Freebase to DBpedia. Although this mapping is conceptually straightforward, there are a number of nuances that make the task non-trivial, owing to the different conceptual organizations of the two knowledge graphs. To lay the foundation for future research using this dataset, we leverage recent work to provide simple yet strong baselines with and without neural networks.",
    }
    """

_DESCRIPTION = """\
    SimpleDBpediaQA: A Dataset for Simple QA over Knowledge Graphs on DBpedia.
"""

_URL = "https://github.com/castorini/SimpleDBpediaQA"
_DATA_URLS = {
    "train": "https://raw.githubusercontent.com/castorini/SimpleDBpediaQA/master/V1/train.json",
    "dev": "https://raw.githubusercontent.com/castorini/SimpleDBpediaQA/master/V1/valid.json",
    "test": "https://raw.githubusercontent.com/castorini/SimpleDBpediaQA/master/V1/test.json"
}

class SimpleDBpediaQAConfig(datasets.BuilderConfig):
    """BuilderConfig for SimpleDBpediaQA"""
    def __init__(self,
                 data_dir,
                 **kwargs):
        """BuilderConfig for SimpleDBpediaQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SimpleDBpediaQAConfig, self).__init__(**kwargs)
        self.data_dir = data_dir

class SimpleDBpediaQA(datasets.GeneratorBasedBuilder):
    """SimpleDBpediaQA: A Dataset for Simple QA over Knowledge Graphs on DBpedia."""
    BUILDER_CONFIGS = [
        SimpleDBpediaQAConfig(
            name="simple_dbpedia_qa",
            description="SimpleDBpediaQA",
            data_dir="SimpleDBpediaQA"
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
                    "ID": datasets.Value("string"),
                    "Query": datasets.Value("string"),
                    "Subject": datasets.Value("string"),
                    "FreebasePredicate": datasets.Value("string"),
                    "PredicateList": datasets.Sequence(
                        datasets.Features(
                            {
                                "Predicate": datasets.Value("string"),
                                "Direction": datasets.Value("string"),
                                "Constraint": datasets.Value("string")
                            }
                        )
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
            qa_data = json.load(f)
            for idx, question in enumerate(qa_data["Questions"]):
                yield idx, question

"""TimeQuestions"""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{10.1145/3459637.3482416,
        author = {Jia, Zhen and Pramanik, Soumajit and Saha Roy, Rishiraj and Weikum, Gerhard},
        title = {Complex Temporal Question Answering on Knowledge Graphs},
        year = {2021},
        isbn = {9781450384469},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3459637.3482416},
        doi = {10.1145/3459637.3482416},
        abstract = {Question answering over knowledge graphs (KG-QA) is a vital topic in IR. Questions with temporal intent are a special class of practical importance, but have not received much attention in research. This work presents EXAQT, the first end-to-end system for answering complex temporal questions that have multiple entities and predicates, and associated temporal conditions. EXAQT answers natural language questions over KGs in two stages, one geared towards high recall, the other towards precision at top ranks. The first step computes question-relevant compact subgraphs within the KG, and judiciously enhances them with pertinent temporal facts, using Group Steiner Trees and fine-tuned BERT models. The second step constructs relational graph convolutional networks (R-GCNs) from the first step's output, and enhances the R-GCNs with time-aware entity embeddings and attention over temporal relations. We evaluate EXAQT on TimeQuestions, a large dataset of 16k temporal questions we compiled from a variety of general purpose KG-QA benchmarks. Results show that EXAQT outperforms three state-of-the-art systems for answering complex questions over KGs, thereby justifying specialized treatment of temporal QA.},
        booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
        pages = {792–802},
        numpages = {11},
        keywords = {complex questions, knowledge graphs, temporal question answering},
        location = {Virtual Event, Queensland, Australia},
        series = {CIKM '21}
        }
    """

_DESCRIPTION = """\
    TimeQuestions Dataset Description
"""

_URL = "https://exaqt.mpi-inf.mpg.de/"

class TimeQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for TimeQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for TimeQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TimeQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class TimeQuestions(datasets.GeneratorBasedBuilder):
    """TimeQuestions"""
    BUILDER_CONFIGS = [
        TimeQuestionsConfig(
            name="time_questions",
            description="TimeQuestions",
            data_dir="TimeQuestions",
            data_url="https://qa.mpi-inf.mpg.de/exaqt/TimeQuestions.zip"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Id": datasets.Value("int32"),
                    "Question": datasets.Value("string"),
                    "Temporal signal": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "Temporal question type": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "Answer": datasets.Value("string"),
                    "Data source": datasets.Value("string"),
                    "Question creation date": datasets.Value("string"),
                    "Data set": datasets.Value("string")
                }
            ),
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
                    "data_file": os.path.join(data_dir, "train.json"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, "dev.json"),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, "test.json"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            timequestions = json.load(f)
            for idx, question in enumerate(timequestions):
                question["Answer"] = json.dumps(question["Answer"])
                yield idx, question
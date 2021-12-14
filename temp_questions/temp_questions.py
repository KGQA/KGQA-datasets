"""TempQuestions"""

import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
    @inproceedings{jia2018tempquestions,
      title={Tempquestions: A benchmark for temporal question answering},
      author={Jia, Zhen and Abujabal, Abdalghani and Saha Roy, Rishiraj and Str{\"o}tgen, Jannik and Weikum, Gerhard},
      booktitle={Companion Proceedings of the The Web Conference 2018},
      pages={1057--1062},
      year={2018}
    }
    """

_DESCRIPTION = """\
    TempQuestions Dataset Description
"""

_URL = "https://tequila.mpi-inf.mpg.de/"

class TempQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for TempQuestions"""
    def __init__(self,
                 data_url,
                 data_dir,
                 **kwargs):
        """BuilderConfig for TempQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TempQuestionsConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.data_dir = data_dir

class TempQuestions(datasets.GeneratorBasedBuilder):
    """TimeQuestions"""
    BUILDER_CONFIGS = [
        TempQuestionsConfig(
            name="temp_questions",
            description="TempQuestions",
            data_dir="TempQuestions",
            data_url="http://qa.mpi-inf.mpg.de/TempQuestions.zip"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "Id": datasets.Value("int32"),
                    "Question": datasets.Value("string"),
                    "Gold answer": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "Temporal signal": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "Type": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "Data source": datasets.Value("string"),
                    "Question creation date": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(self.config.data_url)
        data_dir = os.path.join(download_dir, self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, "TempQuestions.json"),
                    "split": "train"
                }
            )
        ]

    def _generate_examples(self, data_file, **kwargs):
        with open(data_file, encoding="utf8") as f:
            tempquestions = json.load(f)
            for idx, question in enumerate(tempquestions):
                yield idx, question
# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MetaQA dataset"""


import os

import numpy as np

import datasets


_DESCRIPTION = """\
MetaQA Dataset Description.
"""

_CITATION = """\
    @inproceedings{zhang2017variational,
      title={Variational Reasoning for Question Answering with Knowledge Graph},
      author={Zhang, Yuyu and Dai, Hanjun and Kozareva, Zornitsa and Smola, Alexander J and Song, Le},
      booktitle={AAAI},
      year={2018}
    }
"""

_HOMEPAGE = "https://github.com/yuyuz/MetaQA"

_LICENSE = "CC License"

class MetaQAConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super(MetaQAConfig, self).__init__(**kwargs)


class MetaQA(datasets.GeneratorBasedBuilder):
    """MetaQA."""

    BUILDER_CONFIGS = [
        MetaQAConfig(
            name="metaqa-1hop-vanilla",
            description="MetaQA 1hop vanilla"
        ),
        MetaQAConfig(
            name="metaqa-2hop-vanilla",
            description="MetaQA 2hop vanilla"
        ),
        MetaQAConfig(
            name="metaqa-3hop-vanilla",
            description="MetaQA 2hop vanilla"
        ),
        MetaQAConfig(
            name="metaqa-1hop-ntm",
            description="MetaQA 1hop ntm"
        ),
        MetaQAConfig(
            name="metaqa-2hop-ntm",
            description="MetaQA 2hop ntm"
        ),
        MetaQAConfig(
            name="metaqa-3hop-ntm",
            description="MetaQA 3hop ntm"
        ),
        MetaQAConfig(
            name="kb",
            description="MetaQA kb.txt"
        ),
        MetaQAConfig(
            name="qtype-1hop",
            description="MetaQA 1hop question type files"
        ),
        MetaQAConfig(
            name="qtype-2hop",
            description="MetaQA 2hop question type files"
        ),
        MetaQAConfig(
            name="qtype-3hop",
            description="MetaQA 3hop question type files"
        )
    ]

    @property
    def manual_download_instructions(self):
        return """\
            To use MetaQA dataset you have to download it manually from Google Drive: https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g
            Please extract all files in one folder and then load the dataset with:
            `datasets.load_dataset('meta_qa', data_dir='/path/to/extracted/data/')`"""

    def _info(self):

        if self.config.name == "kb":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                citation=_CITATION,
                supervised_keys=None,
                homepage=_HOMEPAGE,
                license=_LICENSE,
                features=datasets.Features(
                    {
                        "subject": datasets.Value("string"),
                        "predicate": datasets.Value("string"),
                        "object": datasets.Value("string")
                    }
                )
            )

        if "qtype" in self.config.name:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                citation=_CITATION,
                supervised_keys=None,
                homepage=_HOMEPAGE,
                license=_LICENSE,
                features=datasets.Features(
                    {
                        "qtype": datasets.Value("string")
                    }
                )
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        datasets.Value("string")
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"{data_dir} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('meta_qa', data_dir=...)`. Manual download instructions: {self.manual_download_instructions}"
            )

        if self.config.name == "kb":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split("kb"),
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "kb.txt"),
                        "split": "kb"
                    }
                )
            ]

        if "qtype" in self.config.name:
            hop_dir = "{}-hop".format(self.config.name.split("-")[1][0])
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, hop_dir, "qa_train_qtype.txt"),
                        "split": "train"
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, hop_dir, "qa_dev_qtype.txt"),
                        "split": "validation"
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, hop_dir, "qa_test_qtype.txt"),
                        "split": "test"
                    }
                )
            ]

        hop_dir = "{}-hop".format(self.config.name.split("-")[1][0])
        type_dir = self.config.name.split("-")[2]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, hop_dir, type_dir, "qa_train.txt"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, hop_dir, type_dir, "qa_dev.txt"),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(data_dir, hop_dir, type_dir, "qa_test.txt"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, data_file, split, **kwargs):
        """Yields examples."""
        if self.config.name == "kb":
            all_lines = np.loadtxt(data_file, delimiter="|", dtype=str)
            for idx, line in enumerate(all_lines):
                print(line)
                yield idx, {
                    "subject": line[0],
                    "predicate": line[1],
                    "object": line[2]
                }
        elif "qtype" in self.config.name:
            all_lines = open(data_file, encoding="utf-8").readlines()
            for idx, line in enumerate(all_lines):
                yield idx, {
                    "qtype": line.strip()
                }
        else:
            all_lines = np.loadtxt(data_file, delimiter="\t", dtype=str)
            for idx, line in enumerate(all_lines):
                answers = [str(answer) for answer in line[1].split("|")]
                yield idx, {
                    "question": line[0],
                    "answers": answers
                }
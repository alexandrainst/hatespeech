import os
import random
import datasets
import pandas as pd

random.seed(703)

_DESCRIPTION = """DR Hatespeech dataset."""


class HateSpeechData(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="DR_Hatespeech",
            version=VERSION,
            description="Hatespeech data from DR Facebook.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "dr_hate"

    def _info(self):
        if self.config.name == "dr_hate":
            features = datasets.Features(
                {
                    "platform": datasets.Value("string"),
                    "account": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "action": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("text", "action"),
        )

    def _split_generators(self, dl_manager):
        data_dir = "data/split"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": os.path.join(data_dir, "train.parquet")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": os.path.join(data_dir, "test.parquet")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": os.path.join(data_dir, "val.parquet")
                },
            ),
        ]

    def _generate_examples(self, filepath):
        data = pd.read_parquet(filepath)
        for i, row in enumerate(data.iterrows()):
            yield i, {
                "platform": row["platform"],
                "account": row["account"],
                "text": row["text"],
                "url": row["url"],
                "date": row["date"],
                "action": row["action"],
            }

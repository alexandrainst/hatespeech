"""Prepares the weakly supervised data for annotation in LabelStudio."""

from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .load_data import load_weakly_supervised_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def prepare_data_for_annotation(config: DictConfig) -> List[str]:
    """Prepares the weakly supervised data for annotation in LabelStudio."""

    # Load the weakly supervised data
    df = load_weakly_supervised_data(config)

    # Use the weakly supervised labels to sample a balanced subset of the data
    sample_size = config.data.val.size + config.data.test.size
    df_off = df.query("label == 1").sample(sample_size // 2, random_state=config.seed)
    df_not = df.query("label == 0").sample(sample_size // 2, random_state=config.seed)

    # Concatenate the dataframes, shuffle them and extract the text
    df_merged = pd.concat([df_off, df_not]).sample(frac=1, random_state=config.seed)
    texts = df_merged.text.tolist()

    # Save the data to a txt file
    data_dir = Path(config.data.for_annotation.dir)
    path = data_dir / config.data.for_annotation.fname
    path.write_text("\n".join(texts))

    # If we are testing then create some fake labels for the data and store the
    # resulting dataframe in the annotated directory, and split up the dataframe into a
    # validation and test split, and store those too
    if config.testing:
        # Create test annotated data
        labels = [np.random.choice(["Offensive", "Not Offensive"]) for _ in texts]
        df_test = pd.DataFrame({"text": texts, "label": labels})
        df_test.to_parquet(data_dir / config.data.annotated.fname)

        # Create test validation and test data
        val_path = Path(config.data.val.dir) / config.data.val.fname
        df_test.iloc[:2].to_parquet(val_path)
        test_path = Path(config.data.test.dir) / config.data.test.fname
        df_test.iloc[2:].to_parquet(test_path)

    # Return the prepared data
    return texts


if __name__ == "__main__":
    prepare_data_for_annotation()

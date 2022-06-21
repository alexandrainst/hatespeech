"""Splits the weakly supervised data into training, validation and test sets."""

from pathlib import Path
from typing import Dict

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .load_data import load_weakly_supervised_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def split_data(config: DictConfig) -> Dict[str, pd.DataFrame]:
    """Splits the weakly supervised data into training, validation and test sets.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        dict:
            Dictionary containing the training, validation and test sets, with keys
            "train", "val" and "test".
    """
    # Load the weakly supervised data
    df = load_weakly_supervised_data(config=config)["df"]

    # Split the data into a training set and a combined validation and test set
    train, val_test = train_test_split(
        df,
        test_size=config.data.val_size + config.data.test_size,
        stratify=df.label,
        random_state=config.data.seed,
    )

    # Split the combined validation and test set into validation and test set
    val, test = train_test_split(
        val_test,
        test_size=config.data.test_size,
        stratify=df.label,
        random_state=config.data.seed,
    )

    # Store the splits in the `final` data directory
    train.to_parquet(Path(config.data.final_dir) / "train.parquet")
    val.to_parquet(Path(config.data.final_dir) / "val.parquet")
    test.to_parquet(Path(config.data.final_dir) / "test.parquet")

    # Return the training, validation and test sets
    return dict(train=train, val=val, test=test)

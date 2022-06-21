"""Splits the weakly supervised data into training, validation and test sets."""

from pathlib import Path
from typing import Dict

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .load_data import load_weakly_supervised_data


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

    # Get validation and test sizes from the config
    if config.test:
        val_size = 1
        test_size = 1
    else:
        val_size = config.data.val_size
        test_size = config.data.test_size

    # Split the data into a training set and a combined validation and test set
    train, val_test = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df.label,
        random_state=config.seed,
    )

    # Split the combined validation and test set into validation and test set
    val, test = train_test_split(
        val_test,
        test_size=test_size,
        stratify=val_test.label,
        random_state=config.seed,
    )

    # Define filenames based on whether `test` is True or False
    if config.test:
        train_file = "test_train.parquet"
        val_file = "test_val.parquet"
        test_file = "test_test.parquet"
    else:
        train_file = "train.parquet"
        val_file = "val.parquet"
        test_file = "test.parquet"

    # Store the splits in the `final` data directory
    train.to_parquet(Path(config.data.final_dir) / train_file)
    val.to_parquet(Path(config.data.final_dir) / val_file)
    test.to_parquet(Path(config.data.final_dir) / test_file)

    # Return the training, validation and test sets
    return dict(train=train, val=val, test=test)

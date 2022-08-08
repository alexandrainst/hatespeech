"""Creates the training split of the data from the weakly supervised data."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from .load_data import load_annotated_data, load_weakly_supervised_data

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def create_train_split(config: DictConfig) -> pd.DataFrame:
    """Creates the training split of the data from the weakly_supervised data.

    This ensures that no training samples are used for validation and testing.

    Args:
        config (DictConfig):
            The configuration.

    Returns:
        Pandas DataFrame:
            The training data.
    """
    # Load the weakly supervised data
    train = load_weakly_supervised_data(config)[["text", "label"]]

    # Load the annotated data
    annotated = load_annotated_data(config)

    # Remove the annotated samples from train
    train = train[~train.text.isin(annotated.text)]

    # Save the training data
    train_path = Path(config.data.train.dir) / config.data.train.fname
    train.to_parquet(train_path)

    # Return the filtered training data
    return train


if __name__ == "__main__":
    create_train_split()

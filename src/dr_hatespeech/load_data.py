"""Loading of data."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_raw_data(config: DictConfig) -> dict:
    """Loading of raw data.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        dict:
            A dictionary with a key `df`, containing the weakly supervised data, and
            `path`, being a Path object pointing to the location of the data.

    Raises:
        FileNotFoundError:
            If the raw data file does not exist.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.raw_dir)

    # Ensure that the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of CSV files in the data directory
    csv_paths = [
        path
        for path in data_dir.glob("*.csv")
        if (config.testing and path.name.startswith("test_"))
        or (not config.testing and not path.name.startswith("test_"))
    ]

    # If there are no CSV files in the data directory then raise an error
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")

    # Log loading of dataset
    logger.info(f"Loading data from {csv_paths[0]}")

    # Read the CSV file
    cols = ["account", "url", "text", "date", "action"]
    df = pd.read_csv(
        csv_paths[0], encoding="windows-1252", usecols=cols, low_memory=False
    )

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return a dictionary containing both the dataframe and the path to the CSV file
    return dict(df=df, path=csv_paths[0])


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_cleaned_data(config: DictConfig) -> dict:
    """Loading of cleaned data.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        dict:
            A dictionary with a key `df`, containing the weakly supervised data, and
            `path`, being a Path object pointing to the location of the data.

    Raises:
        FileNotFoundError:
            If the cleaned data file does not exist.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.processed_dir)

    # Ensure that the processed data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of parquet files in the processed data directory
    parquet_paths = [
        path
        for path in data_dir.glob("*_cleaned.parquet")
        if (config.testing and path.name.startswith("test_"))
        or (not config.testing and not path.name.startswith("test_"))
    ]

    # If there are no parquet files in the data directory then raise an error
    if len(parquet_paths) == 0:
        raise FileNotFoundError(f"No cleaned data files found in {data_dir}.")

    # Log loading of dataset
    logger.info(f"Loading data from {parquet_paths[0]}")

    # Read the parquet file
    df = pd.read_parquet(parquet_paths[0])

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return a dictionary containing both the dataframe and the path to the parquet file
    return dict(df=df, path=parquet_paths[0])


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_final_data(config: DictConfig) -> dict:
    """Loading of final data, split into a training, validation and test set.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        dict:
            A dictionary with a keys `train`, `val` and `test`, containing the
            training, validation and test data, respectively.

    Raises:
        FileNotFoundError:
            If one of "train.parquet", "val.parquet" or "test.parquet" do not exist in
            the final data directory.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.final_dir)

    # Ensure that the processed data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of training, validation and test parquet files in the final data
    # directory
    train_paths = [
        path
        for path in data_dir.glob("*train.parquet")
        if (config.testing and path.name.startswith("test_"))
        or (not config.testing and not path.name.startswith("test_"))
    ]
    val_paths = [
        path
        for path in data_dir.glob("*val.parquet")
        if (config.testing and path.name.startswith("test_"))
        or (not config.testing and not path.name.startswith("test_"))
    ]
    test_paths = [
        path
        for path in data_dir.glob("*test.parquet")
        if (config.testing and path.name.startswith("test_"))
        or (not config.testing and not path.name.startswith("test_"))
    ]

    # If any of the paths are missing then split the data
    if len(train_paths) == 0:
        raise FileNotFoundError(f"No training data file found in {data_dir}.")
    if len(val_paths) == 0:
        raise FileNotFoundError(f"No validation data file found in {data_dir}.")
    if len(test_paths) == 0:
        raise FileNotFoundError(f"No test data file found in {data_dir}.")

    # Log loading of dataset
    logger.info(
        f"Loading data from {train_paths[0]}, {val_paths[0]} and {test_paths[0]}"
    )

    # Read the parquet files
    train = pd.read_parquet(train_paths[0])[["text", "label"]]
    val = pd.read_parquet(val_paths[0])[["text", "label"]]
    test = pd.read_parquet(test_paths[0])[["text", "label"]]

    # Remove the val/test samples from train
    train = train[~train.text.isin(val.text)]
    train = train[~train.text.isin(test.text)]

    # Log the number of rows in the dataframe
    logger.info(
        f"Loaded {len(train):,}, {len(val):,} and {len(test):,} rows from the "
        "training, validation and test sets, respectively"
    )

    # Return a dictionary containing the training, validation and test data
    return dict(train=train, val=val, test=test)

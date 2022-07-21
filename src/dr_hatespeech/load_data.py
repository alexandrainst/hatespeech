"""Loading of data."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_raw_data(config: DictConfig) -> pd.DataFrame:
    """Loading of raw data.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        Pandas DataFrame:
            The raw data.

    Raises:
        FileNotFoundError:
            If the raw data file does not exist.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.raw.dir)

    # Ensure that the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the data file
    if config.testing:
        csv_path = data_dir / config.data.raw.test_fname
    else:
        csv_path = data_dir / config.data.raw.fname

    # If the CSV file was not found in the data directory then raise an error
    if not csv_path.exists():
        raise FileNotFoundError(
            f"The file {csv_path.name} was not  found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {csv_path}")

    # Read the CSV file
    cols = ["account", "url", "text", "date", "action"]
    df = pd.read_csv(csv_path, encoding="windows-1252", usecols=cols, low_memory=False)

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return the dataframe
    return df


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_cleaned_data(config: DictConfig) -> pd.DataFrame:
    """Loading of cleaned data.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        Pandas DataFrame:
            The cleaned data.

    Raises:
        FileNotFoundError:
            If the cleaned data file does not exist.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.cleaned.dir)

    # Ensure that the processed data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the data file
    if config.testing:
        parquet_path = data_dir / config.data.cleaned.test_fname
    else:
        parquet_path = data_dir / config.data.cleaned.fname

    # If there are no parquet files in the data directory then raise an error
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"The file {parquet_path.name} was not found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {parquet_path}")

    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return the dataframe
    return df


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_weakly_supervised_data(config: DictConfig) -> pd.DataFrame:
    """Loading of weakly supervised data.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        Pandas DataFrame:
            The weakly supervised data.

    Raises:
        FileNotFoundError:
            If the weakly supervised data file does not exist.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.weakly_supervised.dir)

    # Ensure that the processed data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the data file
    if config.testing:
        parquet_path = data_dir / config.data.weakly_supervised.test_fname
    else:
        parquet_path = data_dir / config.data.weakly_supervised.fname

    # If there are no parquet files in the data directory then raise an error
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"The file {parquet_path.name} was not found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {parquet_path}")

    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return the dataframe
    return df


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_annotated_data(config: DictConfig) -> dict:
    """Loading of annotated data, split into a training, validation and test set.

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
    data_dir = Path(config.data.annotated.dir)

    # Ensure that the processed data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the training, validation and test parquet file paths
    if config.testing:
        train_path = data_dir / config.data.annotated.train.test_fname
        val_path = data_dir / config.data.annotated.val.test_fname
        test_path = data_dir / config.data.annotated.test.test_fname
    else:
        train_path = data_dir / config.data.annotated.train.fname
        val_path = data_dir / config.data.annotated.val.fname
        test_path = data_dir / config.data.annotated.test.fname

    # If any of the paths are missing then split the data
    if not train_path.exists():
        raise FileNotFoundError(
            f"The file {train_path.name} was not found in {data_dir}."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"The file {val_path.name} was not found in {data_dir}."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"The file {test_path.name} was not found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {train_path}, {val_path} and {test_path}")

    # Read the parquet files
    train = pd.read_parquet(train_path)[["text", "label"]]
    val = pd.read_parquet(val_path)[["text", "label"]]
    test = pd.read_parquet(test_path)[["text", "label"]]

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

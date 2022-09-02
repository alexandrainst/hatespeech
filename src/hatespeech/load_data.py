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
            If the file is not found.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.raw.dir)

    # Ensure that the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the data file
    csv_path = data_dir / config.data.raw.fname

    # If the CSV file was not found in the data directory then raise an error
    if not csv_path.exists():
        raise FileNotFoundError(
            f"The file {csv_path.name} was not found in {data_dir}."
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
            If the file is not found.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.cleaned.dir)

    # Ensure that the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the data file
    parquet_path = data_dir / config.data.cleaned.fname

    # If there are no parquet files in the data directory then raise an error
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"The file {parquet_path.name} was not found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {parquet_path}")

    # Read the parquet file
    try:
        df = pd.read_parquet(parquet_path, engine="fastparquet")
    except TypeError:
        df = pd.read_parquet(parquet_path, engine="pyarrow")

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
            If the file is not found.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.weakly_supervised.dir)

    # Ensure that the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the path to the data file
    fname = config.data.weakly_supervised.fname.format(config.label_model.name)
    parquet_path = data_dir / fname

    # If there are no parquet files in the data directory then raise an error
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"The file {parquet_path.name} was not found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {parquet_path}")

    # Read the parquet file
    try:
        df = pd.read_parquet(parquet_path, engine="fastparquet")
    except TypeError:
        df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return the dataframe
    return df


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_annotated_data(config: DictConfig) -> pd.DataFrame:
    """Loading of annotated data.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        Pandas DataFrame:
            The annotated data.

    Raises:
        FileNotFoundError:
            If the file is not found.
    """
    # Set up the path to the data directory
    data_dir = Path(config.data.annotated.dir)

    # Ensure that the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get the parquet file path
    parquet_path = data_dir / config.data.annotated.fname

    # If the path is missing then raise an error
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"The file {parquet_path.name} was not found in {data_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {parquet_path}")

    # Read the parquet files
    try:
        df = pd.read_parquet(parquet_path, engine="fastparquet")
    except TypeError:
        df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Log the number of rows in the dataframe
    logger.info(f"Loaded {len(df):,} rows")

    # Return the dataframe
    return df


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def load_splits(config: DictConfig) -> dict:
    """Loading of training splits, split into a training, validation and test set.

    Args:
        config (DictConfig):
            Configuration object.

    Returns:
        dict:
            A dictionary with a keys `train`, `val` and `test`, containing the
            training, validation and test data, respectively.

    Raises:
        FileNotFoundError:
            If one of the files was not found.
    """
    # Set up the paths to the data directories
    train_dir = Path(config.data.train.dir)
    val_dir = Path(config.data.val.dir)
    test_dir = Path(config.data.test.dir)

    # Ensure that the data directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get the training, validation and test parquet file paths
    train_path = train_dir / config.data.train.fname.format(config.label_model.name)
    val_path = val_dir / config.data.val.fname
    test_path = test_dir / config.data.test.fname

    # If any of the paths are missing then split the data
    if not train_path.exists():
        raise FileNotFoundError(
            f"The file {train_path.name} was not found in {train_dir}."
        )
    if not val_path.exists():
        raise FileNotFoundError(f"The file {val_path.name} was not found in {val_dir}.")
    if not test_path.exists():
        raise FileNotFoundError(
            f"The file {test_path.name} was not found in {test_dir}."
        )

    # Log loading of dataset
    logger.info(f"Loading data from {train_path}, {val_path} and {test_path}")

    # Read the parquet files
    try:
        train = pd.read_parquet(train_path, engine="fastparquet")[["text", "label"]]
        val = pd.read_parquet(val_path, engine="fastparquet")[["text", "label"]]
        test = pd.read_parquet(test_path, engine="fastparquet")[["text", "label"]]
    except TypeError:
        train = pd.read_parquet(train_path, engine="pyarrow")[["text", "label"]]
        val = pd.read_parquet(val_path, engine="pyarrow")[["text", "label"]]
        test = pd.read_parquet(test_path, engine="pyarrow")[["text", "label"]]

    # Log the number of rows in the dataframe
    logger.info(
        f"Loaded {len(train):,}, {len(val):,} and {len(test):,} rows from the "
        "training, validation and test sets, respectively"
    )

    # Return a dictionary containing the training, validation and test data
    return dict(train=train, val=val, test=test)

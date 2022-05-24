"""Functions related to the processing and loading of data."""

import pandas as pd
from pathlib import Path
import re
from unicodedata import normalize
from typing import Union
from tqdm.auto import tqdm


# Enable `progress_apply` method for DataFrame objects
tqdm.pandas()


def clean_account(account: str) -> str:
    """Clean an account.

    Args:
        account (str):
            The account to clean.

    Returns:
        str:
            The cleaned account.
    """
    # Split the account into its parts
    parts = account.split(": ")

    # Remove the parts that we don't care about
    bad_parts = [
        "page",
        "page activity",
        "published page activity",
        "unpublished page activity",
        "videos",
        "wall posts",
        "mentions",
        "private messages",
    ]
    parts = [part for part in parts if part not in bad_parts]

    # Fix bad words
    word_mapping = {
        "?stjylland": "østjylland",
        "k?benhavn": "københavn",
        "sj?lland": "sjælland",
    }
    for idx, part in enumerate(parts):
        for key, val in word_mapping.items():
            part = part.replace(key, val)
        parts[idx] = part

    # Remove the parentheses
    parts = [re.sub(r"\(.*\)", "", part).strip() for part in parts]

    return " ".join(parts)


def clean_text(text: str) -> Union[str, None]:
    """Clean a Facebook post.

    This will NFKC normalize the text, remove unwanted symbols, replace
    hyperlinks with '[LINK]', replace phone numbers with '[PHONE]',
    replace CPR-numbers with '[CPR]', replace mail adresses with '[EMAIL]',
    and remove superfluous whitespace.

    Args:
        text (str):
            The text to clean.

    Returns:
        str or None:
            The cleaned text. If the cleaned text is empty or only consists of
            a hyperlink then None is returned.
    """
    # Normalize the text
    text = normalize("NFKC", text)

    # Remove the \x1a character
    text = re.sub("\x1a", "", text)

    # Replace newlines with spaces
    text = re.sub("\n", " ", text)

    # Replace hyperlinks with " [LINK] "
    text = re.sub(r"http[.\/?&a-zA-Z0-9\-\:\=\%\_\;]+", " [LINK] ", text)

    # Replace 8 digits with " [CVR] " if "cvr" is in the text, else replace with " [PHONE] "
    # Check if an 8 digit number is present in text
    if re.search(r"(?<!\d)\d{8}(?!\d)", text):

        # Check if 'cvr' in text
        if "cvr" in text.lower():
            text = re.sub(r"(?<!\d)\d{8}(?!\d)", " [CVR] ", text)
            
        # Assume 8 digits is a phone number if 'cvr' not in text.
        else:
            text = re.sub(r"(?<!\d)\d{8}(?!\d)", " [PHONE] ", text)

    # Replace CPR with " [CPR] "
    text = re.sub(
        r"(?<![\w.+-])(0[1-9]|[1-2]\d|30|31)(0\d|1[0-2])\d{2}-?\d{4}\b",
        " [CPR] ",
        text,
    )

    # Replace telephone number with international prefix, limited to Europe with " [PHONE] "
    text = re.sub(
        r"(?<![\w.+-])(?:[+][34]\d{1,2}[ .-]?)(?:[(]\d{1,3}[)][ .-]?)?(?:\d[ .-]?){8,13}\b",
        " [PHONE] ",
        text,
    )

    # E-mail
    text = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " [EMAIL] ", text)

    # Remove duplicate whitespace
    text = re.sub(" +", " ", text)

    # Strip trailing whitespace
    text = text.strip()

    # Define replacement strings
    replacement_string = ["[LINK]", "[PHONE]", "[CPR]", "[EMAIL]"]

    # Return None if the text is empty or if only contains a replacement string
    if len(text) == 0 or any(
        [text == replacement for replacement in replacement_string]
    ):
        return None

    return text


def process_data(data_dir: Union[str, Path] = "data", test: bool = False):
    """Process the raw data and store the processed data.

    Args:
        data_dir (str or Path, optional):
            The path to the data directory. Defaults to 'data'.
        test (bool, optional):
            Whether to process the test data. Defaults to False.
    """
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Ensure that `data_dir` exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create the path to the raw data directory
    raw_dir = data_dir / "raw"

    # Ensure that the raw data directory exists
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create the path to the processed data directory
    processed_dir = data_dir / "processed"

    # Ensure that the processed data directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to the raw data file
    if test:
        raw_paths = [
            path for path in raw_dir.glob("*.csv") if path.name.startswith("test_")
        ]
    else:
        raw_paths = [
            path for path in raw_dir.glob("*.csv") if not path.name.startswith("test_")
        ]

    # Read the CSV file
    cols = ["account", "text", "date", "action"]
    df = pd.read_csv(
        raw_paths[0], encoding="windows-1252", usecols=cols, low_memory=False
    )

    # Replace the NaN values in `action` by 'none'
    df.action.fillna(value="none", inplace=True)

    # Cast `date` column as datetime
    df.date = pd.to_datetime(df.date)

    # Remove NaN values
    df.dropna(inplace=True)

    # Clean the `text` column
    df.text = df.text.progress_apply(clean_text)

    # Clean the `account` column
    df.account = df.account.progress_apply(clean_account)

    # Remove NaN values again
    df.dropna(inplace=True)

    # Cast `account` and `action` columns as categories
    df = df.astype(dict(account="category", action="category"))

    # Save the dataframe as a parquet file
    processed_path = processed_dir / f"{raw_paths[0].stem}_processed.parquet"
    df.to_parquet(processed_path)


def load_data(data_dir: Union[str, Path] = "data", test: bool = False) -> pd.DataFrame:
    """Load the processed data.

    If the data has not been processed then first process it.

    Args:
        data_dir (str or Path, optional):
            The path to the data directory. Defaults to 'data'.
        test (bool, optional):
            Whether to load the test data. Defaults to False.

    Returns:
        pd.DataFrame:
            The processed data.
    """
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Ensure that `data_dir` exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create the path to the processed data directory
    processed_dir = data_dir / "processed"

    # Ensure that the processed data directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of parquet files in the processed data directory
    if test:
        parquet_paths = [
            path
            for path in processed_dir.glob("*.parquet")
            if path.name.startswith("test_")
        ]
    else:
        parquet_paths = [
            path
            for path in processed_dir.glob("*.parquet")
            if not path.name.startswith("test_")
        ]

    # If there are no parquet files then process the data
    if len(parquet_paths) == 0:
        process_data(data_dir=data_dir, test=test)
        if test:
            parquet_paths = [
                path
                for path in processed_dir.glob("*.parquet")
                if path.name.startswith("test_")
            ]
        else:
            parquet_paths = [
                path
                for path in processed_dir.glob("*.parquet")
                if not path.name.startswith("test_")
            ]

    # Read the parquet file
    df = pd.read_parquet(parquet_paths[0])

    return df

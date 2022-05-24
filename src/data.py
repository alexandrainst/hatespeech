"""Functions related to the processing and loading of data."""

import pandas as pd
from pathlib import Path
import re
from unicodedata import normalize
from typing import Union, Optional
from tqdm.auto import tqdm
import logging


# Set up logging
fmt = "%(asctime)s [%(levelname)s] <%(name)s> %(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


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
    replace CVR-numbers with '[CVR]' and remove superfluous whitespace.

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

    # Replace 8 digits with " [CVR] " if "cvr" is in the text, else replace with " [PHONE] "
    # Check if an 8 digit number is present in text
    if re.search(r"(?<!\d)(\d\d ?){4}(?!\d)", text):

        # Check if 'cvr' in text
        if "cvr" in text.lower():
            text = re.sub(r"(?<!\d)(\d\d ?){4}(?!\d)", " [CVR] ", text)

        # Assume 8 digits is a phone number if 'cvr' not in text.
        else:
            text = re.sub(r"(?<!\d)(\d\d ?){4}(?!\d)", " [PHONE] ", text)

    # Replace CPR with " [CPR] "
    text = re.sub(
        r"(?<![\w.+-])(0[1-9]|[1-2]\d|30|31)(0\d|1[0-2])\d{2}-?\d{4}\b",
        " [CPR] ",
        text,
    )

    # Replace telephone number with international prefix with " [PHONE] "
    text = re.sub(
        r"[\w.+-]\+[2-4]\d{1,2}[ .-]?([(]\d{1,3}[)][ .-]?)?(\d[ .-]?){8,13}\b",
        " [PHONE] ",
        text,
    )

    # Replace hyperlinks with " [LINK] "
    text = re.sub(
        r"(http|www\.)[.\/?&a-zæøåA-ZÆØÅ0-9\-\:\=\%\_\;\$\~\#\[\]\(\)\{\}\,\+\@]+",
        " [LINK] ",
        text,
    )

    # E-mail
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " [EMAIL] ", text)

    # Remove duplicate whitespace
    text = re.sub(" +", " ", text)

    # Strip trailing whitespace
    text = text.strip()

    # Define replacement strings
    replacement_string = ["[LINK]", "[PHONE]", "[CVR]", "[CPR]", "[EMAIL]"]

    # Return None if the text is empty or if only contains a replacement string
    if len(text) == 0 or any(
        [text == replacement for replacement in replacement_string]
    ):
        return None

    return text


def get_post_id(url: Optional[str]) -> Union[int, None]:
    """Extracts the post ID from the URL.

    Args:
        url (str or None):
            The URL of the post.

    Returns:
        int:
            The post ID if the URL is not None, otherwise None.
    """
    if url is None or not isinstance(url, str):
        return None

    # Extract the URL parts
    parts = url.split("/")

    # Case 1: We extract the post ID through the "fbid" GET parameter. If the
    # URL has no GET parameters then return None.
    if len(parts) == 4:
        get_args_list = [
            get_arg.split("=") for get_arg in url.split("?")[-1].split("&")
        ]
        if all(len(lst) == 2 for lst in get_args_list):
            get_args = {key: val for key, val in get_args_list}
            return int(get_args["fbid"])
        else:
            return None

    # Case 2: The post ID is the last number before the GET parameters
    else:
        core_url = re.split(r"/?\?", url)[0]
        post_id = [part for part in core_url.split("/") if part != ""][-1]
        return int(post_id)


def get_comment_id(url: Optional[str]) -> Union[int, None]:
    """Extracts the comment ID from the URL.

    Args:
        url (str or None):
            The URL of the comment.

    Returns:
        int:
            The comment ID if the post is a comment and the URL is not None,
            otherwise None.
    """
    if url is None or not isinstance(url, str) or "comment_id" not in url:
        return None
    else:
        matches = re.search(r"(?<=comment_id=)\d+", url)
        if matches is None:
            return None
        else:
            return int(matches[0])


def get_reply_comment_id(url: Optional[str]) -> Union[int, None]:
    """Extracts the reply comment ID from the URL.

    Args:
        url (str or None):
            The URL of the comment.

    Returns:
        int:
            The comment ID if the post is a reply and the URL is not None,
            otherwise None.
    """
    if url is None or not isinstance(url, str) or "reply_comment_id" not in url:
        return None
    else:
        matches = re.search(r"(?<=reply_comment_id=)\d+", url)
        if matches is None:
            return None
        else:
            return int(matches[0])


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
    logger.info(f"Loading data from {raw_paths[0]}")
    cols = ["account", "url", "text", "date", "action"]
    df = pd.read_csv(
        raw_paths[0], encoding="windows-1252", usecols=cols, low_memory=False
    )
    logger.info(f"Loaded {len(df):,} rows")

    # Replace the NaN values in `action` by 'none'
    df.action.fillna(value="none", inplace=True)

    # Cast `date` column as datetime
    df.date = pd.to_datetime(df.date)

    # Remove NaN values from the `text` and `account` columns
    num_rows = len(df)
    df.dropna(subset=["text", "account"], inplace=True)
    logger.info(f"Removed {num_rows - len(df):,} rows with NaN values")

    # Clean the `text` column
    tqdm.pandas(desc="Cleaning text")
    df.text = df.text.progress_apply(clean_text)

    # Clean the `account` column
    tqdm.pandas(desc="Cleaning account")
    df.account = df.account.progress_apply(clean_account)

    # Remove NaN values again from the `text` and `account` columns
    num_rows = len(df)
    df.dropna(subset=["text", "account"], inplace=True)
    logger.info(f"Removed {num_rows - len(df):,} rows with NaN values")

    # Extract post_id from the url
    tqdm.pandas(desc="Extracting post_id")
    df["post_id"] = df.url.progress_apply(get_post_id)

    # Extract comment_id from the url
    tqdm.pandas(desc="Extracting comment_id")
    df["comment_id"] = df.url.progress_apply(get_comment_id)

    # Extract reply_comment_id from the url
    tqdm.pandas(desc="Extracting reply_comment_id")
    df["reply_comment_id"] = df.url.progress_apply(get_reply_comment_id)

    # Remove duplicates
    num_rows = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    logger.info(f"Removed {num_rows - len(df):,} duplicates")

    # Cast `account` and `action` columns as categories, and the ID columns as
    # nullable integers
    df = df.astype(
        dict(
            account="category",
            action="category",
            post_id="Int64",
            comment_id="Int64",
            reply_comment_id="Int64",
        )
    )

    # Save the dataframe as a parquet file
    processed_path = processed_dir / f"{raw_paths[0].stem}_cleaned.parquet"
    df.to_parquet(processed_path)
    logger.info(f"Saved processed data with {len(df):,} rows to {processed_path}")


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

    # Create the path to the processed data directory
    processed_dir = data_dir / "processed"

    # Ensure that the processed data directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of parquet files in the processed data directory
    if test:
        parquet_paths = [
            path
            for path in processed_dir.glob("*_cleaned.parquet")
            if path.name.startswith("test_")
        ]
    else:
        parquet_paths = [
            path
            for path in processed_dir.glob("*_cleaned.parquet")
            if not path.name.startswith("test_")
        ]

    # If there are no parquet files then process the data
    if len(parquet_paths) == 0:
        process_data(data_dir=data_dir, test=test)
        if test:
            parquet_paths = [
                path
                for path in processed_dir.glob("*_cleaned.parquet")
                if path.name.startswith("test_")
            ]
        else:
            parquet_paths = [
                path
                for path in processed_dir.glob("*_cleaned.parquet")
                if not path.name.startswith("test_")
            ]

    # Read the parquet file
    df = pd.read_parquet(parquet_paths[0])

    return df


if __name__ == "__main__":
    load_data()

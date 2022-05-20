'''Functions related to the processing and loading of data.'''

import pandas as pd
from pathlib import Path
import re
from unicodedata import normalize
from typing import Union
from tqdm.auto import tqdm


# Enable `progress_apply` method for DataFrame objects
tqdm.pandas()


def clean_account(account: str) -> str:
    '''Clean an account.

    Args:
        account (str):
            The account to clean.

    Returns:
        str:
            The cleaned account.
    '''
    # Split the account into its parts
    parts = account.split(': ')

    # Remove the parts that we don't care about
    bad_parts = [
        'page',
        'page activity',
        'published page activity',
        'unpublished page activity',
        'videos',
        'wall posts',
        'mentions',
        'private messages',
    ]
    parts = [part for part in parts if part not in bad_parts]

    # Fix bad words
    word_mapping = {
        '?stjylland': 'østjylland',
        'k?benhavn': 'københavn',
        'sj?lland': 'sjælland'
    }
    for idx, part in enumerate(parts):
        for key, val in word_mapping.items():
            part = part.replace(key, val)
        parts[idx] = part

    # Remove the parentheses
    parts = [re.sub(r'\(.*\)', '', part).strip() for part in parts]

    return ' '.join(parts)


def clean_text(text: str) -> Union[str, None]:
    '''Clean a Facebook post.

    This will NFKC normalize the text, remove unwanted symbols, replace
    hyperlinks with '[LINK]' and remove superfluous whitespace.

    Args:
        text (str):
            The text to clean.

    Returns:
        str or None:
            The cleaned text. If the cleaned text is empty or only consists of
            a hyperlink then None is returned.
    '''
    # Normalize the text
    text = normalize('NFKC', text)

    # Remove the \x1a character
    text = text.replace('\x1a', '')

    # Replace the \x93 and \x94 characters with quotes
    text = text.replace('[\x93\x94]', '"')

    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Replace hyperlinks with [LINK]
    text = re.sub('http[.\/?&a-zA-Z0-9\-\:=]+', '[LINK]', text)

    # Remove duplicate whitespace
    text = text.replace(' +', ' ')

    # Strip trailing whitespace
    text = text.strip()

    # Return None if the text is empty or if only contains a link
    if len(text) == 0 or text == '[LINK]':
        return None

    return text


def process_data(data_dir: Union[str, Path] = 'data'):
    '''Process the raw data and store the processed data.

    Args:
        data_dir (str or Path, optional):
            The path to the data directory. Defaults to 'data'.
    '''
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Ensure that `data_dir` exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create the path to the raw data directory
    raw_dir = data_dir / 'raw'

    # Ensure that the raw data directory exists
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create the path to the processed data directory
    processed_dir = data_dir / 'processed'

    # Ensure that the processed data directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to the raw data file
    raw_path = next(raw_dir.glob('*.csv'))

    # Read the CSV file
    cols = ['account', 'text', 'date', 'action']
    df = pd.read_csv(
        raw_path,
        encoding='latin_1',
        usecols=cols,
        low_memory=False
    )

    # Replace the NaN values in `action` by 'none'
    df.action.fillna(value='none', inplace=True)

    # Cast `date` column as datetime
    df.date = pd.to_datetime(df.date)

    # Remove NaN values
    df.dropna(inplace=True)

    # Clean the `account` column
    df.account = df.account.progress_apply(clean_account)

    # Clean the `text` column
    df.text = df.text.progress_apply(clean_text)

    # Remove NaN values again
    df.dropna(inplace=True)

    # Cast `account` and `action` columns as categories
    df = df.astype(dict(account='category', action='category'))

    # Save the dataframe as a parquet file
    df.to_parquet(processed_dir / f'{raw_path.stem}_processed.parquet')


def load_data(data_dir: Union[str, Path] = 'data') -> pd.DataFrame:
    '''Load the processed data.

    If the data has not been processed then first process it.

    Args:
        data_dir (str or Path, optional):
            The path to the data directory. Defaults to 'data'.

    Returns:
        pd.DataFrame:
            The processed data.
    '''
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Ensure that `data_dir` exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create the path to the processed data directory
    processed_dir = data_dir / 'processed'

    # Ensure that the processed data directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of parquet files in the processed data directory
    parquet_paths = list(processed_dir.glob('*.parquet'))

    # If there are no parquet files then process the data
    if len(parquet_paths) == 0:
        process_data(data_dir=data_dir)
        parquet_paths = list(processed_dir.glob('*.parquet'))

    # Read the parquet file
    df = pd.read_parquet(parquet_paths[0])

    return df

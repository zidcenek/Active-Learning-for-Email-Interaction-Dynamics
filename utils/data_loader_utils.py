import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_FOLDER = Path('data') / 'subsets'

def file_name_for_sender(sender_id: int) -> str:
    return f'sender_mails_{sender_id}.parquet'

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def load_subset_for_sender(sender_id: int, path: Path = DATA_FOLDER) -> pd.DataFrame:
    return load_parquet(path / file_name_for_sender(sender_id))

def load_data_for_sender(sender_id: int) -> pd.DataFrame:
    """Load data for a sender from a parquet file."""
    try:
        return load_subset_for_sender(sender_id)
    except FileNotFoundError as e:
        logger.warning(f"No data found for sender {sender_id}")
        raise e
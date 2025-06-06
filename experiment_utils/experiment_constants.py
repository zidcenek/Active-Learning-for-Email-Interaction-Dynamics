import datetime
from pathlib import Path

from utils.constants import ROOT_FOLDER

EXPERIMENTS_FOLDER = ROOT_FOLDER / "experiments"
TEST_SET_FOLDER = EXPERIMENTS_FOLDER / "test_set"

CONTEXTUAL_BANDIT_FOLDER_NAME = "contextual_bandit"
DDQN_FOLDER_NAME = "ddqn"
TWITTER_RL_FOLDER_NAME = "twitter_rl"
RANDOM_FOLDER_NAME = "random"
THOMPSON_SAMPLING_FOLDER_NAME = "thompson_sampling"
FMRSAL_FOLDER_NAME = "fmrsal"
FMPSAL_FOLDER_NAME = "fmpsal"
MODEL_BASED_RL_FOLDER_NAME = "model_based_rl"
FMFCDB_FOLDER_NAME = "fmfcdb"
FACTOR_UCB_FOLDER_NAME = "factor_ucb"

EXPERIMENT_RESULTS_FOLDER_NAME = "exp"

EXPERIMENTS_FOLDER.mkdir(parents=True, exist_ok=True)

AUTOENCODER_CONFIG_NAME = "autoencoder_config.json"
CONTEXTUAL_BANDIT_CONFIG_NAME = "contextual_bandit_config.json"

GRID_SEARCH_FILE_NAME = "grid_search_config.json"

def get_experiment_folder_path(sender_id: int, model_name: str, version: str) -> Path:
    return EXPERIMENTS_FOLDER / str(sender_id) / model_name / version

def get_test_folder_path(sender_id: int, model_name: str, version: str) -> Path:
    if version is None:
        return Path()
    return TEST_SET_FOLDER / str(sender_id) / model_name / version

def generate_timestamp_version() -> str:
    """Generate a version string in the format '20250515-004928' using current timestamp."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

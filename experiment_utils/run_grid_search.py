import re
import sys
from pathlib import Path
import datetime

# TODO: change the ROOT_FOLDER_NAME
ROOT_FOLDER = Path(re.sub(r"(.*?/YOUR_PROJECT_ROOT_FOLDER/).*", r"\1", __file__))
sys.path.append(str(ROOT_FOLDER))

import argparse
import json
import logging
from typing import Dict, Any

from ml.baselines.fmfcdb import FMFCDBModel
from ml.baselines.fmpsal import FMPSALModel
from ml.baselines.model_based_rl import ModelBasedRL
from ml.baselines.fmrsal import FMRSALModel
from ml.baselines.random_model import RandomModel
from ml.baselines.ddqn_linkedin import DoubleDQNTrainer
from ml.baselines.factor_ucb import FactorUCBModel
from ml.baselines.thompson_sampling import ThompsonSamplingModel
from ml.shallow_autoencoder.contextual_bandit_with_autoencoder import ContextualBanditWithAutoencoder
from experiment_utils.experiment_constants import get_experiment_folder_path, GRID_SEARCH_FILE_NAME

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_model_class(model_name: str):
    """Get the model class for the specified model name."""
    model_classes = {
        "fmrsal": FMRSALModel,
        "fmpsal": FMPSALModel,
        "fmfcdb": FMFCDBModel,
        "ddqn": DoubleDQNTrainer,
        "thompson_sampling": ThompsonSamplingModel,
        "random": RandomModel,
        "contextual_bandit": ContextualBanditWithAutoencoder,
        "model_based_rl": ModelBasedRL,
        "factor_ucb": FactorUCBModel,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_classes.keys())}")

    return model_classes[model_name]


def load_config(config_folder: Path, config_file: str = GRID_SEARCH_FILE_NAME) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = config_folder / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["param_grid"]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration."""
    # Validate param_grid
    if not isinstance(config["param_grid"], dict):
        raise ValueError("param_grid must be a dictionary")

    if not config["param_grid"]:
        raise ValueError("param_grid cannot be empty")

    # Validate split_sizes
    split_sizes = config.get("split_sizes", [5, 10])
    if not isinstance(split_sizes, list) or len(split_sizes) != 2:
        raise ValueError("split_sizes must be a list of exactly 2 integers")

    # Validate optional numeric parameters
    for param in ["n_samples", "n_jobs", "random_state"]:
        if param in config and not isinstance(config[param], (int, type(None))):
            raise ValueError(f"{param} must be an integer or null")


def generate_timestamp_version() -> str:
    """Generate a version string in the format '20250515-004928' using current timestamp."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def save_config_to_experiment_folder(sender_id: int, model_name: str, version: str, config: Dict[str, Any]) -> None:
    """Save the configuration to the experiment folder as grid_search_config.json."""
    experiment_folder = get_experiment_folder_path(sender_id, model_name, version)

    # Create the folder if it doesn't exist
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Save config to the experiment folder
    config_path = experiment_folder / "grid_search_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    logger.info(f"Configuration saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search based on a configuration file"
    )
    parser.add_argument("--sender_id", type=int, help="Sender ID (overrides config file if provided)")
    parser.add_argument("--model", type=str, help="Model name (overrides config file if provided)")
    parser.add_argument("--version", type=str, help="Experiment version (overrides config file if provided)")
    parser.add_argument("--split_sizes", type=int, nargs=2, default=[5, 10], help="Split sizes for validation and test sets")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use for grid search")

    args = parser.parse_args()

    # Load configuration from file
    sender_id = args.sender_id
    split_sizes = args.split_sizes
    model_name = args.model
    version = args.version
    n_jobs = args.n_jobs
    n_samples = args.n_samples

    # Set and override parameters as needed
    if sender_id is None:
        raise ValueError("--sender_id is required")
    if model_name is None:
        raise ValueError("--model is required")
    if n_jobs is None:
        n_jobs = 1
    if version is None:
        # Auto-generate the timestamp-based version if not provided
        logger.info(f"Using auto-generated timestamp version: {version}")

    model_class = get_model_class(model_name)
    results_folder = get_experiment_folder_path(sender_id, model_class.model_name(), version)
    config: Dict[str, Any] = load_config(config_folder=results_folder)

    # Extract parameters from config
    param_grid = config["param_grid"]
    random_state = config.get("random_state", 42)

    logger.info(f"Starting grid search for model: {model_name}")
    logger.info(f"Sender ID: {sender_id}")
    logger.info(f"Version: {version}")
    logger.info(f"Split sizes: {split_sizes}")
    logger.info(f"N samples: {n_samples}")
    logger.info(f"N jobs: {n_jobs}")
    logger.info(f"Parameter grid: {param_grid}")

    # Create a complete config for saving
    complete_config = {
        "sender_id": sender_id,
        "model": model_name,
        "version": version,
        "split_sizes": split_sizes,
        "param_grid": param_grid,
        "n_samples": n_samples,
        "n_jobs": n_jobs,
        "random_state": random_state
    }

    # Save configuration to an experiment folder
    save_config_to_experiment_folder(
        sender_id=sender_id,
        model_name=model_name,
        version=version,
        config=complete_config
    )

    # Run grid search
    model_class.grid_search(
        sender_id=sender_id,
        split_sizes=split_sizes,
        version=version,
        param_grid=param_grid,
        n_samples=n_samples,
        n_jobs=n_jobs,
        random_state=random_state
    )

    # Log results location
    logger.info("Grid search completed!")
    logger.info(f"Results saved to: {results_folder}")


if __name__ == '__main__':
    main()

"""
Configuration generator for grid search experiments.
This tool creates example parameter configurations for different models.
"""
import re
import sys
from pathlib import Path
import argparse
import json
import logging
from typing import Dict, Any

ROOT_FOLDER = Path(re.sub(r"(.*?/YOUR_PROJECT_ROOT_FOLDER/).*", r"\1", __file__))
sys.path.append(str(ROOT_FOLDER))

from experiment_utils.experiment_constants import get_experiment_folder_path, GRID_SEARCH_FILE_NAME, \
    generate_timestamp_version

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def create_example_config(model_name: str, sender_id: int, output_file_name: str = GRID_SEARCH_FILE_NAME) -> Dict[str, Any]:
    """Create an example parameter grid configuration for the specified model.
    If output_path is provided, saves the configuration to that path.
    Returns the created configuration dictionary.
    """
    # Get the model class to ensure it's valid

    # Create a basic configuration with the provided parameters
    config = dict()
    config["param_grid"] = {}

    # Add model-specific parameter grid
    if model_name == "fmrsal":
        config["param_grid"] = {
            "latent_dim": [16, 32, 64],
            "lr": [1e-3, 5e-3, 1e-2],
            "epochs": [20, 30, 40],
            "repetitions": [1],
            "subsample_ratio": [0.3, 0.5, 0.7],
            "exploration_ratio": [0.05, 0.10, 0.15],
            "wait_hours": [12, 24, 48]
        }
    elif model_name == "fmpsal":
        config["param_grid"] = {
            "latent_dim": [16, 32, 64],
            "lr": [1e-3, 5e-3, 1e-2],
            "epochs": [20, 30, 40],
            "repetitions": [1],
            "subsample_ratio": [0.3, 0.5, 0.7],
            "top_k_ratio": [0.03, 0.07, 0.15],
            "wait_hours": [48],
        }
    elif model_name == "ddqn":
        config["param_grid"] = {
            "epochs": [10, 20],
            "user_emb_dim": [32, 64],
            "mail_emb_dim": [8, 16],
            "hidden_dim": [32, 64],
            "lr": [1e-4, 3e-4, 1e-3],
            "wd": [1e-5, 1e-4],
            "gamma": [0.25, 0.4, 0.7, 0.9],
            "batch_size": [1024, 2048]
        }
    elif model_name == "contextual_bandit":
        config["param_grid"] = {
            "d": [8, 12, 16],
            "epochs": [10, 20, 30],
            "lr": [1e-4, 1e-3, 1e-2],
            "wd": [1e-4, 1e-5],
            "batch_size": [32, 64, 128],
            "G": [1, 10, 100, 10000],
            "T": [1, 3, 6, 12, 24, 48],
            "sent_by_T": [0.03, 0.08, 0.15],
            "num_splits": [6, 12, 24, 48],
            "s_alpha": [0.1, 0.2, 0.3],
        }
    elif model_name == "thompson_sampling":
        config["param_grid"] = {
            "repetitions": [1]
        }
    elif model_name == "random":
        config["param_grid"] = {
            "seed": [42, 123, 456]
        }
    elif model_name == "model_based_rl":
        config["param_grid"] = {
            "gamma": [0.7, 0.8, 0.9],
            "max_streak": [3, 5, 7],
            "n_value_iter": [30, 50, 70],
            "alpha_prior": [0.5, 1.0, 2.0],
            "beta_prior": [0.5, 1.0, 2.0],
            "kappa": [0, 0.2, 0.4, 0.6, 0.8],
        }
    elif model_name == "fmfcdb":
        config["param_grid"] = {
            'exploration_rate': [0.01, 0.03, 0.05, 0.08],
            'learning_rate': [0.01, 0.05],
            'n_epochs': [40, 60],
            'batch_size': [1000],
            'sample_rate': [0.1, 0.5, 0.7, 1.0],
            'feature_dim': [4, 6],
            'num_batches': [48],
            'observation_hours': [1.0],
            'active_learning_weight': [0.3],
            'early_exploration_boost': [2.0],
            'update_frequency': [1, 5]
        }
    elif model_name == "factor_ucb":
        config["param_grid"] = {
            'latent_dim': [16, 32, 64],
            'reg': [0.001, 0.01, 0.1],
            'epochs': [10, 20, 30],
            'alpha': [0.1, 0.5, 1.0],
            'alpha_conf': [20.0],
        }


    output_folder = get_experiment_folder_path(sender_id, model_name, generate_timestamp_version())
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / output_file_name, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Example configuration saved to: {output_folder / output_file_name}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Create example grid search configuration for a model"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Model name for which to create a configuration")
    parser.add_argument("--sender_id", type=int, required=True,
                       help="Sender ID for the experiment")

    args = parser.parse_args()

    # Create example config with provided parameters
    config = create_example_config(
        model_name=args.model,
        sender_id=args.sender_id,
        output_file_name=GRID_SEARCH_FILE_NAME
    )

    # Save to experiment folder if requested
    version = generate_timestamp_version()

    # Save the config to the experiment folder
    save_config_to_experiment_folder(
        sender_id=args.sender_id,
        model_name=args.model,
        version=version,
        config=config
    )

    logger.info(f"Example configuration created for model: {args.model}")
    logger.info(f"Sender ID: {args.sender_id}")
    logger.info(f"Version: {version}")
    logger.info(f"Configuration saved to experiment folder")


if __name__ == '__main__':
    main()

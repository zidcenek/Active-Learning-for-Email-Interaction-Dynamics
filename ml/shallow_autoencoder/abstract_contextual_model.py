import csv
import gzip
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, TypeVar, Dict, Any, Optional, Self, Type, Callable, Sequence, Tuple
import json
import pandas as pd

import numpy as np
from sklearn.model_selection import ParameterSampler
from joblib import Parallel, delayed

from experiment_utils.experiment_constants import EXPERIMENT_RESULTS_FOLDER_NAME, get_test_folder_path, get_experiment_folder_path
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.abstract_metrics import AbstractMetrics
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

TConfig = TypeVar("TConfig", bound="AbstractConfig")
logger = logging.getLogger(__name__)


@dataclass
class ContextualBanditMetrics(AbstractMetrics):
    present_in_prediction: bool
    user_cluster: int = None
    stage: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mailshot_id": int(self.mailshot_id) if hasattr(self.mailshot_id, "item") else self.mailshot_id,
            "present_in_prediction": bool(self.present_in_prediction),
            "opened": bool(self.opened),
            "prediction": float(self.prediction) if hasattr(self.prediction, "item") else self.prediction,
            "user_cluster": int(self.user_cluster) if self.user_cluster is not None and hasattr(self.user_cluster,
                                                                                                "item") else self.user_cluster,
            "stage": int(self.stage) if hasattr(self.stage, "item") else self.stage
        }

    @staticmethod
    def _to_bool(value: str | int | bool | None) -> bool:
        """Convert various truthy / falsy representations to bool."""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "t", "yes", "y")

    @staticmethod
    def _to_int_or_none(value: str | None) -> Optional[int]:
        """Return an int unless the value is empty / None / 'None'."""
        if value in ("", None, "None"):
            return None
        return int(value)

    @classmethod
    def from_csv_gz(
        cls,
        folder: Path,
        filename: str = "results.csv.gz"
    ) -> List[Self]:
        """
        Read a gz-compressed CSV located in *folder* / *filename* and
        return a list of ContextualBanditMetrics – one per row.

        Expected column names:
            mailshot_id, present_in_prediction, opened, prediction,
            user_cluster (optional), stage (optional)
        """
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(path)

        instances: List[ContextualBanditMetrics] = []

        with gzip.open(path, mode="rt", newline="") as fh:          # text mode
            reader = csv.DictReader(fh)

            # Quick sanity-check
            required = {
                "mailshot_id",
                "opened",
                "prediction",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {', '.join(missing)}"
                )

            # Build objects
            for row in reader:
                instances.append(
                    cls(
                        mailshot_id=int(row["mailshot_id"]),
                        present_in_prediction=cls._to_bool(
                            row["present_in_prediction"]
                        ),
                        opened=cls._to_bool(row["opened"]),
                        prediction=float(row["prediction"]),
                        user_cluster=cls._to_int_or_none(row.get("user_cluster")),
                        stage=int(row["stage"])
                        if row.get("stage") not in ("", None, "None")
                        else -1,
                    )
                )

        return instances
    @classmethod
    def from_lists(cls, mailshot_id: int, present_in_prediction: List[bool], opened: List[bool], prediction: List[float], user_clusters: List[int]) -> List[Self]:
        return [cls(mailshot_id, o, pred, p, cluster) for p, o, pred, cluster in zip(present_in_prediction, opened, prediction, user_clusters)]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List[Self]:
        return [cls(row.mailshot_id, row.opened, row.prediction, row.present_in_prediction, row.user_cluster) for _, row in df.iterrows()]


@dataclass
class AbstractConfig(ABC):

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        pass

    def to_json_file(self, folder: Path, filename: str = "config.json") -> None:
        with open(folder / filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


class AbstractContextualModel(ABC):

    def __init__(self, config: TConfig):
        self.config = config
        self._train = None
        self._val = None

    @property
    def train(self) -> AutoencoderDataset:
        if self._train is None:
            raise ValueError("Train dataset not set")
        return self._train

    @property
    def val(self) -> AutoencoderDataset:
        if self._val is None:
            raise ValueError("Validation dataset not set")
        return self._val

    @abstractmethod
    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        self._train = train
        self._val = val

    @abstractmethod
    def predict(self, data: AutoencoderDataset) -> List[ContextualBanditMetrics]:
        pass

    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        pass

    @classmethod
    def config_class(cls) -> Type[AbstractConfig]:
        return AbstractConfig

    @classmethod
    def metrics_class(cls) -> Type[AbstractMetrics]:
        return DefaultMetrics

    @classmethod
    def test(cls, sender_id: int, split_sizes: List[int], repetitions: int, version: str) -> None:
        if len(split_sizes) != 1:
            raise ValueError("Only one split size is allowed for this experiment.")

        split_size = split_sizes[0]
        experiments_folder = get_test_folder_path(
            sender_id=sender_id,
            model_name=cls.model_name(),
            version=version
        )
        config: AbstractConfig = cls.config_class().from_json_file(experiments_folder)
        for repetition in range(repetitions):
            for i, current_split_size in enumerate(range(split_size, 0, -1)):
                logger.info(f"Experiment {i + 1}/{split_size} - Repetition {repetition + 1}/{repetitions}")
                # Load datasets using the provided sender_id and split configuration
                # Always have size 1 for the test set, the last split is not used
                datasets = AutoencoderDataset.from_disk_data_split(
                    sender_id, split_sizes=[1, current_split_size - 1], remove_users_below_n_opens=1
                )
                train, test = datasets[0], datasets[1]
                model = cls(config=config)
                model.fit(train, test)
                metrics: List[AbstractMetrics] = model.predict(test)

                # Create a folder for the current split size and repetition
                current_folder = experiments_folder / (EXPERIMENT_RESULTS_FOLDER_NAME + f"_{i + 1}_{repetition}")
                current_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Current folder: {current_folder}. Saving results...")
                # Save the prediction metrics
                ContextualBanditMetrics.to_csv_gz(metrics, folder=current_folder)
                config.to_json_file(folder=current_folder)

    @classmethod
    def grid_search(
        cls,
        sender_id: int,
        split_sizes: List[int],
        version: str,
        param_grid: Dict[str, List[Any]],
        n_samples: Optional[int] = None,
        n_jobs: int = 8,
        scoring_function: Optional[Callable[[List[ContextualBanditMetrics]], float]] = None,
        random_state: int = 42
    ) -> None:
        """
        Perform grid search over hyperparameters for the model.
        
        @param sender_id: Sender ID for the dataset
        @param split_sizes: How to split the dataset (e.g., [5, 10] for train/val split)
        @param version: Experiment version string
        @param param_grid: Dictionary of parameter names to lists of values to try
        @param n_samples: Number of parameter combinations to sample (if None, try all combinations)
        @param n_jobs: Number of parallel jobs to run
        @param scoring_function: Function to score metrics (if None, uses default recall-based scoring)
        @param random_state: Random seed for parameter sampling
        """
        logger.info(f"Starting grid search for {cls.model_name()} with {len(param_grid)} parameters")
        
        # Create the results folder
        results_folder = get_experiment_folder_path(sender_id, cls.model_name(), version)
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        logger.info(f"Loading dataset for sender_id {sender_id}...")
        datasets = AutoencoderDataset.from_disk_data_split(
            sender_id, split_sizes=split_sizes, remove_users_below_n_opens=1
        )
        train, val = datasets[0], datasets[1]
        logger.info(f"Dataset loaded. Train size: {len(train.mails)}, Val size: {len(val.mails)}")
        
        # Generate parameter combinations
        if n_samples is None:
            # Use all combinations (grid search)
            import itertools
            param_keys = list(param_grid.keys())
            param_combinations = list(itertools.product(*param_grid.values()))
            grid = [dict(zip(param_keys, combo)) for combo in param_combinations]
        else:
            # Use random sampling
            grid = list(ParameterSampler(param_grid, n_iter=n_samples, random_state=random_state))
        
        logger.info(f"Evaluating {len(grid)} parameter combinations...")
        
        # Save experiment metadata
        metadata = {
            "sender_id": sender_id,
            "model_name": cls.model_name(),
            "version": version,
            "split_sizes": split_sizes,
            "param_grid": param_grid,
            "n_samples": n_samples,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "total_combinations": len(grid),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(results_folder / "grid_search_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Define default scoring function if none provided
        if scoring_function is None:
            def default_scoring(metrics: List[ContextualBanditMetrics]) -> float:
                _, recalls, _ = cls.calculate_recall_precision_at_quantiles(metrics)
                if len(recalls) >= 4:
                    return (0.40 * recalls[0] + 0.30 * recalls[1] + 
                           0.20 * recalls[2] + 0.10 * recalls[3])
                elif len(recalls) > 0:
                    return np.mean(recalls)
                else:
                    return 0.0
            scoring_function = default_scoring
        
        # Run a parallel grid search
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(cls._run_single_experiment)(
                cfg_dict, exp_number, train, val, results_folder, scoring_function
            ) for exp_number, cfg_dict in enumerate(grid)
        )
        elapsed_time = time.time() - t0
        logger.info(f"Grid search completed in {elapsed_time:.1f} seconds")
        
        # Process and save results
        cls._save_grid_search_results(results, results_folder, elapsed_time)

    @classmethod
    def _run_single_experiment(
        cls,
        cfg_dict: Dict[str, Any],
        exp_number: int,
        train: AutoencoderDataset,
        val: AutoencoderDataset,
        results_folder: Path,
        scoring_function: Callable[[List[ContextualBanditMetrics]], float]
    ) -> Dict[str, Any]:
        """
        Run a single experiment with a given configuration.
        
        Returns:
            Dictionary containing experiment results
        """
        # Create experiment folder
        logger.info(f"Starting experiment {exp_number}...")
        exp_folder = results_folder / f"exp_{exp_number}"
        exp_folder.mkdir(exist_ok=True, parents=True)
        
        try:
            # Save configuration
            with open(exp_folder / "config.json", "w") as f:
                json.dump(cfg_dict, f, indent=4)
            
            # Create config object and model
            config = cls.config_class()(**cfg_dict)
            model = cls(config=config)

            # Train and predict
            model.fit(train, val)
            metrics = model.predict(val)

            # Save metrics to compressed CSV
            cls.metrics_class().to_csv_gz(metrics, folder=exp_folder)
            
            # Calculate score
            score = scoring_function(metrics)
            
            # Calculate additional metrics
            _, recalls, precisions = cls.calculate_recall_precision_at_quantiles(metrics)
            
            # Save summary results
            results_summary = {
                "experiment_number": exp_number,
                "score": float(score),
                "recalls": [float(r) for r in recalls] if recalls else [],
                "precisions": [float(p) for p in precisions] if precisions else [],
                "config": cfg_dict,
                "n_metrics": len(metrics)
            }
            
            with open(exp_folder / "results_summary.json", "w") as f:
                json.dump(results_summary, f, indent=4)
            
            logger.info(f"Experiment {exp_number} completed with score: {score:.4f}")
            return results_summary
            
        except Exception as e:
            logger.error(f"Experiment {exp_number} failed: {str(e)}")
            error_summary = {
                "experiment_number": exp_number,
                "score": 0.0,
                "recalls": [],
                "precisions": [],
                "config": cfg_dict,
                "error": str(e),
                "n_metrics": 0
            }
            
            with open(exp_folder / "error.json", "w") as f:
                json.dump(error_summary, f, indent=4)
            
            return error_summary

    @classmethod
    def _save_grid_search_results(
        cls,
        results: List[Dict[str, Any]],
        results_folder: Path,
        elapsed_time: float
    ) -> None:
        """
        Save aggregated grid search results.
        """
        # Create DataFrame with all results
        results_data = []
        for result in results:
            row = {
                "experiment": result["experiment_number"],
                "score": result["score"],
                **result["config"]
            }
            
            # Add recall metrics if available
            recalls = result.get("recalls", [])
            for i, recall in enumerate(recalls):
                row[f"recall_{i}"] = recall
            
            # Add precision metrics if available  
            precisions = result.get("precisions", [])
            for i, precision in enumerate(precisions):
                row[f"precision_{i}"] = precision
                
            row["n_metrics"] = result.get("n_metrics", 0)
            row["error"] = result.get("error", "")
            
            results_data.append(row)
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_folder / "all_results.csv", index=False)
        
        # Sort by score and save top results
        top_k = min(10, len(results))
        top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        # Save summary
        summary = {
            "total_experiments": len(results),
            "elapsed_time_seconds": elapsed_time,
            "best_score": top_results[0]["score"] if top_results else 0.0,
            "best_config": top_results[0]["config"] if top_results else {},
            "top_results": top_results
        }
        
        with open(results_folder / "grid_search_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        
        # Log top results
        logger.info(f"\nTop {top_k} configurations:")
        for rank, result in enumerate(top_results, 1):
            recalls = result.get("recalls", [])
            recall_str = " ".join([f"R{i}={r:.3f}" for i, r in enumerate(recalls)])
            logger.info(f"#{rank:02d} Score={result['score']:.4f} {recall_str} "
                       f"Exp={result['experiment_number']} Config={result['config']}")

    @staticmethod
    def calculate_recall_precision_at_quantiles(
            metrics: List[AbstractMetrics],
            quantiles: Sequence[float] = (0.25, 0.5, 0.75, 0.85),
    ) -> Tuple[List[float], List[float], List[float]]:

        by_mailshot = defaultdict(list)
        for m in metrics:
            if hasattr(m, "present_in_prediction") and not m.present_in_prediction:
                continue
            by_mailshot[m.mailshot_id].append(m)

        q = np.asarray(quantiles, dtype=np.float64)
        n_q = q.size
        rec_sum = np.zeros(n_q, dtype=np.float64)
        prec_sum = np.zeros(n_q, dtype=np.float64)
        weights = np.zeros(n_q, dtype=np.float64)

        for ms in by_mailshot.values():
            y_true = np.fromiter((m.opened for m in ms), dtype=np.int8)
            y_pred = np.fromiter(
                ((m.prediction - m.stage) if hasattr(m, "stage") else m.prediction
                 for m in ms),
                dtype=np.float64,
            )
            pos = y_true.sum()
            if pos == 0:
                continue

            order = np.argsort(y_pred)[::-1]
            y_true_sorted = y_true[order]
            cum_tp = np.cumsum(y_true_sorted, dtype=np.int64)

            k = np.ceil((1.0 - q) * len(y_pred)).astype(int) - 1
            k = np.clip(k, 0, len(y_pred) - 1)

            tp_k = cum_tp[k]
            rec_k = tp_k / pos
            pre_k = tp_k / (k + 1)

            w = len(ms)
            rec_sum += rec_k * w
            prec_sum += pre_k * w
            weights += w

        nz = weights > 0
        final_rec = np.zeros_like(rec_sum)
        final_prec = np.zeros_like(prec_sum)
        final_rec[nz] = rec_sum[nz] / weights[nz]
        final_prec[nz] = prec_sum[nz] / weights[nz]

        sent_pct = (1.0 - q).tolist()
        return sent_pct, final_rec.tolist(), final_prec.tolist()

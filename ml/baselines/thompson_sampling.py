import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self, Type
import torch

from ml.shallow_autoencoder.abstract_contextual_model import AbstractConfig, AbstractContextualModel
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

logger = logging.getLogger(__name__)

@dataclass
class ThompsonSamplingConfig(AbstractConfig):
    repetitions: int = 1

    def to_dict(self):
        return {
            "repetitions": self.repetitions
        }

    def to_filename(self):
        return f"thompson_sampling_g={self.G}_repetitions={self.repetitions}.json"

    @classmethod
    def from_filename(cls, filename: str) -> Self:
        parts = filename.split("_")
        repetitions = int(parts[2].split("=")[1].split(".")[0])
        return cls(G=g, repetitions=repetitions)

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, 'r') as f:
            data = json.load(f)
        return cls(repetitions=data["repetitions"])


class ThompsonSamplingModel(AbstractContextualModel):
    def __init__(self, config: ThompsonSamplingConfig):
        super().__init__(config)
        self.config = config
        self.repetitions = config.repetitions

    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        super().fit(train, val)
        logger.info("Fitting Thompson Sampling Model")

    def predict(self, data: AutoencoderDataset) -> List[DefaultMetrics]:
        train = self.train
        val = data

        alphas = torch.ones(train.shape[1]) * 1e-12
        betas = torch.ones(train.shape[1]) * 1e-12
        for user, grp in train.mails.groupby('user_id'):
            user = int(user)
            successes = int(grp['opened'].sum())
            failures = int(len(grp) - successes)
            alphas[user] += successes  # α = 1 + successes
            betas[user] += failures  # β = 1 + failures

        metrics_list: List[DefaultMetrics] = []
        for i in range(self.config.repetitions):
            samples = torch.distributions.Beta(alphas, betas).sample()
            predictions = torch.concatenate([samples[val.mailshot_user_indices(i)] for i in val.mailshot_ids])
            embeddings = torch.concatenate(
                [torch.tensor(val._mailshot_embeddings[i])[val.mailshot_user_indices(mailshot_id)] for i, mailshot_id in enumerate(val.mailshot_ids)])
            gold_data = embeddings
            for j, (opened, prediction) in enumerate(zip(gold_data, predictions)):
                metrics_list.append(DefaultMetrics(
                    mailshot_id=i,
                    opened=bool(opened),
                    prediction=float(prediction),
                ))

        return metrics_list

    @classmethod
    def model_name(cls) -> str:
        return "thompson_sampling"

    @classmethod
    def config_class(cls) -> Type[ThompsonSamplingConfig]:
        return ThompsonSamplingConfig







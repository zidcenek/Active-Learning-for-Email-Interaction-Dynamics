import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self, Type

import torch

from ml.shallow_autoencoder.abstract_contextual_model import (
    AbstractConfig,
    AbstractContextualModel,
)
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

logger = logging.getLogger(__name__)


@dataclass
class RandomConfig(AbstractConfig):
    seed: int = None

    def to_dict(self) -> dict:
        return {"seed": self.seed}

    def to_filename(self) -> str:
        return f"random_seed={self.seed}.json"

    @classmethod
    def from_filename(cls, filename: str) -> Self:
        stem = Path(filename).stem          # strips extension
        seed = int(stem.split("=")[1])
        return cls(seed=seed)

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, "r") as fh:
            data = json.load(fh)
        return cls(seed=data.get("seed", None))


class RandomModel(AbstractContextualModel):
    """
    A very simple baseline:  every (user, mailshot) interaction receives a score sampled uniformly from U(0, 1).
    """
    def __init__(self, config: RandomConfig):
        super().__init__(config)
        self.config = config

        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    # ------------------------------- training ------------------------------ #
    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        """
        Nothing to learn
        """
        super().fit(train, val)
        logger.info("RandomModel – no fitting required.")

    # ------------------------------ inference ------------------------------ #
    def predict(self, data: AutoencoderDataset) -> List[DefaultMetrics]:
        metrics: List[DefaultMetrics] = []

        for mailshot_id in range(len(data)):
            # Boolean ground-truth tensor for the current mailshot.
            opened_tensor = torch.tensor(
                data._mailshot_embeddings[mailshot_id]
            )[data.mailshot_user_indices(mailshot_id)]

            for opened in opened_tensor:
                random_score = random.random()  # uniform in [0, 1)
                metrics.append(
                    DefaultMetrics(
                        mailshot_id=mailshot_id,
                        opened=bool(opened),
                        prediction=float(random_score),
                    )
                )

        return metrics

    @classmethod
    def model_name(cls) -> str:
        return "random"

    @classmethod
    def config_class(cls) -> Type[RandomConfig]:
        return RandomConfig
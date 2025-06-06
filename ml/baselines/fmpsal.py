import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Self, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ml.shallow_autoencoder.abstract_contextual_model import \
        AbstractConfig, AbstractContextualModel
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class FMPSALConfig(AbstractConfig):
    latent_dim: int = 16
    lr: float = 1e-2
    epochs: int = 30
    repetitions: int = 3
    subsample_ratio: float = 0.5

    top_k_ratio: float = 0.1
    wait_hours: float = 4.0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__annotations__}

    def to_filename(self) -> str:
        return (f"fm_psal_k={self.latent_dim}"
                f"_lr={self.lr}_epochs={self.epochs}"
                f"_rep={self.repetitions}_sub={self.subsample_ratio}"
                f"_topk={self.top_k_ratio}"
                f"_wait={self.wait_hours}.json")

    @classmethod
    def from_json_file(cls, folder: Path,
                       filename: str = "config.json") -> Self:
        with open(folder / filename, "r") as fp:
            d = json.load(fp)
        return cls(**d)


# Factorization Machine
class _FM(nn.Module):
    def __init__(self, n_features: int, k: int):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.v = nn.Embedding(n_features, k)
        nn.init.normal_(self.linear.weight, 0., 1e-2)
        nn.init.normal_(self.v.weight, 0., 1e-2)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        lin = self.linear(x).sum(dim=1)
        Vx = self.v(x)
        interactions = 0.5 * ((Vx.sum(1) ** 2 - (Vx ** 2).sum(1))).sum(1)
        return (lin.squeeze(1) + interactions)


# FMPSAL MODEL
class FMPSALModel(AbstractContextualModel):
    def __init__(self, config: FMPSALConfig):
        super().__init__(config)
        self.cfg = config
        self.user2idx: Dict[int, int] = {}
        self.mail2idx: Dict[int, int] = {}
        self.n_features: int = 0

    def fit(self, train: AutoencoderDataset,
            val: AutoencoderDataset) -> None:

        super().fit(train, val)

        logger.info("[FM-PSAL] Preparing encoders")

        self.user2idx = {u: i for i, u in enumerate(sorted(train.mails['user_id'].unique()))}
        offset = len(self.user2idx)
        self.mail2idx = {m: i + offset for i, m in enumerate(sorted(train.mails['mailshot_id'].unique()))}
        self.n_features = offset + len(self.mail2idx) + 1  # +1 for <UNK>

        self.unk_mail_idx = self.n_features - 1
        self.train_user_popularity = train.mails[train.mails['opened'] == 1]['user_id'].value_counts()

    def predict(self, data: AutoencoderDataset) -> List[DefaultMetrics]:
        df_val = data.mails.copy()
        metrics: List[DefaultMetrics] = []

        popular_users = self.train_user_popularity.index.values
        total_users = len(popular_users)
        top_k_users = int(max(1, self.cfg.top_k_ratio * total_users))
        selected_users = set(popular_users[:top_k_users])

        for mail_id in df_val['mailshot_id'].unique():

            df_mail = df_val[df_val.mailshot_id == mail_id]

            df_explore = df_mail[df_mail.user_id.isin(selected_users)].copy()
            df_target  = df_mail[~df_mail.user_id.isin(selected_users)].copy()

            df_explore['time_to_open'] = pd.to_timedelta(df_explore['time_to_open'], unit='h')
            df_explore['original_opened'] = df_explore['opened'].copy()
            mask = (df_explore['opened'] == 1) & (df_explore['time_to_open'] <= pd.Timedelta(hours=self.cfg.wait_hours))
            df_explore['opened'] = mask.astype(int)

            if mail_id not in self.mail2idx:
                self.mail2idx[mail_id] = self.unk_mail_idx
                self.unk_mail_idx += 1
                self.n_features += 1

            df_train_aug = pd.concat([self.train.mails, df_explore], ignore_index=True)

            X_train, y_train = self._df_to_tensor(df_train_aug)
            X_target, y_target = self._df_to_tensor(df_target, return_y=True)

            for r in range(self.cfg.repetitions):
                logger.info(f"[FM-PSAL] mail={mail_id} rep={r+1}/{self.cfg.repetitions}")

                n = X_train.size(0)
                idx = torch.randint(high=n, size=(int(self.cfg.subsample_ratio * n),), device=DEVICE)
                X_boot = X_train[idx]
                y_boot = y_train[idx]

                model = _FM(self.n_features, self.cfg.latent_dim).to(DEVICE)
                self._train_one(model, X_boot, y_boot)

                with torch.no_grad():
                    probs = torch.sigmoid(model(X_target)).cpu().numpy()

                for mail_id, opened in zip(df_explore['mailshot_id'].values, df_explore['original_opened'].values):
                    metrics.append(DefaultMetrics(
                        mailshot_id=int(mail_id),
                        opened=bool(opened),
                        prediction=1.0
                    ))
                for opened, prob in zip(y_target, probs):
                    metrics.append(DefaultMetrics(
                        mailshot_id=int(mail_id),
                        opened=bool(opened),
                        prediction=float(prob),
                    ))

        return metrics

    def _train_one(self, model: _FM, X: torch.LongTensor, y: torch.FloatTensor) -> None:
        crit = nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)
        model.train()

        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()

    def _df_to_tensor(self, df: pd.DataFrame, return_y: bool = False)\
            -> Tuple[torch.LongTensor, torch.FloatTensor | np.ndarray]:

        user_ids = df['user_id'].map(lambda u: self.user2idx.get(u, 0)).values
        mailshot_ids = df['mailshot_id'].map(lambda m: self.mail2idx.get(m, self.unk_mail_idx)).values
        X_np = np.array([user_ids, mailshot_ids])
        X = torch.tensor(X_np, dtype=torch.long, device=DEVICE).t()
        y_arr = df['opened'].values.astype(np.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=DEVICE)

        return (X, y_arr) if return_y else (X, y_t)

    @classmethod
    def model_name(cls) -> str:
        return "fmpsal"

    @classmethod
    def config_class(cls) -> Type[FMPSALConfig]:
        return FMPSALConfig
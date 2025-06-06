import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Self, Type

import torch
import pandas as pd

from ml.shallow_autoencoder.abstract_contextual_model import (
    AbstractConfig,
    AbstractContextualModel,
)
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelBasedRLConfig(AbstractConfig):
    gamma: float = 0.9           # discount factor
    max_streak: int = 5          # cap for the open–streak state
    n_value_iter: int = 50       # how many value-iteration passes there are
    alpha_prior: float = 1.0     # Beta prior for successes
    beta_prior: float = 1.0      # Beta prior for failures
    kappa: float = 0.0           # Correlation adjustment parameter

    def to_dict(self):
        return {
            "gamma": self.gamma,
            "max_streak": self.max_streak,
            "n_value_iter": self.n_value_iter,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "kappa": self.kappa
        }

    def to_filename(self) -> str:
        return (
            f"model_based_rl_gamma={self.gamma}_maxStreak={self.max_streak}"
            f"_iters={self.n_value_iter}.json"
        )

    @classmethod
    def from_filename(cls, filename: str) -> Self:
        parts = {kv.split("=")[0]: kv.split("=")[1] for kv in filename.split("_")}
        return cls(
            gamma=float(parts["gamma"]),
            max_streak=int(parts["maxStreak"]),
            n_value_iter=int(parts["iters"].split(".")[0]),
        )

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, "r") as f:
            data = json.load(f)
        return cls(**data)


class ModelBasedRL(AbstractContextualModel):
    SEND, SKIP = 1, 0  # action encoding

    def __init__(self, config: ModelBasedRLConfig):
        super().__init__(config)
        self.cfg: ModelBasedRLConfig = config

        self._p_open: torch.Tensor | None = None       # P(open | streak)
        self._Q: torch.Tensor | None = None            # Q-table [S, A]

    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        """
        1. Build one Beta posterior per streak-state from the training logs.
        2. Solve the induced tabular MDP with value-iteration.
        """
        super().fit(train, val)
        logger.info("[ModelBasedRL] Fitting …")

        df: pd.DataFrame = train.mails.sort_values(["user_id", "mailshot_id"])

        # 1) COMPUTE the streak for every row in the log
        streaks = []
        last_open_streak: Dict[int, int] = {}  # user_id -> current streak
        for (_, row) in df.iterrows():
            uid = int(row["user_id"])
            opened = int(row["opened"])
            streak = last_open_streak.get(uid, 0)
            streaks.append(streak)

            # Update streak for the next mail to this user
            if opened:
                streak = min(streak + 1, self.cfg.max_streak)
            else:
                streak = 0
            last_open_streak[uid] = streak

        df = df.copy()
        df["streak"] = streaks

        # 2) BETA POSTERIOR for P(open | streak)
        s_max = self.cfg.max_streak
        alpha = torch.full((s_max + 1,), self.cfg.alpha_prior)
        beta = torch.full((s_max + 1,), self.cfg.beta_prior)

        for s in range(s_max + 1):
            subset = df[df["streak"] == s]
            succ = int(subset["opened"].sum())
            fail = int(len(subset) - succ)
            alpha[s] += succ
            beta[s] += fail

        self._p_open = alpha / (alpha + beta)  # posterior mean
        # Compute baseline open rate (user's marginal open rate)
        baseline_open_rate = df["opened"].mean()
        # Adjust probabilities based on kappa
        # When kappa=0, all streaks have the same probability (no causal effect)
        # When kappa=1, use the full observed correlation
        adjusted_p_open = baseline_open_rate + self.cfg.kappa * (self._p_open - baseline_open_rate)
        self._p_open = adjusted_p_open.clamp(min=0.0, max=1.0)

        # 3) VALUE / Q COMPUTATION (tabular)
        gamma = self.cfg.gamma
        S = s_max + 1
        A = 2  # send / skip
        V = torch.zeros(S)
        Q = torch.zeros(S, A)

        # Deterministic next-state helper
        next_state_open = torch.arange(S).clamp(max=s_max - 1) + 1  # min(s+1, max)
        next_state_no_open = torch.zeros(S, dtype=torch.long)

        for _ in range(self.cfg.n_value_iter):
            # Action SEND:
            #   reward  = p_open
            #   next V  = p_open * V[next_state_open] + (1-p_open) * V[next_state_no_open]
            vs_open = V[next_state_open]
            vs_no   = V[next_state_no_open]
            Q_send = self._p_open + gamma * (self._p_open * vs_open + (1 - self._p_open) * vs_no)

            # Action SKIP:
            Q_skip = gamma * V  # reward 0, streak unchanged

            Q[:, self.SEND] = Q_send
            Q[:, self.SKIP] = Q_skip
            V = torch.max(Q_send, Q_skip)

        self._Q = Q
        logger.info("[ModelBasedRL] Fit finished.")

    def predict(self, data: AutoencoderDataset) -> List[DefaultMetrics]:
        """
        For every mail in *chronological* order we
          • look up the user’s current streak,
          • compute P(open | streak) * (posterior mean),
          • store the metric,
          • update the user streak with the *observed* label so that
            evaluation in “online replay” style is possible.
        """
        assert self._p_open is not None and self._Q is not None, "Call fit() first."

        df: pd.DataFrame = data.mails.sort_values(["user_id", "mailshot_id"])
        metrics: List[DefaultMetrics] = []

        # Current streak for each user; seeded with streaks from training
        user_streak: Dict[int, int] = {}
        if hasattr(self, "train") and self.train is not None:
            # Initialise with streak after the last training mail per user
            last = (
                self.train.mails.sort_values(["user_id", "mailshot_id"])
                .groupby("user_id")
                .tail(1)
            )
            for (_, row) in last.iterrows():
                user_streak[int(row["user_id"])] = int(row.get("streak", 0))

        for _, row in df.iterrows():
            uid = int(row["user_id"])
            mailshot_id = int(row["mailshot_id"])
            opened = bool(row["opened"])

            streak = user_streak.get(uid, 0)
            streak_capped = min(streak, self.cfg.max_streak)

            # Policy decision – not actually used for evaluation (offline log replay)
            send_advantage = (
                self._Q[streak_capped, self.SEND] - self._Q[streak_capped, self.SKIP]
            )
            will_send = send_advantage > 0

            # Our “prediction” is the probability of open *if we decide to send*.
            # If the policy would skip we can return 0.0
            pred = float(self._p_open[streak_capped] if will_send else 0.0)

            metrics.append(
                DefaultMetrics(
                    mailshot_id=mailshot_id,
                    opened=opened,
                    prediction=pred,
                )
            )

            # Update streak with the *observed* outcome for the next step
            if opened and will_send:
                streak = min(streak + 1, self.cfg.max_streak)
            elif will_send:
                streak = 0  # we sent but the user ignored
            # Else (we decided to skip) keep the same streak

            user_streak[uid] = streak

        return metrics

    @classmethod
    def model_name(cls) -> str:
        return "model_based_rl"

    @classmethod
    def config_class(cls) -> Type[ModelBasedRLConfig]:
        return ModelBasedRLConfig
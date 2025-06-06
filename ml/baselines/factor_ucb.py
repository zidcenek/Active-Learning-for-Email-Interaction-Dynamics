import logging, json, numpy as np
import pandas as pd, scipy.sparse as sp
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Type, Self
from implicit.als import AlternatingLeastSquares
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression

from ml.shallow_autoencoder.abstract_contextual_model import AbstractContextualModel, AbstractConfig
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

logger = logging.getLogger(__name__)


@dataclass
class FactorUCBConfig(AbstractConfig):
    latent_dim: int = 32
    reg: float = 1e-2
    epochs: int = 20
    alpha: float = 0.5
    alpha_conf: float = 20.0
    score_scale: float = 10.0
    random_state: int = 42

    def to_dict(self):
        return asdict(self)

    def to_filename(self) -> str:
        return (f"factorUCB_l={self.latent_dim}"
                f"_reg={self.reg}"
                f"_epochs={self.epochs}"
                f"_alpha={self.alpha}.json")

    @classmethod
    def from_filename(cls, filename: str) -> Self:
        parts = {kv.split('=')[0]: kv.split('=')[1]
                 for kv in filename[:-5].split('_')[1:]}
        return cls(latent_dim=int(parts['l']),
                   reg=float(parts['reg']),
                   epochs=int(parts['epochs']),
                   alpha=float(parts['alpha']))

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, 'r') as f:
            data = json.load(f)
        return cls(latent_dim=data["latent_dim"],
                   reg=data["reg"],
                   epochs=data["epochs"],
                   alpha=data["alpha"],
                   # random_state=data["random_state"]
                   )


class FactorUCBModel(AbstractContextualModel):
    def __init__(self, config: FactorUCBConfig):
        super().__init__(config)
        self.cfg = config
        self.user_factors: np.ndarray | None = None   # (n_users, l)
        self.item_factors: np.ndarray | None = None   # (n_items, l)
        self._lambda_sqrt_inv = (1.0 / np.sqrt(self.cfg.reg))

    # Helpers
    @staticmethod
    def _create_id_maps(df: pd.DataFrame):
        """Map original ids to {0 … n-1} for implicit/NumPy."""
        u2i = {u: idx for idx, u in enumerate(df['user_id'].unique())}
        m2i = {m: idx for idx, m in enumerate(df['mailshot_id'].unique())}
        return u2i, m2i

    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset):
        super().fit(train, val)
        logger.info("Training factorUCB (ALS + bias + calibration)")

        df_train = train.mails[['user_id', 'mailshot_id', 'opened']]

        # integer codes -----------------------------------------------------
        u_codes, u_uniques = pd.factorize(df_train['user_id'])
        m_codes, m_uniques = pd.factorize(df_train['mailshot_id'])

        self.user2idx = dict(zip(u_uniques, range(len(u_uniques))))
        self.item2idx = dict(zip(m_uniques, range(len(m_uniques))))

        n_users, n_items = len(u_uniques), len(m_uniques)

        # confidence-weighted implicit matrix ------------------------------
        pref = df_train['opened'].astype(np.float32).to_numpy()  # p_ui
        conf = 1.0 + self.cfg.alpha_conf * pref  # c_ui
        matrix = sp.coo_matrix((conf, (m_codes, u_codes)),
                               shape=(n_items, n_users)).tocsr()

        als = AlternatingLeastSquares(
            factors=self.cfg.latent_dim,
            regularization=self.cfg.reg,
            iterations=self.cfg.epochs,
            use_gpu=False,
            random_state=self.cfg.random_state
        )
        als.fit(matrix)

        # ALS returns factors scaled for confidence; bring them back to
        # the preference scale so that dot-products are comparable with biases.
        scale = np.sqrt(self.cfg.alpha_conf)
        self.item_factors = als.item_factors * scale
        self.user_factors = als.user_factors * scale

        self.global_bias = df_train.opened.mean()

        self.user_bias = (
                df_train.groupby('user_id').opened.mean() - self.global_bias
        ).to_dict()

        self.item_bias = (
                df_train.groupby('mailshot_id').opened.mean() - self.global_bias
        ).to_dict()

        df_val = val.mails[['user_id', 'mailshot_id', 'opened']]
        raw_scores, labels = [], []
        for u_id, m_id, opened in df_val.itertuples(index=False):
            raw_scores.append(self._raw_score(u_id, m_id))
            labels.append(opened)

        # fall-back in case the validation set is all 0 or all 1
        if len(set(labels)) > 1:
            logger.warning("Multiple labels found for user_id, mailshot_id, opened")
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(np.array(raw_scores).reshape(-1, 1), labels)
            self.platt_a = float(lr.coef_[0][0])
            self.platt_b = float(lr.intercept_[0])
        else:
            self.platt_a, self.platt_b = 1.0, 0.0

        logger.info("Model ready   (#users %d , #items %d)", n_users, n_items)

    def _raw_score(self, u_id, m_id):
        """μ + b_u + b_a + uᵀv  + UCB(v)   (no sigmoid, no calibration)"""
        u_idx = self.user2idx.get(u_id)
        m_idx = self.item2idx.get(m_id)

        if u_idx is None or u_idx >= self.user_factors.shape[0]:
            u_vec = np.zeros(self.cfg.latent_dim, dtype=np.float32)
        else:
            u_vec = self.user_factors[u_idx]

        if m_idx is None or m_idx >= self.item_factors.shape[0]:
            v_vec = np.zeros(self.cfg.latent_dim, dtype=np.float32)
        else:
            v_vec = self.item_factors[m_idx]

        dot = float(np.dot(u_vec, v_vec))
        ucb = self.cfg.alpha * self._lambda_sqrt_inv * np.linalg.norm(v_vec)

        bias_u = self.user_bias.get(u_id, 0.0)
        bias_m = self.item_bias.get(m_id, 0.0)

        return self.global_bias + bias_u + bias_m + dot + ucb

    def predict(self, data: AutoencoderDataset) -> List[DefaultMetrics]:
        if self.user_factors is None:
            raise RuntimeError("call fit() first")

        df = data.mails[['user_id', 'mailshot_id', 'opened']]
        metrics: List[DefaultMetrics] = []

        for u_id, m_id, opened in df.itertuples(index=False):
            raw = self._raw_score(u_id, m_id)
            prob = sigmoid(self.platt_a * raw + self.platt_b)

            metrics.append(DefaultMetrics(
                mailshot_id=m_id,
                opened=bool(opened),
                prediction=float(prob)
            ))
        return metrics

    @classmethod
    def model_name(cls) -> str:
        return "factor_ucb"

    @classmethod
    def config_class(cls) -> Type[FactorUCBConfig]:
        return FactorUCBConfig
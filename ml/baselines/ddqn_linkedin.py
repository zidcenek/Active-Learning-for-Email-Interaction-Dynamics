import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Self, Optional, Type
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging

from experiment_utils.experiment_constants import DDQN_FOLDER_NAME
from ml.shallow_autoencoder.abstract_contextual_model import AbstractContextualModel, AbstractConfig
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.abstract_metrics import AbstractMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class DDQNMetrics(AbstractMetrics):
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mailshot_id": int(self.mailshot_id) if hasattr(self.mailshot_id, "item") else self.mailshot_id,
            "opened": bool(self.opened),
            "prediction": float(self.prediction) if hasattr(self.prediction, "item") else self.prediction
        }


    @classmethod
    def from_csv_gz(cls, folder: Path, filename: str = "results.csv.gz") -> List["DDQNMetrics"]:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(folder / filename, compression='gzip')

        # Convert each row to a DDQNMetrics object
        metrics_list = [
            cls(mailshot_id=row["mailshot_id"], opened=row["opened"], prediction=row["prediction"])
            for _, row in df.iterrows()
        ]

        return metrics_list

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List[Self]:
        """
        Convert a DataFrame to a list of DDQNMetrics objects.
        """
        metrics_list = []
        for _, row in df.iterrows():
            metrics_list.append(cls(
                mailshot_id=int(row["mailshot_id"]),
                opened=bool(row["opened"]),
                prediction=float(row["prediction"])
            ))
        return metrics_list


# Dataset for Offline RL
class OfflineRLDataset(Dataset):
    """
    Holds (s, a, r, s') transitions for offline RL.
    Expects each item in `transitions` to be (state, action, reward, next_state).
    """
    def __init__(self, transitions: List[Tuple[np.ndarray, int, float, np.ndarray]], mailshot_ids: Optional[List[int]] = None) -> None:
        super().__init__()
        self.data = transitions
        # Store mailshot IDs if provided (for metrics collection)
        self.mailshot_ids = mailshot_ids if mailshot_ids is not None else []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s, a, r, s_next = self.data[idx]
        return {
            "state": s.astype(np.float32),  # [state_dim]
            "action": a,
            "reward": r,
            "next_state": s_next.astype(np.float32),
            "mailshot_id": self.mailshot_ids[idx] if idx < len(self.mailshot_ids) else -1
        }

    @staticmethod
    def augment_df_with_open_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds two columns for each user:
          - total_opens_so_far: how many mails the user opened before this row
          - mails_since_last_open: a 'post-increment' counter of how many mails
            have been sent since the last open, where the row that opened=1
            displays the final count before resetting.

        Example:
          opens -> 0, 1, 1, 0, 0, 1
          desired mails_since_last_open -> 0, 1, 0, 0, 1, 2
        """

        # Sort so rows are in ascending order per user (and mailshot/time if needed)
        df = df.sort_values(["user_id", "mailshot_id"]).copy()

        logger.info("Augmenting DF with custom open stats (post-increment logic) ...")

        def _compute_user_stats(user_df: pd.DataFrame) -> pd.DataFrame:
            """
            For a single user's rows (already sorted), create two columns:
              mails_since_last_open, total_opens_so_far.
            The row that has opened=1 contains the count so far, then resets for subsequent rows.
            """
            mails_since_open = 0
            opens_so_far = 0
            mslo_list = []
            tosf_list = []

            for idx, row in user_df.iterrows():
                # Store current counters on this row
                mslo_list.append(mails_since_open)
                tosf_list.append(opens_so_far)

                if row["opened"] == 1:
                    # This row is an open => increment opens_so_far, reset mails_since_open
                    opens_so_far += 1
                    mails_since_open = 0
                else:
                    # Not opened => increment mails_since_open for the next row
                    mails_since_open += 1

            # Return a small DataFrame of computed columns, aligned to user_df's index
            return pd.DataFrame({
                "mails_since_last_open": mslo_list,
                "total_opens_so_far": tosf_list
            }, index=user_df.index)

        # Apply to each user separately, then rejoin
        stats_df = df.groupby("user_id", group_keys=False).apply(_compute_user_stats)
        df = df.join(stats_df)

        return df

    @staticmethod
    def _build_transitions(
            df: pd.DataFrame,
            user2idx: Dict[int, int],
            mailshot2idx: Dict[int, int],
            numeric_cols: List[str],
            unknown_mailshot_idx: int
    ) -> Tuple[List[Tuple[np.ndarray, int, float, np.ndarray]], List[int]]:
        """
        Builds (state , action , reward , next_state) tuples.

        • state      = [user_idx , mail_idx] + numeric features
        • next_state = the state that the *same* user will be in at the next
                       mail he/she receives (terminal rows → zeros /
                       “unknown-mailshot” embedding index).

        Args / Returns are unchanged.
        """
        logger.info(f"Building transitions for DataFrame of shape {df.shape}")

        # Make a defensive copy and sort chronologically per user
        df = df.sort_values(["user_id", "mailshot_id"]).copy()

        # 1) indices for user and mailshot
        df["user_idx"] = df["user_id"].map(user2idx)
        df["user_idx"] = df["user_idx"].fillna(len(user2idx)).astype(int)

        df["mail_idx"] = df["mailshot_id"].map(mailshot2idx)
        df["mail_idx"] = df["mail_idx"].fillna(unknown_mailshot_idx).astype(int)

        # 2) numeric features                                                 #
        numeric_array = df[numeric_cols].values.astype(np.float32)

        # 3) build CURRENT-STATE array                                        #
        states_array = np.concatenate([
            df["user_idx"].values.reshape(-1, 1).astype(np.float32),
            df["mail_idx"].values.reshape(-1, 1).astype(np.float32),
            numeric_array
        ], axis=1)

        # 4) build NEXT-STATE columns by *shifting inside each user*          #
        # shift(-1) gives the values of the next row within the same user;
        # the last mail of every user will therefore contain NaNs.
        df["next_user_idx"] = df.groupby("user_id")["user_idx"].shift(-1)
        df["next_mail_idx"] = df.groupby("user_id")["mail_idx"].shift(-1)

        for col in numeric_cols:
            df[f"next_{col}"] = df.groupby("user_id")[col].shift(-1)

        terminal_mask = df["next_mail_idx"].isna()

        # 5) fill the NaNs that belong to terminal rows
        #   • keep the same user_idx (it is still the same customer)
        #   • use the special unknown_mailshot_idx for mailshot
        #   • numeric features → 0
        df.loc[terminal_mask, "next_user_idx"] = df.loc[terminal_mask, "user_idx"]
        df.loc[terminal_mask, "next_mail_idx"] = unknown_mailshot_idx
        for col in numeric_cols:
            df.loc[terminal_mask, f"next_{col}"] = 0.0

        # After the fill all columns are numeric → convert to ndarray
        next_numeric_cols = [f"next_{c}" for c in numeric_cols]
        next_states_array = np.concatenate([
            df["next_user_idx"].values.reshape(-1, 1).astype(np.float32),
            df["next_mail_idx"].values.reshape(-1, 1).astype(np.float32),
            df[next_numeric_cols].values.astype(np.float32)
        ], axis=1)

        # 6) action / reward                                                 #
        rewards = (df["opened"] == 1).astype(np.float32).values
        actions = np.ones(len(df), dtype=np.int32)  # always SEND=1

        # 7) pack everything together                                         #
        transitions = list(zip(
            states_array,
            actions,
            rewards,
            next_states_array
        ))

        mailshot_ids = df["mailshot_id"].tolist()
        return transitions, mailshot_ids

    @classmethod
    def from_autoencoder_datasets(cls, train: AutoencoderDataset, val: AutoencoderDataset) -> Tuple[Self, Self, int, int]:
        num_val_mailshots = len(val.mailshot_ids)
        num_train_mailshots = len(train.mailshot_ids)
        val.mails['mailshot_id'] += num_train_mailshots
        df = pd.concat([train.mails, val.mails], axis=0)
        df = df[["user_id", "mailshot_id", "time_to_open", "opened"]].copy()
        df = cls.augment_df_with_open_stats(df)
        logger.info(f"Cols in df: {df.columns.tolist()}")

        unique_mailshots_sorted = sorted(df["mailshot_id"].unique())
        train_mshots = unique_mailshots_sorted[:-num_val_mailshots]
        val_mshots = unique_mailshots_sorted[-num_val_mailshots:]

        df_train = df[df["mailshot_id"].isin(train_mshots)].copy()
        df_val = df[df["mailshot_id"].isin(val_mshots)].copy()

        logger.info(f"Train mailshots: {len(df_train['mailshot_id'].unique())}")
        logger.info(f"Val mailshots: {len(df_val['mailshot_id'].unique())}")

        train_users = df_train["user_id"].unique()
        train_mailshots = df_train["mailshot_id"].unique()

        # Mappings
        user2idx = {u: i for i, u in enumerate(train_users)}
        mailshot2idx = {m: i for i, m in enumerate(train_mailshots)}
        UNKNOWN_MAILSHOT_IDX = len(mailshot2idx)

        # Build transitions
        numeric_cols = ["total_opens_so_far", "mails_since_last_open"]
        train_transitions, train_mailshot_ids = cls._build_transitions(df_train, user2idx, mailshot2idx, numeric_cols, UNKNOWN_MAILSHOT_IDX)
        val_transitions, val_mailshot_ids = cls._build_transitions(df_val, user2idx, mailshot2idx, numeric_cols, UNKNOWN_MAILSHOT_IDX)

        return cls(train_transitions, train_mailshot_ids), cls(val_transitions, val_mailshot_ids), len(train_mailshots), len(train_users)


# 2) QNetwork: a feed-forward net for Q(s,a) with embeddings
class EmbeddingQNetwork(nn.Module):
    def __init__(
            self,
            num_users: int,
            user_emb_dim: int,
            num_mailshots: int,
            mailshot_emb_dim: int,
            hidden_dim: int = 128,
            action_dim: int = 2,
            numerical_dim: int = 2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.mshot_emb = nn.Embedding(num_mailshots, mailshot_emb_dim)

        input_dim = user_emb_dim + mailshot_emb_dim + numerical_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[:,0] = user_idx, x[:,1] = mailshot_idx
        # x[:,2:] = numeric features
        user_ids = x[:, 0].long()
        mail_ids = x[:, 1].long()
        numeric_feats = x[:, 2:]  # shape [B, num_extra_features]

        user_vec = self.user_emb(user_ids)  # [B, user_emb_dim]
        mail_vec = self.mshot_emb(mail_ids)  # [B, mailshot_emb_dim]

        # Concatenate embeddings + numeric features
        cat = torch.cat([user_vec, mail_vec, numeric_feats], dim=1)
        return self.net(cat)


@dataclass
class DoubleDQNConfig(AbstractConfig):
    epochs: int = 5
    batch_size: int = 4096
    seed: int = 42
    user_emb_dim: int = 64
    mail_emb_dim: int = 16
    hidden_dim: int = 128
    lr: float = 1e-3
    wd: float = 1e-4
    gamma: float = 0.0  # single-step => gamma=0
    target_update_freq: int = 100
    device: str = "cpu"
    recall_fractions: List[float] = (0.15, 0.25, 0.50, 0.75)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return {
            "user_emb_dim": self.user_emb_dim,
            "mail_emb_dim": self.mail_emb_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "wd": self.wd,
            "gamma": self.gamma,
            "target_update_freq": self.target_update_freq,
            "device": self.device,
        }

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        """
        Loads the configuration from a JSON file.
        The file should be named according to the format specified in `to_filename`.
        """
        config_path = folder / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)


# 3) Double DQN Trainer (modified to do val loss & recall)
class DoubleDQNTrainer(AbstractContextualModel):
    def __init__(self, config: DoubleDQNConfig):
        super().__init__(config)
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq
        self.device = config.device
        self._train_rl_dataset: Optional[OfflineRLDataset] = None
        self._val_rl_dataset: Optional[OfflineRLDataset] = None
        self._q_net: Optional[nn.Module] = None
        self._target_net: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self.loss_fn = nn.MSELoss()
        self.global_step = 0
        self.unknown_mailshot_idx = -1
        
        # Create timestamp-based run ID and results directory
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")  # Format: 20250514-104420
        self.results_dir = os.path.join(os.getcwd(), "results", self.run_id)
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Create metrics subdirectory within the results directory
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

    @classmethod
    def model_name(cls) -> str:
        return DDQN_FOLDER_NAME

    @classmethod
    def config_class(cls) -> Type[DoubleDQNConfig]:
        return DoubleDQNConfig

    @classmethod
    def metrics_class(cls) -> Type[DDQNMetrics]:
        return DDQNMetrics

    @property
    def q_net(self) -> nn.Module:
        if self._q_net is None:
            raise ValueError("Q-network not initialized")
        return self._q_net

    @property
    def target_net(self) -> nn.Module:
        if self._target_net is None:
            raise ValueError("Target network not initialized")
        return self._target_net

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None:
            raise ValueError("Optimizer not initialized")
        return self._optimizer

    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        self._train = train
        self._val = val
        self._train_rl_dataset, self._val_rl_dataset, num_mailshots, num_users = OfflineRLDataset.from_autoencoder_datasets(train, val)
        self.unknown_mailshot_idx = num_mailshots  # keep for later use
        num_mailshots += 1  # add one for the unknown mailshot
        seed = self.config.seed
        
        # Save config to the result's directory
        config_path = os.path.join(self.results_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Config saved to {config_path}")

        # Build Q-net
        self._q_net = EmbeddingQNetwork(
            num_users=num_users,
            user_emb_dim=self.config.user_emb_dim,
            num_mailshots=num_mailshots,
            mailshot_emb_dim=self.config.mail_emb_dim,
            hidden_dim=self.config.hidden_dim,
            action_dim=2,
            numerical_dim=2,
        ).to(self.device)

        self._target_net = EmbeddingQNetwork(
            num_users=num_users,
            user_emb_dim=self.config.user_emb_dim,
            num_mailshots=num_mailshots,
            mailshot_emb_dim=self.config.mail_emb_dim,
            hidden_dim=self.config.hidden_dim,
            action_dim=2,
            numerical_dim=2,
        ).to(self.device)
        self._optimizer = optim.AdamW(self.q_net.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        self.target_net.load_state_dict(self.q_net.state_dict())

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create a training dataloader
        logger.info(f"Training on {len(self._train_rl_dataset)} transitions")
        dataset = self._train_rl_dataset
        val_dataset = self._val_rl_dataset
        logger.info(f"Batch size: {self.config.batch_size}")
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch+1}")
            # ---- TRAINING LOOP ----
            self.q_net.train()
            logger.info(f"Training on {len(self._train_rl_dataset)} transitions")
            total_loss = 0.0
            for i, batch in enumerate(dataloader):
                if i % 100 == 0:
                    logger.info(f"Batch {i}/{len(dataloader)}")
                states = batch["state"].to(self.device)   # shape [B, 2]
                actions = batch["action"].long().to(self.device)
                rewards = batch["reward"].float().to(self.device)
                next_states = batch["next_state"].to(self.device)
                dones = torch.tensor(
                    (batch["next_state"][:, 1] == self.unknown_mailshot_idx).float(),
                    device=self.device
                )

                # Q(s,a)
                q_values = self.q_net(states)
                q_values_chosen = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    q_next_online = self.q_net(next_states)
                    next_actions = q_next_online.argmax(dim=1)

                    q_next_target = self.target_net(next_states)
                    q_next_target_chosen = q_next_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

                    # If gamma=0 => q_target = reward
                    q_target = rewards + (1.0 - dones) * self.gamma * q_next_target_chosen

                loss = self.loss_fn(q_values_chosen, q_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self.global_step += 1

                if self.global_step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

            avg_loss = total_loss / len(dataloader)

            # ---- VALIDATION LOSS (every epoch) ----
            logger.info(f"Calculating validation loss...")
            val_loss = None
            if val_dataset is not None and len(val_dataset) > 0:
                val_loss = self.compute_validation_loss(val_dataset, batch_size=self.config.batch_size)

                # Collect validation metrics
                metrics = self.collect_validation_metrics(val_dataset, batch_size=self.config.batch_size)
                self.save_metrics_to_file(metrics, epoch)
                
                # Save model checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.save_model(epoch)

            if val_loss is not None:
                logger.info(f"[Epoch {epoch+1}/{self.config.epochs}] TrainLoss={avg_loss:.4f} | ValLoss={val_loss:.4f}")
            else:
                logger.info(f"[Epoch {epoch+1}/{self.config.epochs}] TrainLoss={avg_loss:.4f}")

            # ---- RECALL@k% every 'recall_eval_freq' epochs ----
            if val_dataset is not None and (epoch+1) % 1 == 0:
                recall_dict = compute_recall_at_k_percent_list(
                    q_net=self.q_net,
                    validation_data=val_dataset,
                    top_fractions=self.config.recall_fractions,
                    device=self.device
                )
                recall_str = ", ".join(
                    f"Recall@{int(f*100)}%={rec:.3f}"
                    for f, rec in recall_dict.items()
                )
                logger.info(f"   >> {recall_str}")

    def collect_validation_metrics(
        self,
        val_dataset: OfflineRLDataset,
        batch_size: int = 1024
    ) -> List[DDQNMetrics]:
        """
        Collects validation metrics for each sample in the validation dataset.
        Returns a list of DDQNMetrics objects containing mailshot_id, opened, and prediction.
        """
        logger.info("Collecting validation metrics...")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        metrics_list = []

        self.q_net.eval()
        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"].to(self.device)
                rewards = batch["reward"].float().to(self.device)  # 1.0 if opened else 0.0
                mailshot_ids = batch.get("mailshot_id", [-1] * len(states))  # Default to -1 if not present

                q_values = self.q_net(states)
                send_scores = q_values[:, 1].cpu().numpy()  # Get Q(s, action=1) for SEND action

                # Convert rewards to boolean opened status
                opened_status = rewards.cpu().numpy() > 0.5

                # Create a DDQNMetrics object for each sample and add to list
                for i in range(len(states)):
                    metrics_list.append(DDQNMetrics(
                        mailshot_id=int(mailshot_ids[i]),
                        opened=bool(opened_status[i]),
                        prediction=float(send_scores[i])
                    ))

        logger.info(f"Collected {len(metrics_list)} validation metrics")
        return metrics_list

    def save_metrics_to_file(self, metrics: List[DDQNMetrics], epoch: int) -> str:
        """
        Saves validation metrics to a CSV file with the epoch number included.
        Returns the path to the saved file.
        """
        # Create a filename with an epoch number
        metrics_file = os.path.join(self.metrics_dir, f"validation_metrics_epoch{epoch}.csv")

        # Write metrics to CSV
        with open(metrics_file, 'w', newline='') as csvfile:
            fieldnames = ['mailshot_id', 'opened', 'prediction', 'epoch']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for m in metrics:
                writer.writerow({
                    'mailshot_id': m.mailshot_id,
                    'opened': int(m.opened),  # Convert bool to int (1/0) for easier analysis
                    'prediction': m.prediction,
                    'epoch': epoch
                })

        logger.info(f"Saved {len(metrics)} validation metrics to {metrics_file}")
        return metrics_file
    
    def save_model(self, epoch: Optional[int] = None) -> str:
        """
        Saves the Q-network model to the results directory.
        If epoch is provided, includes it in the filename.
        Returns the path to the saved model.
        """
        filename = f"q_network_model{'_epoch'+str(epoch) if epoch is not None else '_final'}.pt"
        model_path = os.path.join(self.results_dir, filename)
        
        torch.save({
            'model_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch if epoch is not None else self.config.epochs,
            'global_step': self.global_step,
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path

    def predict(self, data: AutoencoderDataset) -> List[DDQNMetrics]:
        # data are expected to be the same as the validation set
        if self.val.shape == data.shape:
            return self.collect_validation_metrics(self._val_rl_dataset, batch_size=self.config.batch_size)
        else:
            raise NotImplementedError

    def compute_validation_loss(
        self,
        val_dataset: OfflineRLDataset,
        batch_size: int = 1024
    ) -> float:
        """
        Computes the Double DQN loss on the validation set (no gradient update).
        """
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.q_net.eval()
        self.target_net.eval()

        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"].to(self.device)
                actions = batch["action"].long().to(self.device)
                rewards = batch["reward"].float().to(self.device)
                next_states = batch["next_state"].to(self.device)

                q_values = self.q_net(states)
                q_values_chosen = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                # Double DQN target
                q_next_online = self.q_net(next_states)
                next_actions = q_next_online.argmax(dim=1)

                q_next_target = self.target_net(next_states)
                q_next_target_chosen = q_next_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

                dones = (batch["next_state"][:, 1] == self.unknown_mailshot_idx).float().to(self.device)

                q_target = rewards + (1.0 - dones) * self.gamma * q_next_target_chosen

                loss = self.loss_fn(q_values_chosen, q_target)
                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)


# 4) Evaluation: Accuracy & Recall@k%
def compute_accuracy(
        q_net: nn.Module,
        validation_dataset: OfflineRLDataset,
        device: str = "cpu"
) -> float:
    q_net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (state, _, opened, _) in validation_dataset.data:
            gt_action = 1 if opened > 0 else 0

            s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_net(s_tensor)
            pred_action = q_values.argmax(dim=1).item()

            if pred_action == gt_action:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def compute_recall_at_k_percent_list(
    q_net: nn.Module,
    validation_data: OfflineRLDataset,
    top_fractions: List[float],
    batch_size: int = 8192,
    device: str = "cpu"
) -> Dict[float, float]:
    """
    Vectorized version of recall@k%:
      1) Collect all states + opened labels.
      2) Forward pass in large batches to get Q(s, SEND).
      3) Sort scores descending and do a prefix sum of positives.
      4) Compute recall for each fraction in top_fractions.
    """
    q_net.eval()

    # 1) Gather states and opened
    logger.info("Gathering states and opened labels...")
    states_list = []
    opened_list = []
    for (state, _, opened, _) in validation_data.data:
        states_list.append(state)
        opened_list.append(int(opened > 0))

    states_array = np.stack(states_list, axis=0)  # [N, state_dim]
    opened_array = np.array(opened_list, dtype=np.int32)  # [N]
    N = len(states_array)

    total_positives = int(opened_array.sum())
    if total_positives == 0:
        return {f: 0.0 for f in top_fractions}

    # 2) Forward pass in batches to get Q(s, SEND=1)
    scores = np.zeros(N, dtype=np.float32)
    idx_start = 0
    with torch.no_grad():
        while idx_start < N:
            idx_end = min(idx_start + batch_size, N)
            batch_states = torch.tensor(
                states_array[idx_start:idx_end],
                dtype=torch.float32,
                device=device
            )
            q_values = q_net(batch_states)        # [B, action_dim]
            q_send = q_values[:, 1].cpu().numpy() # Q(s, action=1)
            scores[idx_start:idx_end] = q_send
            idx_start = idx_end

    # 3) Sort by q_send descending
    combined = np.stack([scores, opened_array], axis=1)  # shape [N, 2]
    combined = combined[combined[:, 0].argsort()[::-1]]

    # 4) Prefix the sum of positives
    prefix_positives = np.cumsum(combined[:,1])  # shape [N]; prefix_positives[i] = # of opens up to i-th sorted item

    # 5) Compute recall
    results = {}
    for f in top_fractions:
        K = int(f * N)
        if K <= 0:
            results[f] = 0.0
        else:
            # number of positives in top K => prefix_positives[K-1]
            positives_in_top_K = prefix_positives[K-1]
            recall = positives_in_top_K / total_positives
            results[f] = float(recall)

    return results


# Example usage / main
if __name__ == "__main__":
    sender_id =230711124317191757  # 240812112031380872
    logger.info(f"Loading dataset for sender_id {sender_id}...")
    datasets = AutoencoderDataset.from_disk_data_split(sender_id, split_sizes=[5, 10], remove_users_below_n_opens=1)
    train = datasets[0]
    val = datasets[1]
    test = datasets[2]
    logger.info(f"Dataset loaded. Train size: {len(train.mails)}, Val size: {len(val.mails)}, Test size: {len(test.mails)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DoubleDQNConfig(
        epochs=20,
        batch_size=4096,
        seed=123,
        user_emb_dim=64,
        mail_emb_dim=16,
        hidden_dim=32,
        lr=3e-4,
        wd=1e-5,
        gamma=0.95,
        device=device,
        target_update_freq=2000
    )
    ddqn_trainer = DoubleDQNTrainer(config=config)

    # Train, computing validation metrics
    logger.info(f"Starting training for {config.epochs} epochs...")
    ddqn_trainer.fit(train, val)
    logger.info(f"Training completed.")

    # Evaluate final performance
    acc_val = compute_accuracy(ddqn_trainer.q_net, ddqn_trainer._val_rl_dataset, device=device)
    logger.info(f"Validation Accuracy: {acc_val:.3f}")
    
    # Save final model and accuracy
    ddqn_trainer.save_model()
    with open(os.path.join(ddqn_trainer.results_dir, "final_metrics.json"), 'w') as f:
        json.dump({
            "validation_accuracy": acc_val,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info(f"All results saved to: {ddqn_trainer.results_dir}")


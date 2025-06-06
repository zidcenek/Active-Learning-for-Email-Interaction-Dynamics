import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Self, Type, Dict, Tuple
import torch
import numpy as np
from collections import defaultdict

from ml.shallow_autoencoder.abstract_contextual_model import AbstractConfig, AbstractContextualModel
from ml.shallow_autoencoder.dataset.autoencoder_dataset import AutoencoderDataset
from ml.shallow_autoencoder.metrics.default_metrics import DefaultMetrics

logger = logging.getLogger(__name__)


@dataclass
class FMFCDBConfig(AbstractConfig):
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    feature_dim: int = 4
    batch_size: int = 1000
    n_epochs: int = 10
    sample_rate: float = 0.1

    # Parameters for batched active learning
    num_batches: int = 48  # Number of batches to split users into
    observation_hours: float = 1.0  # Hours to wait between batches
    active_learning_weight: float = 0.3  # Weight for updating features
    early_exploration_boost: float = 2.0  # Extra exploration in early batches
    update_frequency: int = 5  # Update features every N batches

    def to_dict(self):
        return {
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "feature_dim": self.feature_dim,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "sample_rate": self.sample_rate,
            "num_batches": self.num_batches,
            "observation_hours": self.observation_hours,
            "active_learning_weight": self.active_learning_weight,
            "early_exploration_boost": self.early_exploration_boost,
            "update_frequency": self.update_frequency
        }

    def to_filename(self):
        return f"fmfc_db_er={self.exploration_rate}_lr={self.learning_rate}_nb={self.num_batches}.json"

    @classmethod
    def from_filename(cls, filename: str) -> Self:
        parts = filename.split("_")
        er = float(parts[3].split("=")[1])
        lr = float(parts[4].split("=")[1])
        nb = int(parts[5].split("=")[1].split(".")[0])
        return cls(exploration_rate=er, learning_rate=lr, num_batches=nb)

    @classmethod
    def from_json_file(cls, folder: Path, filename: str = "config.json") -> Self:
        with open(folder / filename, 'r') as f:
            data = json.load(f)
        return cls(**data)


class FMFCDBModel(AbstractContextualModel):

    def __init__(self, config: FMFCDBConfig):
        super().__init__(config)
        self.config = config
        self.user_features = None
        self.feature_weights = None
        self.user_stats = None

        # For tracking metrics with user information
        self.user_metric_map = {}  # Maps (mailshot_id, user_id) -> metric_index

    def fit(self, train: AutoencoderDataset, val: AutoencoderDataset) -> None:
        super().fit(train, val)
        logger.info("Fitting FMFC-DB Model (Optimized)")

        # Extract user features
        self.user_features, self.user_stats = self._extract_user_features_vectorized(train)

        # Initialize feature weights
        self.feature_weights = torch.ones(self.config.feature_dim) * 0.1

        # Learn weights using SGD
        self._learn_weights_sgd(train)

    def _extract_user_features_vectorized(self, data: AutoencoderDataset) -> Tuple[torch.Tensor, Dict]:
        """Extract features using vectorized pandas operations"""
        num_users = data.num_users
        features = torch.zeros(num_users, self.config.feature_dim)

        # Pre-compute user statistics
        user_stats = {}
        user_groups = data.mails.groupby('user_id')
        logger.info("Computing user statistics...")

        # Feature 1: Overall open rate
        open_rates = user_groups['opened'].mean()
        features[:, 0] = torch.tensor(open_rates.reindex(range(num_users), fill_value=0).values, dtype=torch.float32)

        # Feature 2: Activity level (number of mailshots received / total)
        activity_levels = user_groups.size() / len(data)
        features[:, 1] = torch.tensor(activity_levels.reindex(range(num_users), fill_value=0).values,
                                      dtype=torch.float32)

        # Feature 3: Recent activity (last 5 mailshots) - simplified
        def recent_open_rate(group):
            return group.tail(5)['opened'].mean() if len(group) >= 5 else group['opened'].mean()

        recent_rates = user_groups.apply(recent_open_rate)
        features[:, 2] = torch.tensor(recent_rates.reindex(range(num_users), fill_value=0).values, dtype=torch.float32)

        # Feature 4: Consistency (1 - std of open rate)
        def consistency_score(group):
            if len(group) < 3:
                return 0.5
            return max(0, 1 - group['opened'].std())

        consistency = user_groups.apply(consistency_score)
        features[:, 3] = torch.tensor(consistency.reindex(range(num_users), fill_value=0.5).values, dtype=torch.float32)

        # Additional features if feature_dim > 4
        if self.config.feature_dim > 4:
            # Feature 5: Trend (increasing/decreasing open rate)
            def calculate_trend(group):
                if len(group) < 3:
                    return 0.5
                x = np.arange(len(group))
                y = group['opened'].values
                if np.std(x) > 0 and np.std(y) > 0:
                    trend = np.corrcoef(x, y)[0, 1]
                    return (trend + 1) / 2  # Normalize to [0, 1]
                return 0.5

            trends = user_groups.apply(calculate_trend)
            features[:, 4] = torch.tensor(trends.reindex(range(num_users), fill_value=0.5).values, dtype=torch.float32)

        if self.config.feature_dim > 5:
            # Feature 6: Engagement decay (how fast user engagement drops)
            def engagement_decay(group):
                if len(group) < 10:
                    return 0.5
                # Compare first third vs last third
                third = len(group) // 3
                early_rate = group.head(third)['opened'].mean()
                late_rate = group.tail(third)['opened'].mean()
                if early_rate > 0:
                    return min(1.0, late_rate / early_rate)
                return 0.5

            decay = user_groups.apply(engagement_decay)
            features[:, 5] = torch.tensor(decay.reindex(range(num_users), fill_value=0.5).values, dtype=torch.float32)

        # Store pre-computed stats for later use
        user_stats['open_rates'] = open_rates
        user_stats['counts'] = user_groups.size()
        user_stats['last_mailshot'] = user_groups['mailshot_id'].max()

        logger.info(f"Features extracted for {num_users} users with {self.config.feature_dim} dimensions")

        return features, user_stats

    def _learn_weights_sgd(self, train: AutoencoderDataset):
        """Learn weights using mini-batch SGD with sampling"""

        # Sample mailshots for training
        n_mailshots = len(train)
        sample_size = max(1, int(n_mailshots * self.config.sample_rate))

        logger.info(f"Training on {sample_size} sampled mailshots out of {n_mailshots}")

        for epoch in range(self.config.n_epochs):
            # Sample mailshots
            sampled_mailshots = np.random.choice(n_mailshots, sample_size, replace=False)

            total_loss = 0
            n_samples = 0

            # Process in batches
            for mailshot_idx in sampled_mailshots:
                # Get all users for this mailshot at once
                mailshot_data = train._mailshot_embeddings[mailshot_idx]
                mailshot_id = train.mailshot_ids[mailshot_idx]
                user_indices = train.mailshot_user_indices(mailshot_id)

                # Filter valid users
                valid_users = [u for u in user_indices if u < self.user_features.shape[0]]

                if len(valid_users) == 0:
                    continue

                user_features_batch = self.user_features[valid_users]
                predictions = torch.sigmoid(user_features_batch @ self.feature_weights)

                # Get actual values
                actuals = torch.tensor([mailshot_data[u] for u in valid_users])

                # Compute gradients
                errors = actuals - predictions
                gradients = errors.unsqueeze(1) * user_features_batch

                # Update weights
                self.feature_weights += self.config.learning_rate * gradients.mean(dim=0)

                # Track loss
                total_loss += (errors ** 2).sum().item()
                n_samples += len(valid_users)

            if epoch % 2 == 0:
                avg_loss = total_loss / max(n_samples, 1)
                logger.info(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Samples: {n_samples}")

    def predict(self, data: AutoencoderDataset, use_batched_active_learning: bool = True) -> List[DefaultMetrics]:
        if use_batched_active_learning and len(data) == 1:
            # Batched active learning for a single mailshot
            metrics, _ = self.predict_batched_active_learning(data)
            return metrics
        else:
            # Standard prediction without active learning
            return self._predict_standard(data)

    def _predict_standard(self, data: AutoencoderDataset) -> List[DefaultMetrics]:
        """Standard prediction without active learning"""
        train = self.train
        val = data

        metrics_list: List[DefaultMetrics] = []

        # Pre-compute exploration decisions for all predictions
        n_predictions = len(val.mails)
        explore_decisions = np.random.random(n_predictions) < self.config.exploration_rate
        pred_idx = 0

        # Process each mailshot
        for i in range(len(val)):
            mailshot_id = val.mailshot_ids[i]

            # Get the actual mails for this mailshot
            mailshot_mails = val.mails[val.mails['mailshot_id'] == mailshot_id]

            # Create metrics directly from the mails dataframe
            for _, mail in mailshot_mails.iterrows():
                user_id = mail['user_id']

                # Check if a user exists in training
                if user_id < self.user_features.shape[0]:
                    user_feat = self.user_features[user_id]
                    model_prediction = torch.sigmoid(torch.dot(self.feature_weights, user_feat)).item()

                    if explore_decisions[pred_idx]:
                        prediction = np.random.random()
                    else:
                        prediction = model_prediction
                else:
                    prediction = train.average_opens

                pred_idx += 1

                metrics_list.append(DefaultMetrics(
                    mailshot_id=mailshot_id,
                    opened=bool(mail['opened']),
                    prediction=float(prediction),
                ))

        return metrics_list

    def predict_batched_active_learning(self, data: AutoencoderDataset) -> Tuple[List[DefaultMetrics], Dict[int, List[Tuple[int, float]]]]:
        """
        Predict with batched active learning for a single mailshot.
        Returns metrics and user scores for each batch.
        """
        train = self.train
        val = data

        if len(val) != 1:
            raise ValueError("Batched active learning expects exactly one mailshot")

        mailshot_id = val.mailshot_ids[0]
        mailshot_mails = val.mails[val.mails['mailshot_id'] == mailshot_id]
        all_users = mailshot_mails['user_id'].unique()

        # Shuffle users for random batch assignment
        np.random.seed(42)
        shuffled_users = np.random.permutation(all_users)

        # Calculate batch sizes
        users_per_batch = len(shuffled_users) // self.config.num_batches

        # Track which users were sent at which stage
        sent_at_stage: Dict[int, int] = {}  # user_id -> stage

        # Track opens for updating
        observed_opens: Dict[int, bool] = {}  # user_id -> opened

        # Store all metrics
        all_metrics = []
        batch_scores = {}  # stage -> list of (user_id, score)

        # Map to track metrics by user
        self.user_metric_map = {}

        logger.info(f"Starting batched prediction for {len(all_users)} users in {self.config.num_batches} batches")

        for stage in range(self.config.num_batches):
            # Determine users for this batch
            start_idx = stage * users_per_batch
            if stage == self.config.num_batches - 1:
                # Last batch gets any remaining users
                batch_users = shuffled_users[start_idx:]
            else:
                batch_users = shuffled_users[start_idx:start_idx + users_per_batch]

            logger.info(f"Batch {stage + 1}/{self.config.num_batches}: Predicting for {len(batch_users)} users")

            # Make predictions for this batch
            batch_predictions = []
            batch_metrics = []

            for user_id in batch_users:
                sent_at_stage[user_id] = stage

                # Make prediction
                if user_id < self.user_features.shape[0]:
                    user_feat = self.user_features[user_id]

                    # Add some exploration in early batches
                    early_batch_threshold = 5
                    if stage < early_batch_threshold and np.random.random() < self.config.exploration_rate * self.config.early_exploration_boost:
                        prediction = np.random.random()
                    elif np.random.random() < self.config.exploration_rate:
                        prediction = np.random.random()
                    else:
                        prediction = torch.sigmoid(torch.dot(self.feature_weights, user_feat)).item()
                else:
                    prediction = train.average_opens

                batch_predictions.append((user_id, prediction))

                # Create metric
                metric = DefaultMetrics(
                    mailshot_id=mailshot_id,
                    opened=False,  # Will be updated based on observation
                    prediction=float(prediction),
                )

                # Track metric index for this user
                metric_idx = len(all_metrics)
                self.user_metric_map[(mailshot_id, user_id)] = metric_idx
                all_metrics.append(metric)
                batch_metrics.append((user_id, metric_idx))

            # Store predictions for this batch
            batch_scores[stage] = batch_predictions

            # Simulate waiting for the observation period
            observation_minutes = self.config.observation_hours * 60

            # Update opens for ALL previously sent batches
            opened_indices = self._update_opens_all_stages(
                val, mailshot_id, sent_at_stage, stage, observation_minutes
            )

            # Update observed opens
            for user_id in opened_indices:
                observed_opens[user_id] = True
                # Update the corresponding metric
                if (mailshot_id, user_id) in self.user_metric_map:
                    metric_idx = self.user_metric_map[(mailshot_id, user_id)]
                    all_metrics[metric_idx].opened = True

            # Active learning: Update features based on observations
            if stage > 0 and stage % self.config.update_frequency == 0:
                self._update_features_from_observations(sent_at_stage, observed_opens, stage)

            logger.info(f"Batch {stage + 1}: Sent to {len(batch_users)} users, "
                        f"total opens so far: {len(observed_opens)}")

        # The final update of all opens with full observation window
        final_opened = self._update_opens_all_stages(
            val, mailshot_id, sent_at_stage, self.config.num_batches - 1, np.inf
        )

        # Update final opens
        for user_id in final_opened:
            observed_opens[user_id] = True
            if (mailshot_id, user_id) in self.user_metric_map:
                metric_idx = self.user_metric_map[(mailshot_id, user_id)]
                all_metrics[metric_idx].opened = True

        logger.info(f"Completed batched prediction. Total opens: {len(observed_opens)}/{len(all_users)} "
                    f"({100.0 * len(observed_opens) / len(all_users):.1f}%)")

        return all_metrics, batch_scores

    def _update_opens_all_stages(self, dataset: AutoencoderDataset, mailshot_id: int,
                                 sent_at_stage: Dict[int, int], current_stage: int,
                                 frame_minutes: float) -> List[int]:
        """Update opens for all stages up to current stage"""
        opened_indices: List[int] = []
        grouped_sent_at_stage = defaultdict(list)

        for user, stage in sent_at_stage.items():
            grouped_sent_at_stage[stage].append(user)

        for stage, users in grouped_sent_at_stage.items():
            # Calculate the time window for this stage
            time_window = frame_minutes * (current_stage - stage + 1)
            stage_opened = dataset.select_opened_indices(mailshot_id, users, time_window)
            opened_indices.extend(stage_opened)

        return list(set(opened_indices))  # Remove duplicates

    def _update_features_from_observations(self, sent_at_stage: Dict[int, int],
                                           observed_opens: Dict[int, bool],
                                           current_stage: int):
        """Update user features based on observed interactions"""
        logger.info(f"Updating features after stage {current_stage}")

        # Group users by how many stages ago they were sent
        users_by_delay = defaultdict(list)
        for user_id, stage in sent_at_stage.items():
            delay = current_stage - stage
            users_by_delay[delay].append(user_id)

        # Update features for each user
        updates_made = 0
        for user_id in sent_at_stage:
            if user_id < self.user_features.shape[0]:
                opened = observed_opens.get(user_id, False)
                stage = sent_at_stage[user_id]
                delay = current_stage - stage

                # Weight updates based on how recent the observation is
                recency_weight = 0.9 ** delay  # More recent observations get higher weight
                update_weight = self.config.active_learning_weight * recency_weight

                # Update features
                # Feature 0: Overall open rate
                old_val = self.user_features[user_id, 0].item()
                self.user_features[user_id, 0] = (
                        (1 - update_weight) * old_val +
                        update_weight * float(opened)
                )

                # Feature 2: Recent open rate (update more aggressively)
                old_recent = self.user_features[user_id, 2].item()
                self.user_features[user_id, 2] = (
                        (1 - update_weight * 1.5) * old_recent +
                        update_weight * 1.5 * float(opened)
                )

                # Feature 3: Update consistency based on deviation from expected
                if self.config.feature_dim > 3:
                    expected = old_val
                    deviation = abs(float(opened) - expected)
                    consistency_update = 1 - deviation
                    old_consistency = self.user_features[user_id, 3].item()
                    self.user_features[user_id, 3] = (
                            (1 - update_weight * 0.5) * old_consistency +
                            update_weight * 0.5 * consistency_update
                    )

                updates_made += 1

        logger.info(f"Updated features for {updates_made} users")

        # Optionally retrain weights with updated features on a sample
        if current_stage % 10 == 0:
            self._quick_weight_update()

    def _quick_weight_update(self):
        """Quick weight update based on recent observations"""
        # Apply small regularization to prevent weight explosion
        self.feature_weights = self.feature_weights * 0.99

        # Could implement a few gradient steps here based on recent observations
        # For now, we just apply regularization
        logger.info("Applied weight regularization")

    @classmethod
    def model_name(cls) -> str:
        return "fmfcdb"

    @classmethod
    def config_class(cls) -> Type[FMFCDBConfig]:
        return FMFCDBConfig
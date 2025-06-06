import logging
import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import torch.nn.init as init

from utils.common_utils import set_seed

# References:
# - Sedhain S. et al., "AutoRec: Autoencoders Meet Collaborative Filtering," WWW 2015.
#   https://arxiv.org/abs/1508.01195
# - "Shallow linear autoencoders for collaborative filtering" approaches in related CF literature.

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    loss: float
    average_train_ndcg: Optional[float]
    average_ndcg: Optional[float]
    average_f1: Optional[float]
    recall: Optional[float]
    precision: Optional[float]
    val_loss: Optional[float]
    val_average_ndcg: Optional[float]
    val_average_f1: Optional[float]
    val_average_iou: Optional[float]

    def __post_init__(self):
        # print(f"TrainingMetrics: {self}")
        pass

    def __str__(self):
        return f"loss: {self.loss:.4f}, val (masked) NDCG: {self.average_ndcg:.4f}, val (masked) f1: {self.average_f1:.4f}, val loss: {self.val_loss:.4f}, val NDCG: {self.val_average_ndcg:.4f}, val f1: {self.val_average_f1:.4f}"
        # return f"loss: {self.loss:.4f}, recall: {self.recall:.4f}, precision: {self.precision:.4f}, val_loss: {self.val_loss:.4f}, val_recall: {self.val_recall:.4f}, val_precision: {self.val_precision:.4f}"

    @staticmethod
    def plot_average_train_ndcg(results: List["TrainingMetrics"]):
        average_ndcg = [result.average_train_ndcg for result in results]
        print(f"Max average_ndcg: {max(average_ndcg):.4f}")
        print(f"End average_ndcg: {average_ndcg[-1]:.4f}")
        plt.plot(average_ndcg, label='average_ndcg')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')
        plt.show()


    @staticmethod
    def plot_val_precision_recall(results: List["TrainingMetrics"]):
        val_precision = [result.val_precision for result in results]
        val_recall = [result.val_recall for result in results]
        print(f"Max val_precision: {max(val_precision):.4f}, Max val_recall: {max(val_recall):.4f}")
        print(f"End val_precision: {val_precision[-1]:.4f}, End val_recall: {val_recall[-1]:.4f}")
        print(f"Loss at max val_precision: {results[val_precision.index(max(val_precision))].val_loss:.4f}")
        plt.plot(val_precision, label='val_precision')
        plt.plot(val_recall, label='val_recall')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_average_ndcg(results: List["TrainingMetrics"]):
        average_ndcg = [result.average_ndcg for result in results]
        print(f"Max average_ndcg: {max(average_ndcg):.4f}")
        print(f"End average_ndcg: {average_ndcg[-1]:.4f}")
        plt.plot(average_ndcg, label='average_ndcg')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')
        plt.show()

    @staticmethod
    def plot_f1_score(results: List["TrainingMetrics"]):
        f1 = [result.average_f1 for result in results]
        print(f"Max f1: {max(f1):.4f}")
        print(f"End f1: {f1[-1]:.4f}")
        plt.plot(f1, label='f1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_val_f1_score(results: List["TrainingMetrics"]):
        f1 = [result.val_average_f1 for result in results]
        print(f"Max f1: {max(f1):.4f}")
        print(f"End f1: {f1[-1]:.4f}")
        plt.plot(f1, label='f1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_val_loss(results: List["TrainingMetrics"]):
        val_loss = [result.val_loss for result in results]
        print(f"Min val_loss: {min(val_loss):.4f}")
        print(f"End val_loss: {val_loss[-1]:.4f}")
        plt.plot(val_loss, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_val_ndcg(results: List["TrainingMetrics"]):
        val_ndcg = [result.val_average_ndcg for result in results]
        print(f"Max val_ndcg: {max(val_ndcg):.4f}")
        print(f"End val_ndcg: {val_ndcg[-1]:.4f}")
        plt.plot(val_ndcg, label='val_ndcg')
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_and_val_loss(results: List["TrainingMetrics"]):
        train_loss = [result.loss.detach().numpy() for result in results]
        val_loss = [result.val_loss.detach().numpy() for result in results]
        print(f"Min val_loss: {min(val_loss):.4f}")
        print(f"End val_loss: {val_loss[-1]:.4f}")
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class ShallowAutoencoder(nn.Module):
    """
    Shallow linear autoencoder based on the formulation:

       B_{E,D} = E D^T - diag([E ⊙ D] 1_n)
       X B_{E,D} = X E D^T - X diag([E ⊙ D] 1_n)

    E, D in R^{n x d} are trainable parameters (n = #items or #features, d = latent dimension).
    diag(...) ensures the diagonal is zeroed out, avoiding trivial solutions.

    n - usually around 100K (users or features)
    d - latent dimension, usually 8-20
    number of rows usually 100 templates
    """

    def __init__(self, n: int, d: int, device: str = None, layer_norm=False, dropout_p=0.1):
        super().__init__()
        self.E = nn.Parameter(torch.empty(n, d))
        self.D = nn.Parameter(torch.empty(n, d))

        # Use Xavier (Glorot) initialization to avoid outputs hovering around 0.5
        init.xavier_uniform_(self.E)
        init.xavier_uniform_(self.D)
        self.dropout = nn.Dropout(p=dropout_p)

        # BatchNorm1d applied after the hidden layer
        if layer_norm:
            logger.info("Using LayerNorm")
            self.bn = nn.LayerNorm(d)
        else:
            logger.info("Using BatchNorm1d")
            self.bn = nn.BatchNorm1d(d)
        # self.dropout = nn.Dropout(p=dropout_p)
        #
        self.register_buffer("x_min", torch.tensor(0.0))
        self.register_buffer("x_max", torch.tensor(0.0))
        if device != 'cpu':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = "cpu"
        else:
            self.device = 'cpu'

        self.training_metrics = []

    def forward(self, X):
        """
        This function assumes the input X is a batch of data. It will be multiplied by E and D to get the hidden layer.
        :param X: torch.Tensor of shape [batch_size, n]
        :return: torch.Tensor of shape [batch_size, n] -> this should represent the reconstruction of the input.
        """
        # res = X @ torch.diag((E * D) @ torch.ones(d))
        # res2 = X * (E * D).sum(dim=1)
        # res == res2
        diag_vals = (self.E * self.D).sum(dim=1)  # * np.sqrt(1.0 / self.D.size(1))
        hidden = X @ self.E

        # Apply normalization
        hidden = self.bn(hidden)
        hidden = self.dropout(hidden)

        out = hidden @ self.D.t()
        out = out - (X * diag_vals)
        return torch.sigmoid(out)

    def forward_for_user(self, X):
        """
        This function assumes there will be only 1 user in the input X. Exactly one 1 and roughly 100k 0s.
        :param X: torch.Tensor of shape [1, n]
        :return: torch.Tensor of shape [1, n] -> this should represent how predictive this user is for each user.
        """
        diag_vals = (self.E * self.D).sum(dim=1)
        hidden = X @ self.E
        out = hidden @ self.D.t()
        out = out - (X * diag_vals)  # TODO: uncomment this
        return torch.sigmoid(out)

    def fit(
            self,
            train: Dataset | torch.Tensor,
            epochs=10,
            lr=1e-3,
            batch_size=1024,
            weight_decay=1e-4,
            positive_weight=5.0,
            label_smoothing=0.0,
            val: Dataset | torch.Tensor = None,
            full_training: bool = False
    ):
        """
        Trains the model on the given dataset X and calculates validation IoU if a validation set is provided.
        IoU calculation reference (Jaccard index): https://en.wikipedia.org/wiki/Jaccard_index
        """
        set_seed(42)

        if isinstance(train, torch.Tensor):
            train = TensorDataset(train)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

        if val is not None:
            if isinstance(val, torch.Tensor):
                val = TensorDataset(val)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        self.to(self.device)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=6e-5)

        criterion = nn.BCELoss(reduction='none')
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            training_ndcg = 0.0
            for (training_batch, learning_batch, validation_batch) in train_loader:
                if full_training:
                    batch_data = learning_batch
                    target_data = validation_batch
                else:
                    batch_data = training_batch
                    target_data = learning_batch
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()

                reconstruction = self.forward(batch_data)

                target = (
                    target_data * (1 - label_smoothing) + (1 - target_data) * label_smoothing
                    if label_smoothing > 0
                    else batch_data
                )
                loss = criterion(reconstruction, target)
                # print(loss.shape)

                # Weighted loss
                weights = torch.where(batch_data > 0.5, positive_weight, 1)
                mask = target_data - batch_data
                weights_unseen_data = torch.where(mask > 0.5, 0.5, 0)
                weights = weights + weights_unseen_data
                loss.backward(weights)

                # Uncomment this if you want to use weighted loss
                # weighted_loss = (loss * weights).mean()
                # weighted_loss.backward()
                # This is where you clip gradients before taking the optimizer step
                # clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    ndcg_batch = compute_batch_ndcg(batch_data, target_data, reconstruction, k=1000)
                    training_ndcg += ndcg_batch * batch_data.size(0)

                train_loss += loss.mean() * batch_data.size(0)
            scheduler.step()
            avg_train_ndcg = training_ndcg / len(train_loader.dataset)
            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation 1 - NDCG on masked entries of training set
            if epoch % 1 == 0:
                self.eval()
                total_ndcg = 0.0
                total_items = 0
                f1_scores = []
                num_batches = 0
                with torch.no_grad():
                    for (batch_data, _, val_data) in train_loader:
                        # NDCG
                        batch_data, val_data = batch_data.to(self.device), val_data.to(self.device)
                        reconstruction = self.forward(batch_data)
                        ndcg_batch = compute_batch_ndcg(batch_data, val_data, reconstruction, k=1000)
                        total_ndcg += ndcg_batch * batch_data.size(0)
                        total_items += batch_data.size(0)
                        num_batches += 1
                        # Compute F1 using scikit-learn
                        f1 = compute_batch_f1(batch_data, val_data, reconstruction, threshold=0.5)
                        f1_scores.append(f1)

                average_ndcg = total_ndcg / total_items if total_items > 0 else 0.0
                average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                evaluation_metrics = TrainingMetrics(avg_train_loss, avg_train_ndcg, average_ndcg, average_f1, None, None, None, None, None, None)
                self.training_metrics.append(evaluation_metrics)

            # Validation 2 - NDCG on validation set
            if val_loader is not None and epoch % 1 == 0:
                self.eval()
                val_loss = 0.0
                val_total_ndcg = 0.0
                val_total_items = 0
                val_f1_scores = []
                val_iou_scores = []
                with torch.no_grad():
                    for (batch_data_val, _, val_data_val) in val_loader:
                        batch_data_val = batch_data_val.to(self.device)
                        val_data_val = val_data_val.to(self.device)

                        # Forward pass
                        reconstruction_val = self.forward(batch_data_val)

                        # Validation loss (weighted, similar to training)
                        loss_val = criterion(reconstruction_val, val_data_val)
                        weights_val = torch.where(val_data_val > 0.5, positive_weight, 1.0)
                        weighted_loss_val = (loss_val * weights_val).mean()
                        val_loss += weighted_loss_val * batch_data_val.size(0)

                        # NDCG
                        ndcg_val_batch = compute_batch_ndcg(batch_data_val, val_data_val, reconstruction_val, k=1000)
                        val_total_ndcg += ndcg_val_batch * batch_data_val.size(0)

                        # F1
                        f1_val_batch = compute_batch_f1(batch_data_val, val_data_val, reconstruction_val, threshold=0.5)
                        val_f1_scores.append(f1_val_batch)

                        # IoU (Jaccard Index)
                        # Simple example: predictions above threshold 0.5
                        pred_mask_val = (reconstruction_val >= 0.5).float()
                        true_mask_val = (val_data_val >= 0.5).float()
                        intersection = (pred_mask_val * true_mask_val).sum()
                        union = (pred_mask_val + true_mask_val - pred_mask_val * true_mask_val).sum()
                        iou_val_batch = intersection / (union + 1e-7)
                        val_iou_scores.append(iou_val_batch.item())

                        val_total_items += batch_data_val.size(0)

                average_val_loss = val_loss / len(val_loader.dataset)
                average_val_ndcg = val_total_ndcg / val_total_items if val_total_items > 0 else 0.0
                average_val_f1 = sum(val_f1_scores) / len(val_f1_scores) if val_f1_scores else 0.0
                average_val_iou = sum(val_iou_scores) / len(val_iou_scores) if val_iou_scores else 0.0
                self.training_metrics[-1].val_loss = average_val_loss
                self.training_metrics[-1].val_average_ndcg = average_val_ndcg
                self.training_metrics[-1].val_average_f1 = average_val_f1
                self.training_metrics[-1].val_average_iou = average_val_iou
                print(f"Epoch {epoch}: {self.training_metrics[-1]}")

        return self.training_metrics

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            reconstruction = self.forward(X)
        return reconstruction

    def predict_for_user(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            reconstruction = self.forward_for_user(X)
        return reconstruction

def compute_batch_ndcg(batch_data, val_data, reconstruction, k=None):
    """
    Computes NDCG on the 'missing' entries indicated by mask = val_data - batch_data.
    """
    # mask > 0 indicates data that was not present in batch_data but present in val_data
    mask = ~batch_data.bool()

    # Predictions and ground truth only for masked positions
    pred = reconstruction[mask]
    true = val_data[mask]

    # If there are no masked entries, return 0.0 to avoid errors
    if pred.numel() == 0:
        return 0.0

    # Sort by predicted relevance in descending order
    sorted_indices = torch.argsort(pred, descending=True)
    sorted_true = true[sorted_indices]

    # If k is not specified or exceeds the number of items, use all available masked entries
    if (k is None) or (k > sorted_true.numel()):
        k = sorted_true.numel()

    # Calculate DCG
    dcg = 0.0
    for i in range(k):
        rel_i = sorted_true[i].item()
        dcg += (2**rel_i - 1) / math.log2(i + 2)  # i+2 because log2(1+1) for first item

    # Calculate IDCG by sorting the true relevances in descending order
    sorted_true_ideal = torch.sort(true, descending=True).values
    ideal_k = min(k, sorted_true_ideal.numel())
    idcg = 0.0
    for i in range(ideal_k):
        rel_i = sorted_true_ideal[i].item()
        idcg += (2**rel_i - 1) / math.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def compute_batch_f1(batch_data, val_data, reconstruction, threshold=0.5):
    """
    Computes the F1 score on the 'missing' entries indicated by mask = val_data - batch_data.
    Assumes binary relevance (predicted/ground_truth > threshold = 1, otherwise 0).
    """
    # Identify mask
    mask = ~batch_data.bool()

    # If nothing is masked, return 0.0 to avoid division by zero
    if not mask.any():
        return 0.0

    # Predicted labels and ground truth labels at masked positions
    pred = (reconstruction[mask] >= threshold).float()
    true = (val_data[mask] >= threshold).float()

    # Compute true positives, false positives, and false negatives
    tp = (pred * true).sum()
    fp = (pred * (1 - true)).sum()
    fn = ((1 - pred) * true).sum()

    # Compute precision and recall
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)

    # Compute F1
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return f1.item()

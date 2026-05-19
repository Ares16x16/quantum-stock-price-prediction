"""Sequence models for stronger local hybrid experiments.

These models are intentionally separate from the preserved HQNN-FSP circuit.
They are GPU-friendly PyTorch experiments used to explore whether a richer
classical temporal encoder paired with a quantum-inspired head yields stronger
directional forecasts on local hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from qsp.models.quantum_inspired import default_device


class AttentionPool(nn.Module):
    """Soft attention over sequence steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class BidirectionalLSTMBaseline(nn.Module):
    """A stronger classical sequence baseline for direction prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.pool = AttentionPool(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        pooled, _ = self.pool(encoded)
        return self.head(pooled).squeeze(-1)


class TemporalQQTNHybrid(nn.Module):
    """BiLSTM temporal encoder followed by a qutrit-inspired prediction head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        qutrit_dim: int = 24,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.pool = AttentionPool(hidden_dim * 2)
        self.pre_qutrit = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, qutrit_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(qutrit_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def qutrit_encode(x: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(x, 0.0, 1.0)
        theta = 0.5 * torch.pi * clipped
        phi = torch.pi * clipped
        amp0 = torch.cos(theta)
        amp1 = torch.sin(theta) * torch.cos(phi)
        amp2 = torch.sin(theta) * torch.sin(phi)
        return torch.cat([amp0, amp1, amp2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        pooled, _ = self.pool(encoded)
        projected = self.pre_qutrit(pooled)
        encoded_qutrit = self.qutrit_encode(projected)
        return self.head(encoded_qutrit).squeeze(-1)


@dataclass(frozen=True)
class SequenceTrainingHistory:
    model_name: str
    losses: list[float]
    train_losses: list[float]
    device: str


def train_sequence_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 40,
    learning_rate: float = 0.001,
    seed: int = 42,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    patience: int = 10,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    device: torch.device | None = None,
    balance_classes: bool = True,
) -> SequenceTrainingHistory:
    """Train a sequence classifier on time-ordered data."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = default_device() if device is None else device
    model = model.to(device)

    x_all = np.asarray(train_x, dtype=np.float32).copy()
    y_all = np.asarray(train_y, dtype=np.float32).copy().reshape(-1)
    split_idx = max(1, int(len(x_all) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(x_all) - 1) if len(x_all) > 2 else len(x_all)
    x_train = torch.as_tensor(x_all[:split_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_all[:split_idx], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_all[split_idx:], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_all[split_idx:], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    if balance_classes:
        positives = float(y_train.sum().item())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    train_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_count = 0

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []
        permutation = torch.randperm(len(x_train), device=device)
        for start in range(0, len(permutation), batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        model.eval()
        with torch.no_grad():
            if len(x_val) > 0:
                val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
            else:
                val_loss = train_loss

        train_losses.append(train_loss)
        losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return SequenceTrainingHistory(model.__class__.__name__, losses, train_losses, str(device))


def predict_sequence_classifier(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class labels and positive-class probabilities for sequence models."""

    device = next(model.parameters()).device if device is None else device
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features, dtype=np.float32).copy(), device=device)
        probabilities = torch.sigmoid(model(x)).cpu().numpy()
    return (probabilities >= 0.5).astype(int), probabilities


def train_sequence_regressor(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 40,
    learning_rate: float = 0.001,
    seed: int = 42,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    patience: int = 10,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    device: torch.device | None = None,
) -> SequenceTrainingHistory:
    """Train a sequence regressor on time-ordered data."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = default_device() if device is None else device
    model = model.to(device)

    x_all = np.asarray(train_x, dtype=np.float32).copy()
    y_all = np.asarray(train_y, dtype=np.float32).copy().reshape(-1)
    split_idx = max(1, int(len(x_all) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(x_all) - 1) if len(x_all) > 2 else len(x_all)
    x_train = torch.as_tensor(x_all[:split_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_all[:split_idx], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_all[split_idx:], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_all[split_idx:], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    train_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_count = 0

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []
        permutation = torch.randperm(len(x_train), device=device)
        for start in range(0, len(permutation), batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        model.eval()
        with torch.no_grad():
            if len(x_val) > 0:
                val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
            else:
                val_loss = train_loss

        train_losses.append(train_loss)
        losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return SequenceTrainingHistory(model.__class__.__name__, losses, train_losses, str(device))


def predict_sequence_regressor(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return raw regression predictions for sequence models."""

    device = next(model.parameters()).device if device is None else device
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features, dtype=np.float32).copy(), device=device)
        prediction = model(x).cpu().numpy()
    return prediction


__all__ = [
    "BidirectionalLSTMBaseline",
    "SequenceTrainingHistory",
    "TemporalQQTNHybrid",
    "predict_sequence_classifier",
    "predict_sequence_regressor",
    "train_sequence_classifier",
    "train_sequence_regressor",
]

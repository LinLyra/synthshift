"""Protocol-frozen training utilities for raw-sequence neural models."""

from __future__ import annotations

import hashlib
import io
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from synthshift.models.neural import build_neural_model


@dataclass(frozen=True)
class ChannelNormalization:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    best_epoch: int
    best_validation_macro_f1: float
    epochs_run: int
    elapsed_seconds: float
    history: tuple[dict[str, float], ...]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def channel_normalization(windows: np.ndarray) -> ChannelNormalization:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError("Expected windows with shape (rows, time, 6)")
    mean = values.mean(axis=(0, 1)).astype(np.float32)
    std = np.maximum(values.std(axis=(0, 1)), 1e-6).astype(np.float32)
    return ChannelNormalization(mean=mean, std=std)


def encode_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes.tolist())}
    try:
        return np.asarray([lookup[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown label: {error.args[0]}") from error


def normalized_tensors(
    windows: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    normalization: ChannelNormalization,
) -> TensorDataset:
    values = (np.asarray(windows, dtype=np.float32) - normalization.mean) / normalization.std
    values = np.ascontiguousarray(values.transpose(0, 2, 1))
    encoded = encode_labels(labels, classes)
    return TensorDataset(torch.from_numpy(values), torch.from_numpy(encoded))


def inverse_frequency_weights(encoded_labels: np.ndarray, classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(encoded_labels, dtype=np.int64), minlength=classes)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in the training subset")
    weights = 1.0 / counts.astype(np.float64)
    return (weights / weights.mean()).astype(np.float32)


def _loader(dataset: TensorDataset, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _macro_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for values, labels in loader:
            logits = model(values.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return float(f1_score(targets, predictions, average="macro", zero_division=0))


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss, total_rows = 0.0, 0
    for values, labels in loader:
        values, labels = values.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(values), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total_rows += len(labels)
    return total_loss / total_rows


def select_best_epoch(
    architecture: str,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    validation_windows: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    batch_size: int = 128,
    max_epochs: int = 50,
    patience: int = 8,
    minimum_improvement: float = 1e-4,
) -> SelectionResult:
    seed_everything(seed)
    normalization = channel_normalization(train_windows)
    train_data = normalized_tensors(train_windows, train_labels, classes, normalization)
    validation_data = normalized_tensors(
        validation_windows, validation_labels, classes, normalization
    )
    train_loader = _loader(train_data, batch_size, shuffle=True, seed=seed)
    validation_loader = _loader(validation_data, batch_size, shuffle=False, seed=seed)
    encoded = encode_labels(train_labels, classes)
    weights = torch.from_numpy(inverse_frequency_weights(encoded, len(classes))).to(device)
    model = build_neural_model(architecture, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    best_score, best_epoch, stale = -np.inf, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        loss = _train_epoch(model, train_loader, optimizer, loss_function, device)
        score = _macro_f1(model, validation_loader, device)
        history.append({"epoch": float(epoch), "train_loss": loss, "validation_macro_f1": score})
        if score > best_score + minimum_improvement:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return SelectionResult(
        best_epoch=best_epoch,
        best_validation_macro_f1=float(best_score),
        epochs_run=len(history),
        elapsed_seconds=time.perf_counter() - started,
        history=tuple(history),
    )


def refit_exact_epochs(
    architecture: str,
    windows: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[nn.Module, ChannelNormalization, tuple[float, ...]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(seed)
    normalization = channel_normalization(windows)
    dataset = normalized_tensors(windows, labels, classes, normalization)
    loader = _loader(dataset, batch_size, shuffle=True, seed=seed)
    encoded = encode_labels(labels, classes)
    weights = torch.from_numpy(inverse_frequency_weights(encoded, len(classes))).to(device)
    model = build_neural_model(architecture, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    losses = []
    for _ in range(epochs):
        losses.append(_train_epoch(model, loader, optimizer, loss_function, device))
    return model, normalization, tuple(losses)


def checkpoint_sha256(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

"""Higher-resolution Contextual QNN using multi-level return buckets.

This module extends the lightweight ContextualQNN idea from binary returns to
multi-level return quantization. The intended paper-aligned use here is `d=4`
with `T=2` and `tau=1`, which maps naturally to two qubits per symbol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiLevelContextualQNNConfig:
    context_length: int = 2
    horizon: int = 1
    num_levels: int = 4
    num_layers: int = 3
    num_assets: int = 1
    seed: int = 42
    shots: int | None = None
    spsa_perturbation: float = 0.01
    learning_rate: float = 0.1

    @property
    def symbol_bits(self) -> int:
        return int(math.ceil(math.log2(self.num_levels)))

    @property
    def num_qubits(self) -> int:
        return (self.context_length + self.horizon) * self.symbol_bits


def _symbol_to_bits(symbol: int, width: int) -> list[int]:
    return [int(bit) for bit in np.binary_repr(int(symbol), width=width)]


def _bits_to_symbol(bits: np.ndarray) -> int:
    symbol = 0
    for bit in bits.astype(int).tolist():
        symbol = (symbol << 1) | bit
    return symbol


def _basis_index(bits: np.ndarray) -> int:
    index = 0
    for bit in bits.astype(int).tolist():
        index = (index << 1) | bit
    return index


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.asarray([[c, -s], [s, c]], dtype=complex)


def _apply_one_qubit(state: np.ndarray, gate: np.ndarray, wire: int, num_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * num_qubits)
    moved = np.moveaxis(tensor, wire, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    restored = np.moveaxis(updated, 0, wire)
    return restored.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, num_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * num_qubits).copy()
    for index in np.ndindex(*([2] * num_qubits)):
        if index[control] == 1:
            flipped = list(index)
            flipped[target] ^= 1
            flipped = tuple(flipped)
            if index < flipped:
                tensor[index], tensor[flipped] = tensor[flipped], tensor[index]
    return tensor.reshape(-1)


class MultiLevelContextualQNN:
    """Small statevector Contextual QNN for multi-level return buckets."""

    def __init__(self, config: MultiLevelContextualQNNConfig = MultiLevelContextualQNNConfig()):
        if config.horizon != 1:
            raise ValueError("This multilevel scaffold supports tau=1 only.")
        if config.num_assets < 1:
            raise ValueError("num_assets must be positive.")
        if config.num_levels > 2 ** config.symbol_bits:
            raise ValueError("num_levels exceeds the representable symbol space.")
        self.config = config
        rng = np.random.default_rng(config.seed)
        shape = (config.num_layers, config.num_qubits)
        self.shared_params = rng.normal(0.0, 0.05, size=shape)
        self.asset_params = rng.normal(0.0, 0.05, size=(config.num_assets, *shape))
        self.rng = rng

    @property
    def num_parameters(self) -> int:
        return int(self.shared_params.size + self.asset_params.size)

    @property
    def circuit_depth_estimate(self) -> int:
        return int(self.config.num_layers * 4)

    def _flat_params(self) -> np.ndarray:
        return np.concatenate([self.shared_params.ravel(), self.asset_params.ravel()])

    def _set_flat_params(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        split = self.shared_params.size
        self.shared_params = values[:split].reshape(self.shared_params.shape)
        self.asset_params = values[split:].reshape(self.asset_params.shape)

    def _state(self, context: np.ndarray, asset_id: int = 0) -> np.ndarray:
        context = np.asarray(context, dtype=int).ravel()
        if len(context) != self.config.context_length:
            raise ValueError("context length does not match config.")
        if asset_id < 0 or asset_id >= self.config.num_assets:
            raise ValueError("asset_id is out of range.")

        bits = []
        for symbol in context:
            bits.extend(_symbol_to_bits(int(symbol), self.config.symbol_bits))
        bits.extend([0] * (self.config.horizon * self.config.symbol_bits))
        bits_array = np.asarray(bits, dtype=int)

        state = np.zeros(2 ** self.config.num_qubits, dtype=complex)
        state[_basis_index(bits_array)] = 1.0

        for layer in range(self.config.num_layers):
            for wire in range(self.config.num_qubits):
                state = _apply_one_qubit(state, _ry(self.shared_params[layer, wire]), wire, self.config.num_qubits)
            for wire in range(self.config.num_qubits - 1):
                state = _apply_cnot(state, wire, wire + 1, self.config.num_qubits)
            for wire in range(self.config.num_qubits):
                theta = self.asset_params[asset_id, layer, wire]
                state = _apply_one_qubit(state, _ry(theta), wire, self.config.num_qubits)
            for wire in range(self.config.num_qubits - 1, 0, -1):
                state = _apply_cnot(state, wire, wire - 1, self.config.num_qubits)
        return state

    def probability_distribution(self, context: np.ndarray, asset_id: int = 0, shots: int | None = None) -> np.ndarray:
        """Return P(next bucket = k) for every return bucket."""

        state = self._state(context, asset_id)
        probs = np.abs(state) ** 2
        output_probs = np.zeros(self.config.num_levels, dtype=float)
        for index, prob in enumerate(probs):
            bitstring = np.asarray(list(np.binary_repr(index, width=self.config.num_qubits)), dtype=int)
            output_bits = bitstring[-self.config.symbol_bits :]
            symbol = _bits_to_symbol(output_bits)
            if symbol < self.config.num_levels:
                output_probs[symbol] += float(prob)
        total = float(output_probs.sum())
        if total > 0:
            output_probs /= total
        shot_count = self.config.shots if shots is None else shots
        if shot_count:
            counts = self.rng.multinomial(shot_count, output_probs)
            output_probs = counts / shot_count
        return output_probs

    def predict_proba(self, contexts: np.ndarray, asset_ids: np.ndarray | None = None) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=int)
        if asset_ids is None:
            asset_ids = np.zeros(len(contexts), dtype=int)
        return np.asarray(
            [self.probability_distribution(context, int(asset)) for context, asset in zip(contexts, asset_ids)],
            dtype=float,
        )

    def predict(self, contexts: np.ndarray, asset_ids: np.ndarray | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba(contexts, asset_ids), axis=1).astype(int)

    def fidelity_loss(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        asset_ids: np.ndarray | None = None,
    ) -> float:
        contexts = np.asarray(contexts, dtype=int)
        targets = np.asarray(targets, dtype=int).ravel()
        if asset_ids is None:
            asset_ids = np.zeros(len(contexts), dtype=int)
        asset_ids = np.asarray(asset_ids, dtype=int).ravel()
        losses = []
        for asset_id in sorted(set(asset_ids.tolist())):
            asset_mask = asset_ids == asset_id
            asset_contexts = contexts[asset_mask]
            asset_targets = targets[asset_mask]
            for context in sorted({tuple(row.tolist()) for row in asset_contexts}):
                context_array = np.asarray(context, dtype=int)
                context_mask = np.all(asset_contexts == context_array, axis=1)
                target_dist = np.zeros(self.config.num_levels, dtype=float)
                for level in range(self.config.num_levels):
                    target_dist[level] = float(np.mean(asset_targets[context_mask] == level))
                pred_dist = self.probability_distribution(context_array, int(asset_id))
                fidelity = float(np.square(np.sum(np.sqrt(target_dist * pred_dist))))
                losses.append(1.0 - fidelity)
        return float(np.mean(losses)) if losses else 0.0

    def spsa_step(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        asset_ids: np.ndarray | None = None,
    ) -> float:
        base = self._flat_params()
        delta = self.rng.choice([-1.0, 1.0], size=base.shape)
        eps = self.config.spsa_perturbation

        self._set_flat_params(base + eps * delta)
        loss_plus = self.fidelity_loss(contexts, targets, asset_ids)
        self._set_flat_params(base - eps * delta)
        loss_minus = self.fidelity_loss(contexts, targets, asset_ids)

        grad = ((loss_plus - loss_minus) / (2.0 * eps)) * delta
        self._set_flat_params(base)
        current_loss = self.fidelity_loss(contexts, targets, asset_ids)
        self._set_flat_params(base - self.config.learning_rate * grad)
        return current_loss

    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        asset_ids: np.ndarray | None = None,
        epochs: int = 10,
    ) -> list[float]:
        losses = []
        for _ in range(epochs):
            losses.append(self.spsa_step(contexts, targets, asset_ids))
        return losses


__all__ = [
    "MultiLevelContextualQNN",
    "MultiLevelContextualQNNConfig",
]

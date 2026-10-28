"""Línea base analítica: dinámica del replicador sobre estrategias fijas.

Lee matrices de pagos guardadas en disco; no ejecuta el simulador. Las mide train.matriz_pagos.
A[i, j] es el retorno de la estrategia i contra la j y x es la fracción de población de cada una.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def load_payoff_matrix(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Lee una matriz de pagos (fila contra columna) y los nombres de sus estrategias."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["matrix"], dtype=float), tuple(data["modes"])


def _field(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """dx_i/dt = x_i ((Ax)_i - x·Ax); x puede ser un punto (K,) o un lote (n, K)."""
    payoff = x @ A.T
    return x * (payoff - (x * payoff).sum(axis=-1, keepdims=True))


def _step(A: np.ndarray, x: np.ndarray, dt: float) -> np.ndarray:
    """Un paso de Runge-Kutta de cuarto orden, proyectado al símplex."""
    k1 = _field(A, x)
    k2 = _field(A, x + dt / 2 * k1)
    k3 = _field(A, x + dt / 2 * k2)
    k4 = _field(A, x + dt * k3)
    x = np.clip(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0, None)
    return x / x.sum(axis=-1, keepdims=True)


def replicator_flow(A: np.ndarray, x0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """Integra dx_i/dt = x_i ((Ax)_i - x·Ax). Devuelve (steps + 1, K), con x0 en la fila 0."""
    x = np.asarray(x0, dtype=float)
    x = x / x.sum()
    path = np.empty((steps + 1, len(x)))
    path[0] = x
    for index in range(steps):
        x = _step(A, x, dt)
        path[index + 1] = x
    return path


def fixed_points(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Puntos de reposo: para cada soporte, la mezcla que iguala los pagos de sus estrategias.

    Incluye los vértices. Omite los soportes cuyo sistema es singular (continuos de equilibrios).
    """
    size = len(A)
    points = []
    for count in range(1, size + 1):
        for support in itertools.combinations(range(size), count):
            block = A[np.ix_(support, support)]
            system = np.zeros((count + 1, count + 1))
            system[:count, :count] = block
            system[:count, count] = -1
            system[count, :count] = 1
            rhs = np.zeros(count + 1)
            rhs[count] = 1
            try:
                solution = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                continue
            if np.any(solution[:count] < -tol):
                continue
            x = np.zeros(size)
            x[list(support)] = np.clip(solution[:count], 0, None)
            if not any(np.allclose(x, other, atol=1e-7) for other in points):
                points.append(x)
    return np.array(points)


def is_stable(A: np.ndarray, x: np.ndarray, tol: float = 1e-9) -> bool:
    """Estabilidad asintótica: autovalores del jacobiano del replicador sobre el símplex."""
    size = len(x)
    payoff = A @ x
    mean = x @ payoff
    jacobian = np.diag(payoff - mean) + x[:, None] * (A - payoff[None, :] - (A.T @ x)[None, :])
    tangent = np.linalg.qr(np.eye(size)[:, : size - 1] - np.ones((size, size - 1)) / size)[0]
    eigenvalues = np.linalg.eigvals(tangent.T @ jacobian @ tangent)
    return bool(np.all(eigenvalues.real < -tol))


def basin_shares(
    A: np.ndarray, n_starts: int = 1000, steps: int = 4000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cuota de condiciones iniciales aleatorias que terminan dominadas por cada estrategia.

    Devuelve (cuotas (K,), mezcla final media (K,)). El tiempo se reescala con la magnitud de A,
    porque sumar una constante o multiplicar A solo cambia la velocidad del flujo.
    """
    scaled = A / max(np.abs(A).max(), 1e-12)
    rng = np.random.default_rng(seed)
    x = rng.dirichlet(np.ones(len(A)), size=n_starts)
    for _ in range(steps):
        x = _step(scaled, x, 0.2)
    shares = np.bincount(x.argmax(axis=1), minlength=len(A)) / n_starts
    return shares, x.mean(axis=0)


def critical_mass(A: np.ndarray, invader: int, resident: int) -> float:
    """Fracción de invasores a partir de la cual el invasor rinde más que el residente.

    Se calcula en el juego de las dos estrategias: q = (A_rr - A_ir) / ((A_rr - A_ir) + (A_ii -
    A_ri)). Vale 0.0 si el invasor rinde más incluso siendo minoría y 1.0 si nunca lo logra.
    """
    stay = A[resident, resident] - A[invader, resident]
    gain = A[invader, invader] - A[resident, invader]
    if stay <= 0:
        return 0.0
    if gain <= 0:
        return 1.0
    return float(stay / (stay + gain))


def critical_mass_interval(
    A: np.ndarray,
    se: np.ndarray,
    invader: int,
    resident: int,
    n: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mediana y rango del 5% al 95% de la masa crítica al propagar el error de cada pago."""
    rng = np.random.default_rng(seed)
    draws = [critical_mass(A + rng.normal(size=A.shape) * se, invader, resident) for _ in range(n)]
    low, median, high = np.percentile(draws, [5, 50, 95])
    return float(median), float(low), float(high)

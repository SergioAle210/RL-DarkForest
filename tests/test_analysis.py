"""Métricas calculables sin ejecutar el entorno."""

import numpy as np

from analysis.metrics import summarize


def test_summary_uses_executed_actions_and_excludes_dead_agents():
    """Comprueba causas, costos y conteos sobre una trayectoria mínima conocida."""
    trajectory = {
        "alive": np.array([[True, True], [True, False], [True, False]]),
        "emission": np.array([[0, 0], [1, 0], [0, 0]]),
        "info_death_cause": np.array([[0, 2], [0, 0]]),
        "info_executed_kind": np.array([[3, 0], [4, -1]]),
        "info_preventive_attacks": np.array([1, 0]),
        "actions": np.array([[3, 0], [7, 3]]),
        "rewards": np.array([[0.2, -0.6], [0.1, 0.0]]),
    }
    report = summarize(trajectory)
    assert report["survival"] == 0.5
    assert report["death_attack"] == 0.5
    assert report["low_emission"] == 0.5
    assert report["symbols"] == report["shared_measurements"] == 1
    assert report["preventive_attacks"] == 1

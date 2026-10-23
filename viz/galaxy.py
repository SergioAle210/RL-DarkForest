"""Animación de la galaxia: conos de luz, proyectiles y estallidos. Solo lee trayectorias."""

from __future__ import annotations

from pathlib import Path


def render_episode(traj, out: Path, fps: int = 20) -> None:
    """Guarda mp4/gif de un episodio."""
    raise NotImplementedError


def render_with_phase_diagram(traj, grid, out: Path) -> None:
    """Demo: galaxia a la izquierda, diagrama de fases llenándose a la derecha."""
    raise NotImplementedError

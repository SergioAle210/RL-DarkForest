"""Animación de la galaxia: conos de luz, proyectiles y estallidos. Solo lee trayectorias."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from darkforest.trajectories import load_episode


def episode_animation(traj, metadata: dict, fps: int = 8) -> FuncAnimation:
    """Dibuja estados y eventos guardados sin llamar al simulador."""
    fig, (galaxy, resources) = plt.subplots(1, 2, figsize=(10, 4))
    n = traj["alive"].shape[1]
    colors = plt.get_cmap("tab10")(np.arange(n))
    steps = traj["actions"].shape[0]

    def draw(frame: int):
        """Representa un estado y el historial disponible hasta ese instante."""
        galaxy.clear()
        resources.clear()
        positions, alive = traj["pos"][frame], traj["alive"][frame]
        time = int(traj["t"][frame])
        for agent in range(n):
            galaxy.scatter(
                *positions[agent], color=colors[agent], marker="o" if alive[agent] else "x", s=80
            )
            galaxy.text(*positions[agent], f"  C{agent + 1}")
            resources.plot(
                traj["t"][: frame + 1],
                traj["resources"][: frame + 1, agent],
                color=colors[agent],
                label=f"C{agent + 1}",
            )
        for index in np.flatnonzero(traj["sig_valid"][frame]):
            origin = int(traj["sig_origin"][frame, index])
            radius = metadata["signal_speed"] * (time - traj["sig_t"][frame, index])
            explicit = traj["sig_content"][frame, index, 0] >= 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    galaxy.add_patch(
                        Circle(
                            positions[origin] + [dx, dy],
                            radius,
                            fill=False,
                            color=colors[origin],
                            alpha=0.4 if explicit else 0.08,
                        )
                    )
        for index in np.flatnonzero(traj["prj_valid"][frame]):
            origin = int(traj["prj_origin"][frame, index])
            delta = (traj["prj_target"][frame, index] - positions[origin] + 0.5) % 1 - 0.5
            length = np.linalg.norm(delta)
            elapsed = time - traj["prj_t"][frame, index]
            fraction = min(metadata["proj_speed"] * elapsed / max(length, 1e-9), 1)
            point = (positions[origin] + delta * fraction) % 1
            galaxy.scatter(*point, marker=">", color=colors[origin], s=35)
        if frame > 0 and traj["info_bursts"][frame - 1]:
            theta = traj["theta"][frame]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    galaxy.add_patch(
                        Circle(theta[2:4] + [dx, dy], theta[4], color="red", alpha=0.12)
                    )
        galaxy.set(
            xlim=(0, 1),
            ylim=(0, 1),
            aspect="equal",
            title=f"Turno {time} · vivos {alive.sum()}/{n}",
        )
        resources.set(
            xlim=(0, max(steps, 1)),
            ylim=(0, max(float(traj["resources"].max()) * 1.1, 1)),
            xlabel="Turno",
            ylabel="Recursos",
            title="Recursos por civilización",
        )
        resources.legend(loc="upper left")
        fig.suptitle(f"{metadata['policy']} · seed {metadata['seed']} · episodio guardado")
        fig.tight_layout()
        return []

    return FuncAnimation(
        fig, draw, frames=steps + 1, interval=1000 / fps, repeat=False, cache_frame_data=False
    )


def render_episode(traj, out: Path, fps: int = 20, *, metadata: dict | None = None) -> None:
    """Guarda un GIF; MP4 requiere un escritor ffmpeg instalado."""
    if metadata is None:
        if isinstance(traj, (str, Path)):
            traj, metadata = load_episode(traj)
        else:
            raise ValueError("metadata: se requieren velocidades y procedencia del episodio")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation = episode_animation(traj, metadata, fps)
    try:
        animation.save(str(out), writer=PillowWriter(fps=fps) if out.suffix == ".gif" else "ffmpeg")
    finally:
        plt.close(animation._fig)


def render_with_phase_diagram(traj, grid, out: Path) -> None:
    """Demo: galaxia a la izquierda, diagrama de fases llenándose a la derecha."""
    raise NotImplementedError


def main() -> None:
    """Reproduce una trayectoria en disco y exporta su animación."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps debe ser positivo")
    output = args.out or args.episode.parent.parent / "videos" / f"{args.episode.stem}.gif"
    render_episode(args.episode, output, args.fps)
    print(f"Animación guardada: {output}")


if __name__ == "__main__":
    main()

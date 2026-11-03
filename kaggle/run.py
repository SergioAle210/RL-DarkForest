"""Kernel de Kaggle: copia el código del dataset, entrena con IPPO y evalúa el checkpoint.

Variables opcionales: DF_CONFIG (por defecto configs/costo_ignorar.yaml), DF_STEPS (pasos
totales), DF_SEEDS (lista separada por espacios, por defecto 0 1 2 3 4) y DF_NAME (nombre de la
corrida). Las corridas quedan en /kaggle/working/runs.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = next(Path("/kaggle/input").rglob("pyproject.toml")).parent
REPO = Path("/kaggle/working/repo")
config = os.environ.get("DF_CONFIG", "configs/costo_ignorar.yaml")
steps = os.environ.get("DF_STEPS", "10485760")
name = os.environ.get("DF_NAME", "costo_ignorar")
seeds = os.environ.get("DF_SEEDS", "0 1 2 3 4").split()


def sh(*cmd: str) -> None:
    """Ejecuta un comando en el repo y falla si termina con error."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


shutil.copytree(SRC, REPO, dirs_exist_ok=True)
sh(sys.executable, "-m", "pip", "install", "-q", "flax", "optax", "pyyaml", "tensorboard")
if shutil.which("nvidia-smi"):
    sh(sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]")
sh(sys.executable, "-c", "import jax; print(jax.devices())")
for seed in seeds:
    sh(
        sys.executable,
        "-m",
        "train.ippo",
        "--config",
        config,
        "--name",
        name,
        f"train.total_steps={steps}",
        f"train.seed={seed}",
    )
for ckpt in sorted(REPO.glob(f"runs/{name}-*/*/checkpoints/final.msgpack")):
    sh(sys.executable, "-m", "train.evaluar_checkpoint", "--checkpoint", str(ckpt), "--baselines")
shutil.copytree(REPO / "runs", "/kaggle/working/runs", dirs_exist_ok=True)
shutil.rmtree(REPO)

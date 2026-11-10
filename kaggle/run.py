"""Kernel de Kaggle, corrida final: imitación de coop_real, ajuste fino con RL y barrido λ×c.

Apuesta: en el mundo de refugio caro (cost_ref 0.25, ruido 0.3) las redes entrenadas desde cero
colapsan a ignorar, mientras las estrategias fijas que cooperan rinden 2 a 3 puntos más. Se
clona a coop_real con DAgger (train.imitar) y se afina con RL y población (pesos de partida
train.init_from). Etapas, de más a menos importante:

0. imitación por celda (λ, c); 1. ajuste fino por celda (H2, H1, H5); 2. sin imitación, 50 M pasos.

Cada etapa entrena, resume sus checkpoints y escribe todo directamente en OUT, de modo que una
cancelación o el interruptor de tiempo conservan lo ya hecho. DF_DRY=1 imprime los comandos sin
ejecutarlos; DF_OUT y DF_REPO cambian las rutas.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

START = time.monotonic()
# Interruptor: pasadas LIMIT_H horas no se lanza nada nuevo y todo comando en curso se mata. El
# entrenamiento se corta 30 minutos antes para dejar tiempo a los resúmenes.
LIMIT_H = float(os.environ.get("DF_LIMIT_H", "5"))
HARD_S = LIMIT_H * 3600
TRAIN_S = HARD_S - 1800
DRY = os.environ.get("DF_DRY") == "1"

OUT = Path(os.environ.get("DF_OUT", "/kaggle/working"))
REPO = Path(os.environ.get("DF_REPO", "/kaggle/working/repo"))
SRC = Path(".") if DRY else next(Path("/kaggle/input").rglob("pyproject.toml")).parent
RUNS = OUT / "runs"

SEEDS = [0, 1, 2, 3, 4]
STEPS = "3211264"  # 49 iteraciones: la imitación ya deja la política cerca de lo bueno
WORLD = ["env.anomaly.noise_std=0.3", "env.cost_ref=0.25"]
LAMBDAS = ["0.25", "0.5", "1.0"]
SPEEDS = ["0.05", "0.22", "1.0"]
CENTER = ("0.5", "0.22")
POP = "configs/pob_socios.yaml"
LOGGING = [f"logging.out_dir={RUNS}"]
EXPLOIT = [
    "train.self_play.rho=1.0",
    "train.self_play.w_snapshot=1.0",
    "train.self_play.w_recip=0.0",
    "train.self_play.w_coop_real=0.0",
    "train.self_play.w_ataque_solo=0.0",
    "train.self_play.w_solo=0.0",
    "train.self_play.w_ignorar=0.0",
]
FIJOS = ["fijo:coop_real", "fijo:recip", "fijo:solo", "fijo:ataque_solo", "fijo:ignorar"]


def tag(lam: str, speed: str) -> str:
    """Identificador de una celda del barrido, sin puntos para los nombres de corrida."""
    return f"l{lam}_c{speed}".replace(".", "p")


def cell_over(lam: str, speed: str) -> list[str]:
    """Overrides del mundo caro en la celda (λ, c)."""
    return [*WORLD, f"env.anomaly.lethality={lam}", f"env.signal_speed={speed}"]


def imitation_dir(lam: str, speed: str) -> Path:
    """Carpeta con los pesos clonados de una celda."""
    return OUT / "imitacion" / tag(lam, speed)


def finetune(lam: str, speed: str, seeds: list[int] = SEEDS) -> dict:
    """Ajuste fino con RL y población mixta desde la red clonada de la celda."""
    return {
        "name": f"ft_{tag(lam, speed)}",
        "config": POP,
        "over": cell_over(lam, speed),
        "seeds": seeds,
        "steps": STEPS,
        "init_file": str(imitation_dir(lam, speed) / "checkpoints" / "final.msgpack"),
    }


# Segunda corrida final. Las 9 celdas de λ en {0.25, 0.5, 1.0} ya se midieron (docs/
# DIAGNOSTICO_DISENO.md); aquí se agregan λ = 0 (prueba de cordura de H1 con redes) y λ = 0.75 y
# se prueba escapar del colapso sin imitación con 50 M pasos, con y sin más entropía.
NEW_LAMBDAS = ["0.0", "0.75"]
new_cells = [(lam, speed) for lam in NEW_LAMBDAS for speed in SPEEDS]
SCRATCH_STEPS = "52428800"
PLAN = [
    {"imitar": new_cells, "train": []},
    {"imitar": [], "train": [finetune(lam, speed) for lam, speed in new_cells]},
    {
        "imitar": [],
        "train": [
            {
                "name": "largo_cero",
                "config": POP,
                "over": cell_over(*CENTER),
                "seeds": [0, 1, 2],
                "steps": SCRATCH_STEPS,
            },
            {
                "name": "largo_cero_ent",
                "config": POP,
                "over": [*cell_over(*CENTER), "train.ent_coef=0.1", "train.ent_coef_final=0.01"],
                "seeds": [0, 1, 2],
                "steps": SCRATCH_STEPS,
            },
        ],
    },
]


# DF_ONLY="n6" (nombres de trabajos separados por comas) repite solo esos trabajos con la
# imitación de la celda central, sin las demás etapas.
ONLY = [name for name in os.environ.get("DF_ONLY", "").split(",") if name]
if ONLY:
    PLAN = [
        {
            "imitar": [CENTER],
            "train": [j for st in PLAN for j in st["train"] if j["name"] in ONLY],
        }
    ]


def remaining(limit: float) -> float:
    """Segundos que quedan antes del límite indicado, contados desde el inicio del kernel."""
    return limit - (time.monotonic() - START)


def gpu_count() -> int:
    """Cuenta las GPU visibles; sin nvidia-smi usa una sola unidad de cómputo."""
    try:
        text = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, sum(line.startswith("GPU") for line in text.splitlines()))
    except OSError:
        return 1


def sh(gpu: int, log: Path, *cmd: str, limit: float = HARD_S) -> int:
    """Ejecuta un comando del repo en una GPU; devuelve 124 si no empieza o se pasa del límite."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log.parent.mkdir(parents=True, exist_ok=True)
    left = remaining(limit)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        if DRY:
            print("DRY", " ".join(cmd), flush=True)
            return 0
        if left < 60:
            handle.write("saltado: se agotó el tiempo\n")
            return 124
        handle.flush()
        try:
            return subprocess.run(
                cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=left
            ).returncode
        except subprocess.TimeoutExpired:
            handle.write("cortado: se agotó el tiempo\n")
            return 124


def run_pool(tasks, gpus: int) -> None:
    """Reparte las tareas entre las GPU; cada tarea recibe el índice de la GPU que ocupa."""
    free: queue.Queue = queue.Queue()
    for index in range(gpus):
        free.put(index)

    def work(task) -> None:
        """Toma una GPU libre, ejecuta la tarea y la devuelve a la cola."""
        gpu = free.get()
        try:
            task(gpu)
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(max_workers=gpus) as pool:
        list(pool.map(work, tasks))


def final_of(name: str, seed: int) -> Path | None:
    """Checkpoint final de la corrida name con esa seed, si existe."""
    found = sorted(RUNS.glob(f"{name}-*/{seed}/checkpoints/final.msgpack"))
    return found[0] if found else None


def imitate_unit(lam: str, speed: str):
    """Tarea que clona a coop_real con DAgger en el mundo de la celda."""

    def task(gpu: int) -> None:
        """Escribe los pesos clonados en OUT/imitacion/<celda>."""
        code = sh(
            gpu,
            OUT / "logs" / f"imitar_{tag(lam, speed)}.log",
            sys.executable,
            "-m",
            "train.imitar",
            "--config",
            POP,
            *[arg for item in cell_over(lam, speed) for arg in ("--set", item)],
            "--out",
            str(imitation_dir(lam, speed)),
            "--episodes",
            "256",
            "--steps",
            "400",
            "--batch",
            "32",
            "--dagger",
            "4",
            "--dagger-episodes",
            "128",
            "--rare-power",
            "0.5",
            limit=TRAIN_S,
        )
        print(f"imitar {tag(lam, speed)} GPU {gpu}: código {code}", flush=True)

    return task


def train_unit(job: dict, seed: int):
    """Tarea que entrena una seed de un trabajo, con un reintento."""

    def task(gpu: int) -> None:
        """Entrena y deja el código de salida en el log."""
        log = OUT / "logs" / f"{job['name']}-s{seed}.log"
        extra = []
        if "init_file" in job:
            if not DRY and not Path(job["init_file"]).exists():
                print(f"{job['name']} seed {seed}: falta {job['init_file']}", flush=True)
                return
            extra.append(f"train.init_from={job['init_file']}")
        if "init_from" in job:
            start = final_of(job["init_from"], seed)
            if start is None and not DRY:
                print(f"{job['name']} seed {seed}: falta la red {job['init_from']}", flush=True)
                return
            extra.append(f"train.init_from={start}")
        if "partner_from" in job:
            partner = final_of(job["partner_from"], seed)
            if partner is None and not DRY:
                print(f"{job['name']} seed {seed}: falta la red {job['partner_from']}", flush=True)
                return
            extra.append(f"train.partner_checkpoint={partner}")
        for _attempt in range(2):
            code = sh(
                gpu,
                log,
                sys.executable,
                "-m",
                "train.ippo",
                "--config",
                job["config"],
                "--name",
                job["name"],
                f"train.total_steps={job['steps']}",
                f"train.seed={seed}",
                *LOGGING,
                *extra,
                *job["over"],
                limit=TRAIN_S,
            )
            print(f"{job['name']} seed {seed} GPU {gpu}: código {code}", flush=True)
            if code in (0, 124):
                break

    return task


def summary_unit(job: dict, seed: int):
    """Tarea que resume el checkpoint final de una corrida en JSON, con 200 episodios."""

    def task(gpu: int) -> None:
        """Escribe el resumen; un fallo queda en el log y no detiene el resto."""
        final = final_of(job["name"], seed)
        if final is None:
            return
        out = OUT / "resumenes" / job["name"] / f"seed{seed}.json"
        code = sh(
            gpu,
            OUT / "logs" / "resumenes.log",
            sys.executable,
            "-m",
            "train.resumen_corrida",
            "--checkpoint",
            str(final),
            "--episodes",
            "200",
            "--out",
            str(out),
        )
        print(f"resumen {job['name']} seed {seed}: código {code}", flush=True)

    return task


def cross_unit(job: dict):
    """Tarea que cruza las redes greedy de las seeds 0 a 2 con las estrategias fijas."""

    def task(gpu: int) -> None:
        """Escribe la matriz de cruce de un trabajo de 2 civilizaciones."""
        nets = [f"red:{p}:greedy" for p in (final_of(job["name"], s) for s in (0, 1, 2)) if p]
        if not nets:
            return
        out = OUT / "resumenes" / f"cruce_{job['name']}.json"
        code = sh(
            gpu,
            OUT / "logs" / "cruce.log",
            sys.executable,
            "-m",
            "train.cruce",
            "--config",
            job["config"],
            "--episodes",
            "200",
            "--out",
            str(out),
            *[arg for item in job["over"] for arg in ("--set", item)],
            *FIJOS,
            *nets,
        )
        print(f"cruce {job['name']}: código {code}", flush=True)

    return task


def main() -> None:
    """Ejecuta las etapas en orden y resume cada una al terminar."""
    if not DRY:
        shutil.copytree(SRC, REPO, dirs_exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "flax",
                "optax",
                "pyyaml",
                "tensorboard",
            ],
            check=True,
        )
    gpus = 1 if DRY else gpu_count()
    print(f"GPU visibles: {gpus}", flush=True)
    for stage in PLAN:
        run_pool([imitate_unit(*cell) for cell in stage["imitar"]], gpus)
        run_pool([train_unit(j, s) for j in stage["train"] for s in j["seeds"]], gpus)
        run_pool([summary_unit(j, s) for j in stage["train"] for s in j["seeds"]], gpus)
        two = [j for j in stage["train"] if "env.n_civs=6" not in j["over"]]
        run_pool([cross_unit(j) for j in two], gpus)
    (OUT / "plan.json").write_text(json.dumps(PLAN, indent=2, default=str), encoding="utf-8")
    if not DRY:
        shutil.rmtree(REPO)


if __name__ == "__main__":
    main()

# AGENTS.md

Guía para agentes de código (Claude Code, Codex, etc.) y para humanos. `CLAUDE.md` apunta a este archivo.

## Qué es esto
MARL sobre un POSG en JAX. Especificación completa en `docs/PROPUESTA.md`. Hipótesis H1–H5 en la sección 4.

## Comandos
```bash
uv sync                   # instalar (crea .venv; dependencias en pyproject.toml, lock en uv.lock)
uv run pytest             # cualquier comando va con `uv run`
uv run ruff check .       # lint (y `uv run ruff format .` para formatear)
make test                 # pytest
make lint                 # ruff check + format --check
make sanity               # entrena la config H1 (lambda = 0)
make sweep-coarse         # barrido 3x3
```

## Mapa de módulos y contratos
| Módulo | Responsabilidad | Regla |
|---|---|---|
| `darkforest/config.py` | Dataclasses de configuración y carga de YAML | Única fuente de hiperparámetros; nada hardcodeado |
| `darkforest/types.py` | `EnvState`, `Obs`, `Action`, `StepInfo` | Estructuras inmutables (`flax.struct`) |
| `darkforest/env.py` | `reset(key)` / `step(key, state, actions)` puros | Sin efectos secundarios; compatible con `jit`/`vmap` |
| `darkforest/anomaly.py` | θ, proyecciones `h_i(θ)`, estallidos | θ nunca identificable con una proyección |
| `darkforest/policies.py` | Protocolo `Policy` (`init_carry`, `act`) | Toda política (red, arquetipo, aleatoria) lo implementa |
| `darkforest/archetypes.py` | Heurísticas y modelo nulo | Solo dependen de `Policy` y `Obs` |
| `darkforest/networks.py` | Actor-crítico GRU | Sin lógica de entorno |
| `train/` | IPPO, población de autojuego, barrido | No importa nada de `analysis/` ni `viz/` |
| `analysis/` | Métricas, replicador, diagrama de fases | Lee trayectorias/resultados en disco, no corre el entorno en vivo |
| `viz/` | Animación | Solo lectura de trayectorias |

Dependencias permitidas: `darkforest` ← `train`, `analysis`, `viz`. Nunca al revés.

## Reglas del proyecto (RL y reproducibilidad)
- **Aleatoriedad:** todo `jax.random.PRNGKey` se pasa explícito y se divide con `split`. Prohibido `np.random` global.
- **Pureza:** funciones del entorno sin estado global ni `print`; formas estáticas para `jit`.
- **Recompensa:** solo `log(1+r_{t+1}) - log(1+r_t)`. Cualquier otro término viola el criterio de no imposición (sección 5). Hay un test que lo audita.
- **Seeds:** ≥ 5 por configuración reportada. Cada corrida guarda su config resuelta y su seed.
- **Configs:** una corrida = un YAML (`configs/`) + overrides `clave=valor` por CLI. Se guarda la config resuelta en el directorio de la corrida.
- **Checkpoints y resultados** en `runs/<config>/<seed>/` (ignorado por git).
- **Tests antes de entrenar:** invariantes del entorno (recursos ≥ 0, muertos no actúan, señales viajan a `c`), H1 como prueba de cordura.
- **Estilo:** `ruff` (línea 100), tipado en firmas públicas, docstrings cortos. Sin comentarios que repitan el código.
- **Commits:** pequeños, un cambio lógico; no commitear `runs/`, `wandb/` ni datos.

## Cómo añadir cosas
- **Nuevo agente/arquetipo:** clase en `archetypes.py` que implemente `Policy`; regístrala en `ARCHETYPES`.
- **Nueva acción:** ampliar `Action` en `types.py`, el despacho en `env.py`, `n_actions` en `config.py` y los tests.
- **Nueva métrica:** función pura en `analysis/metrics.py` registrada en `METRICS`.
- **Nuevo parámetro de barrido:** campo en `config.py` + entrada en `configs/sweep.yaml`.

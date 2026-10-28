# AGENTS.md

Guía para agentes de código (Claude Code, Codex, etc.) y para humanos. `CLAUDE.md` apunta a este archivo.

## Qué es esto
MARL sobre un POSG en JAX. Especificación completa en `docs/PROPUESTA.md`. Hipótesis H1–H5 en la sección 4.

## Comandos

```bash
uv sync --locked          # dependencias fijadas en uv.lock
uv sync --locked --group notebooks
uv run pytest             # cualquier comando va con `uv run`
uv run ruff check .       # lint (y `uv run ruff format .` para formatear)
uv run --group notebooks jupyter lab notebooks/fundamentos.ipynb
uv run python -m train.ippo --config configs/aprendizaje_dos.yaml
uv run --group notebooks jupyter lab notebooks/demo.ipynb
uv run python -m train.valor_cooperacion --world periodo_aleatorio  # diagnóstico: aporte de cooperar
uv run python -m train.matriz_pagos --lethality 0                   # diagnóstico: matriz de pagos
uv run python -m train.pagos_rejilla --out-dir runs/pagos       # matrices de pagos por celda
uv run python -m train.medir_entrenamiento --config configs/base.yaml --n-envs 1024
uv run python -m train.evaluar_checkpoint --checkpoint <ruta> --baselines
make test                 # pytest
make lint                 # ruff check + format --check
make sanity               # entrena la config H1 (lambda = 0)
make sweep-coarse         # barrido 3x3
```

`make sanity` y el ejecutor del barrido requieren completar el autojuego y los experimentos.
`--dry-run` expande combinaciones y construye la configuración de cada corrida; marca las
inválidas, cuenta los totales y termina con error si hay alguna. Todas las celdas de los YAML
del barrido son válidas y una prueba lo comprueba.
Los entrenamientos de comprobación usan `aprendizaje_dos.yaml` y `aprendizaje_cuatro.yaml`,
con `rho=0`. El contrato operativo y la entrega al equipo están en `docs/EPISODIOS.md`.

## Mapa de módulos y contratos

| Módulo | Responsabilidad | Regla |
|---|---|---|
| `darkforest/config.py` | Dataclasses de configuración y carga de YAML | Única fuente de hiperparámetros; nada hardcodeado |
| `darkforest/types.py` | `EnvState`, `Obs`, `StepInfo` | Estructuras inmutables (`flax.struct`); acciones como arrays enteros |
| `darkforest/env.py` | `reset(key)` / `step(key, state, actions)` puros | Sin efectos secundarios; compatible con `jit`/`vmap` |
| `darkforest/anomaly.py` | θ, proyecciones `h_i(θ)`, estallidos | θ nunca identificable con una proyección |
| `darkforest/policies.py` | Protocolo `Policy` (`init_carry`, `act`) | Toda política (red, arquetipo, aleatoria) lo implementa |
| `darkforest/archetypes.py` | Heurísticas y modelo nulo | Usan `Obs`, carry y configuración del catálogo; nunca `EnvState` |
| `darkforest/networks.py` | Actor-crítico GRU | Sin lógica de entorno |
| `train/` | IPPO, población de autojuego, barrido, diagnósticos con políticas fijas | No importa nada de `analysis/` ni `viz/` |
| `analysis/` | Métricas, replicador, diagrama de fases | Lee trayectorias/resultados en disco, no corre el entorno en vivo |
| `viz/` | Animación | Solo lectura de trayectorias |

Dependencias permitidas: `darkforest` ← `train`, `analysis`, `viz`. Nunca al revés.

## Reglas del proyecto (RL y reproducibilidad)

- **Aleatoriedad:** todo `jax.random.PRNGKey` se pasa explícito y se divide con `split`. Prohibido `np.random` global.
- **Pureza:** funciones del entorno sin estado global ni `print`; formas estáticas para `jit`.
- **Recompensa:** solo `log(1+r_{t+1}) - log(1+r_t)`. Cualquier otro término viola el criterio de no imposición (sección 5). Exigir una prueba con aserciones reales antes de entrenar.
- **Seeds:** ≥ 5 por configuración reportada. Cada corrida guarda su config resuelta y su seed.
- **Configs:** una corrida = un YAML (`configs/`) + overrides `clave=valor` por CLI. Se guarda la config resuelta en el directorio de la corrida.
- **Checkpoints y resultados** en `runs/<config>/<seed>/` (ignorado por git).
- **Tests antes de entrenar:** invariantes del entorno (recursos ≥ 0, muertos no actúan, señales viajan a `c`). Evaluar H1 empíricamente por separado; no modificar la recompensa para forzarla.
- **Estilo:** `ruff` (línea 100), tipado en firmas públicas. Comentarios breves, útiles y en español, sin separadores `===` o `---`. Docstrings cortos entre `"""`, inmediatamente después de declarar la función. Las explicaciones extensas van en documentación o Markdown del notebook.
- **Commits:** pequeños, con conventional commits y un cambio lógico; no commitear `runs/`, `wandb/` ni datos.

## Documentación y avance

- Documentación en `docs/`; README, AGENTS y CLAUDE permanecen en la raíz.
- `docs/TODO.md` es la única fuente de avance y asignaciones. Contiene solo título, fases y tablas con `tarea`, `responsable`, `estado`, `fecha`.
- Estados permitidos: `listo`, `pendiente`, `en proceso`. La fecha `YYYY-MM-DD` cambia únicamente con un cambio real de estado; no es una fecha límite.
- Las tareas futuras sin asignar mantienen responsable, estado y fecha vacíos. El equipo tiene cinco integrantes en total; Sergio realiza los fundamentos comunes.
- `docs/TAREAS.md` mantiene especificaciones, dependencias y criterios por ID, sin estados ni asignaciones personales duplicadas.
- No marcar una funcionalidad lista por preparar su archivo. Pruebas futuras: solo docstring de módulo, sin funciones vacías. No contar `xfail` o pruebas sin aserciones como evidencia funcional.
- Los notebooks importan módulos sin duplicar lógica y se versionan sin salidas. Usar nombres naturales: `fundamentos.ipynb`, `entrenamiento.ipynb`, `resultados.ipynb`, `demo.ipynb`.
- Guardar la copia ejecutada de fundamentos en `runs/notebook_checks/`; los ejemplos sintéticos deben identificarse y no fijan dimensiones físicas pendientes.

## Cómo añadir cosas

- **Nuevo agente/arquetipo:** clase en `archetypes.py` que implemente `Policy`; regístrala en `ARCHETYPES`.
- **Nueva acción:** actualizar `ACTIONS` en `config.py`, la codificación y el despacho de `env.py`, `EnvConfig.n_discrete`, los contratos y los tests. Distinguir las siete familias (`env.n_actions`) del espacio parametrizado (`n_discrete`); no existe una clase `Action`.
- **Nueva métrica:** función pura en `analysis/metrics.py` registrada en `METRICS`.
- **Nuevo parámetro de barrido:** campo en `config.py` + entrada en `configs/sweep.yaml`.

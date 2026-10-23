# RL-DarkForest

Dark Forest: necesitar a los otros, temer a los otros. Proyecto final de CC3092 (Deep Learning y Sistemas Inteligentes). MARL sobre un juego estocástico parcialmente observable (POSG) en JAX.

La propuesta completa está en [PROPUESTA.md](PROPUESTA.md).

## Estructura

| Ruta | Contenido |
|---|---|
| `darkforest/` | Entorno JAX (`env.py`), anomalía (`anomaly.py`), arquetipos (`archetypes.py`), redes (`networks.py`) |
| `train/` | IPPO con autojuego ficticio (`ippo.py`) y barrido (`sweep.py`) |
| `analysis/` | Métricas, replicador, diagrama de fases |
| `viz/` | Animación de la galaxia |
| `configs/` | YAML por configuración del barrido |
| `tests/` | Pruebas de cordura (H1, recompensa, anomalía) |
| `notebooks/` | Exploración y Colab |

## Instalación

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Estado

Esqueleto: todas las funciones públicas están declaradas y lanzan `NotImplementedError`. Ver el plan de trabajo en la sección 7 de la propuesta.

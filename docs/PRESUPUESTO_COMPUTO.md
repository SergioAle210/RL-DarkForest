# Presupuesto de cómputo del entrenamiento

Medido el 28 de octubre de 2026 con `train.medir_entrenamiento`, que ejecuta las iteraciones reales
del entrenamiento (`train.ippo.run_iteration`) y separa la primera iteración, que compila, de la
ejecución estable. Un "paso" es un paso de entorno sumado entre entornos, no un paso por agente.

Máquina: AMD Ryzen 5 7235HS (4 núcleos, 8 hilos), 15.7 GB de RAM, JAX solo en CPU. La GPU
(RTX 3050 de 6 GB) no se usa: JAX con GPU requeriría otra instalación y no se probó.

Configuración: `configs/base.yaml` con `train.self_play.rho=0` y λ = 0.5, red de 128 unidades,
64 pasos por rollout y 4 épocas de actualización.

## Velocidad

| Civilizaciones | Entornos | Primera iteración | Pasos por segundo | Un millón de pasos | Memoria |
|---|---|---|---|---|---|
| 2 | 64 | 4.4 s | 2,843 | 5.9 min | 472 MB |
| 2 | 256 | 5.0 s | 6,852 | 2.4 min | 694 MB |
| 2 | 1024 | 7.3 s | 14,548 | 1.1 min | 1,369 MB |
| 2 | 2048 | 11.1 s | 14,851 | 1.1 min | 1,395 MB |
| 2 | 4096 | 22.2 s | 11,716 | 1.4 min | 3,498 MB |
| 4 | 1024 | 11.2 s | 7,464 | 2.2 min | 1,442 MB |
| 4 | 2048 | 23.8 s | 5,908 | 2.8 min | 3,557 MB |

El mejor tamaño de lote es 1024 entornos: 2048 no gana y 4096 empeora y gasta más memoria. Con 4
civilizaciones el rendimiento es la mitad porque hay el doble de agentes. La memoria de cada
fila es la del proceso al terminar, y las mediciones de varios tamaños en un mismo proceso se
acumulan, así que son una cota superior.

## Tiempo para presupuestar (2 civilizaciones, 1024 entornos, 14.5 mil pasos por segundo)

| Pasos por corrida | Una corrida | Rejilla 3×3 con 3 seeds (27) | Rejilla 5×5 con 5 seeds (125) |
|---|---|---|---|
| 5 millones | 5.7 min | 2.6 h | 11.9 h |
| 10 millones | 11.5 min | 5.2 h | 23.9 h |
| 50 millones (base.yaml) | 57 min | 25.6 h | 119 h |

Con 4 civilizaciones los tiempos se duplican. Las corridas se supusieron secuenciales: ejecutar
varias a la vez en esta CPU no se midió y es probable que no dé ganancia, porque JAX ya usa los
hilos disponibles.

## Reproducir

```bash
uv run python -m train.medir_entrenamiento --config configs/base.yaml --n-envs 64 256 1024 \
    env.n_civs=2 env.anomaly.lethality=0.5 train.self_play.rho=0
```

## Límites

- Cada fila usa 2 o 3 iteraciones estables. Confirma que las métricas son finitas al inicio, pero
  no la estabilidad de una corrida larga.
- Las ventajas y su normalización se ejecutan sin compilar; hay margen de optimización sin medir.
- Los tiempos no incluyen los episodios de evaluación ni el registro, que son pequeños.
- No dicen cuántos pasos hacen falta para que la red aprenda algo.

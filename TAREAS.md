# Plan de tareas



## Fase 0: arranque)
- [ ] `uv sync`, `uv run pytest` y `make lint` funcionan en cada máquina.
- [ ] Acordar convención de ramas y PRs.

## Fase 1: baseline 

### A: Entorno (camino crítico)
- [ ] `types.py`: estructuras inmutables con `flax.struct`.
- [ ] `config.py`: dataclasses, carga de YAML y overrides `clave=valor`.
- [ ] `anomaly.py`: θ, proyecciones `h_i(θ)`, estallidos. θ no identificable con una sola proyección.
- [ ] `env.py`: `reset(key)` y `step(key, state, actions)` puros, compatibles con `jit` y `vmap`.
- [ ] Recompensa solo `log(1+r_{t+1}) - log(1+r_t)`.
- [ ] Señales viajan a velocidad `c`; muertos no actúan; recursos ≥ 0.
- [ ] Tests de invariantes y test que audita la recompensa.
- [ ] Hito: entorno mínimo funcionando a mitad de semana para destrabar a B y C.

### B: Políticas y redes
- [ ] Protocolo `Policy` (`init_carry`, `act`) y política aleatoria.
- [ ] Arquetipos y modelo nulo, registrados en `ARCHETYPES` (solo dependen de `Policy` y `Obs`).
- [ ] Red actor-crítico GRU en `networks.py`, sin lógica de entorno.
- [ ] Tests: cada política respeta el protocolo y es compatible con `jit`.

### C: Entrenamiento
- [ ] IPPO: rollout, GAE, pérdida PPO, GRU con carry.
- [ ] Autojuego ficticio: población de snapshots, probabilidad `rho`, snapshot cada K iteraciones.
- [ ] Logging (TensorBoard/wandb); guardar config resuelta, seed y checkpoints en `runs/<config>/<seed>/`.
- [ ] PRNGKeys explícitas, sin `np.random`.
- [ ] `make sanity`: entrenar la config H1 (λ = 0) como prueba de cordura.
- [ ] Avanzar con un entorno mock mientras A termina.

### D: Barrido y análisis
- [ ] `sweep.py`: lanzar el grid de `configs/sweep.yaml`, empezando por el 3x3 (`make sweep-coarse`).
- [ ] `metrics.py`: funciones puras registradas en `METRICS`; leen de disco, no corren el entorno.
- [ ] `replicator.py` y `phase_diagram.py` (diagrama λ × c).
- [ ] Definir con C y el supervisor el formato de trayectorias y resultados.
- [ ] CI (pytest + ruff) y notebook de Colab para correr en GPU.

## Fase 2: integración 
- [ ] integrar arquetipos y redes en el entorno real; tests end-to-end con cada arquetipo.
- [ ] `make sanity` con el entorno real. **Puerta de calidad: H1 debe pasar antes de seguir.**
- [ ]  barrido 3x3 con 3 seeds y primeros diagramas de fase.
- [ ] ayudar a depurar rendimiento, NaNs y shapes para `jit`.
- [ ] revisar resultados del barrido grueso y decidir si el baseline es confiable.

## Fase 3: barrido completo 
- [ ] barrido final 5x5 con 5 seeds y barridos secundarios (`p_kill`, `cost_atk`, `costly_signals`, `n_civs`).
- [ ] optimizar entrenamiento (`n_envs`, velocidad, estabilidad).
- [ ] cobertura de tests y limpieza del entorno.
- [ ] documentar cómo añadir arquetipos; docstrings.
- [ ] escribir la sección de cada módulo para el informe.

## Extensiones
- [ ] `viz/galaxy.py`: animación de la galaxia.
- [ ] Extensión J-04 (`interfere` en la anomalía), tras `anomaly.py` .
- [ ] Mejoras de algoritmos sobre: variantes de IPPO, nuevos arquetipos o acciones.
- [ ] Pulido de visualizaciones y figuras del informe.


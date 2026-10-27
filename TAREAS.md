# Plan de tareas para cinco personas

**Presentación: 9 al 13 de noviembre de 2026. Cierre de materiales: 8 de noviembre.**

Este es el calendario operativo vigente; sustituye el cronograma histórico de la sección 7
de [PROPUESTA.md](PROPUESTA.md). Las reglas están en [AGENTS.md](AGENTS.md) y los contratos
técnicos en [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

## Estado de partida: 26 de octubre de 2026

Se revisaron el esqueleto inicial (`c48b4ea`), el primer plan de tareas (`7bea76d`), los módulos
y todos los Markdown del repositorio. El avance se conserva; no se parte de cero.

| Área | Evidencia existente | Trabajo pendiente |
|---|---|---|
| Configuración | Dataclasses, carga YAML, `extends` de un nivel y overrides por claves punteadas | Completar parámetros, validación física y pruebas de casos inválidos |
| Tipos | `EnvState`, `Obs` y `StepInfo` como `flax.struct` | Cerrar dimensiones, memoria de detecciones y eventos necesarios |
| Políticas | Protocolo `Policy` y registro de cuatro arquetipos | Implementar las acciones de políticas y la GRU |
| Barrido | Producto cartesiano y CLI con `--dry-run` | Ejecutar corridas, nombrarlas y guardar resultados |
| Entorno, anomalía y entrenamiento | Firmas públicas y docstrings | Implementación funcional e integración |
| Análisis y visualización | Firmas de métricas, replicador, diagramas y animación | Cálculos, carga de resultados y renderizado |
| Pruebas y CI | Tests de configuración, comprobación superficial del protocolo y workflow | Completar pruebas vacías; retirar `xfail` por funcionalidad; verificar ejecución |

La presencia de código es evidencia de implementación parcial, no de pruebas aprobadas.
En la sesión de revisión **`uv` no está disponible**; no se ejecutaron pytest, ruff ni
entrenamientos. Resolverlo está incluido en G-01.

## Equipo y coordinación

| Persona | Frente | Primera entrega | Responsabilidad posterior | Revisor principal |
|---|---|---|---|---|
| **1 / A** | Entorno, configuración, tipos y anomalía | Episodio con dos civilizaciones, señales, recompensa e invariantes | Anomalía activa y rendimiento del simulador | B |
| **2 / B** | Política aleatoria, arquetipos y modelo GRU | Políticas intercambiables mediante `Policy` | Red recurrente y adaptación a IPPO | C |
| **3 / C** | Ejecución, entrenamiento y persistencia | Configuración, seed y episodios guardados | IPPO, GAE, autojuego y checkpoints | D |
| **4 / D** | Experimentos, métricas, replicador, CI y Colab | Métricas sobre datos sintéticos y validación del barrido | Diagramas, ablaciones y evaluación de hipótesis | A |
| **5 / E** | Matplotlib, notebooks, figuras y demo | Reproducir un episodio guardado | Animación, video y presentación visual | D |

Cada integrante escribe la sección de su módulo para el informe. E integra los materiales
visuales y D revisa la coherencia de resultados. Los nombres pueden sustituir estas etiquetas
sin cambiar los identificadores de tareas.

- Ramas cortas `codex/<id>-<tema>` y PRs con un cambio lógico, evidencia y tareas relacionadas.
- Cada PR recibe revisión de otra persona; los cambios de contratos incluyen a sus consumidores.
- A publica primero observaciones y acciones. C publica el formato de episodios con revisión
  de A, D y E; así D y E pueden avanzar con datos sintéticos sin esperar al simulador completo.
- El estado se actualiza aquí. Una tarea termina con su criterio de aceptación, no al declarar
  una función ni al obtener un `xfail` o `XPASS`.
- Estados: **Pendiente**, **Parcial**, **Completa** y **Bloqueada**. Al bloquear una tarea,
  registrar motivo y dependencia. Los IDs no se renumeran al mover tareas entre hitos.

## Calendario vigente

| Hito | Fechas de 2026 | Entregable y criterio de salida |
|---|---|---|
| M0: arranque documental | 26 octubre | Propuesta transcrita, estado auditado, cinco responsables y arquitectura documentada |
| M1: primera integración | 26–29 octubre | Contratos, entorno mínimo con dos civilizaciones, arquetipos, trayectoria guardada, métricas básicas y reproducción visual (I-01) |
| M2: aprendizaje y exploración | 30 octubre–2 noviembre | IPPO con GRU, anomalía activa, revisión empírica de H1, barrido 3×3 y medición de cómputo (I-02 a I-04) |
| M3: resultados | 3–5 noviembre | Consolidar H1–H2 con al menos cinco seeds; 5×5 si es viable, replicador, ablaciones, H3 y H5 según presupuesto (I-05) |
| M4: cierre | 6–8 noviembre | Figuras, video, informe, referencias verificadas y ensayo; H4 solo si el núcleo está completo (F-01 a F-03) |
| Presentación | 9–13 noviembre | Exposición y correcciones menores; materiales preparados desde el 8 |

Los hitos son objetivos de planificación, no evidencia de avance. Si se atrasa el camino
crítico, reducir barridos y extensiones para proteger M4; no recortar invariantes ni seeds de
resultados reportados. H4 no debe consumir el tiempo reservado al cierre.

## Fase 0: documentación y preparación

| ID | Estado | Responsable | Depende de | Entregable y aceptación |
|---|---|---|---|---|
| DOC-01 | Completa | E | PDF original | `PROPUESTA.md` con once secciones, tablas, ecuaciones y pseudocódigo; contraste visual de las nueve páginas y fecha corregida |
| DOC-02 | Completa | E | DOC-01 | README, tareas y arquitectura revisados; enlaces locales válidos, IDs únicos, dependencias sin ciclos y cambios limitados a documentación |
| G-01 | Pendiente | D; participan A–E | — | Preparar `uv` y ejecutar `uv sync --locked`, pytest, ruff check y format --check en cada máquina; registrar versiones, fallos esperados y fallos reales |
| G-02 | Pendiente | E; participan A–E | DOC-02 | Asignar los cinco nombres a los roles y abrir el primer trabajo de cada frente usando la convención de ramas y revisión |

## Fase 1: implementación por frente

Los responsables de las tablas siguientes son los del encabezado; los revisores son los
indicados en la tabla del equipo. Las dependencias permiten preparar trabajo con fixtures
antes de que esté lista la integración real.

### A: entorno y anomalía — Persona 1

| ID | Estado | Depende de | Entregable y aceptación |
|---|---|---|---|
| A-01 | Parcial | DOC-02 | Cerrar tipos, dimensiones `P/D/F/Qs/Qp`, detecciones persistentes, eventos y codificación de acciones con B y C; documentar capacidades de colas, desbordamiento, recursos iniciales, acciones inválidas y duración del refugio |
| A-02 | Parcial | A-01, G-01 | Completar configuración y validaciones, incluidos rangos y `proj_speed <= signal_speed` en todas las celdas; hiperparámetros únicamente en configuración y casos válidos/inválidos comprobados |
| A-03 | Pendiente | A-02 | `reset`, `step`, `get_obs`, `decode_action` y recompensa funcionando para dos civilizaciones con λ=0; señales y proyectiles respetan distancias en el toro y el orden del paso |
| A-04 | Pendiente | A-01, A-02 | Implementar θ, proyecciones, ruido y estallidos; probar que una proyección no identifica θ y que proyecciones distintas aportan información complementaria; conectar con A-03 |
| A-05 | Pendiente | A-03 | Pruebas reales de recursos no negativos, muertos sin acciones/recompensa, velocidad de señales, recompensa logarítmica, reproducibilidad y `jit`/`vmap`; añadir casos de anomalía al completar A-04 y retirar `xfail` correspondiente |
| A-06 | Pendiente | A-04, D-04 | Especificar con D qué hace una señal costosa y por qué es costosa de falsificar; implementar y probar el mecanismo antes de comparar H3; un booleano sin efecto no satisface la tarea |

**Desbloqueo temprano:** A-01 se entrega primero. A-03 debe estar disponible antes del cierre
del 29 de octubre para que B y C integren y D/E reproduzcan datos reales.

### B: políticas y redes — Persona 2

| ID | Estado | Depende de | Entregable y aceptación |
|---|---|---|---|
| B-01 | Parcial | A-01, G-01 | Conservar `Policy` e implementar `RandomPolicy`; acciones de forma `(N,)`, enteras y dentro de `n_discrete`, con PRNGKey explícita |
| B-02 | Pendiente | B-01 | Implementar los cuatro arquetipos registrados en `ARCHETYPES`; definir comportamiento usando solo `Obs` y carry, con casos sintéticos de detección y ausencia de detecciones |
| B-03 | Pendiente | A-01, G-01 | Actor-crítico GRU, `flatten_obs` y adaptador a `Policy`; logits para `n_discrete`, valor y carry con formas comprobadas; sin información oculta del entorno |
| B-04 | Pendiente | B-02, B-03 | Probar todas las políticas bajo `jit`, manejo y reinicio de carry y uso de observaciones parciales; documentar cómo añadir arquetipos y mantener firmas tipadas |

### C: ejecución y entrenamiento — Persona 3

| ID | Estado | Depende de | Entregable y aceptación |
|---|---|---|---|
| C-01 | Pendiente | A-01 | Cerrar con A, D y E el formato versionado de trayectorias/resultados de la arquitectura; producir un episodio sintético pequeño con campos, formas, máscaras y alineación temporal conocidos |
| C-02 | Pendiente | C-01, G-01 | Implementar persistencia y logging; resolver TensorBoard faltante, guardar config resuelta/seed/metadatos y checkpoints; usar nombres únicos y probar que corridas distintas no se sobrescriben |
| C-03 | Pendiente | A-03, B-02, C-02 | Ejecutor de episodios para política aleatoria y cada arquetipo; guardar y recargar NPZ; las corridas reales se distinguen de fixtures sintéticos |
| C-04 | Pendiente | B-03, C-03 | IPPO con rollout `scan`/`vmap`, GAE y pérdida PPO; manejar muerte/truncamiento y carry; probar GAE en un ejemplo calculable y una actualización finita; preparar lógica con un mock mientras termina A |
| C-05 | Pendiente | C-04 | Autojuego ficticio con snapshots congelados, `rho`, límite de población y frecuencia desde configuración; verificar asignación reproducible y pérdidas solo de agentes de la política actual |
| C-06 | Pendiente | C-03 | Evaluar políticas contra sí mismas y contra arquetipos, guardando episodios, retornos y etiquetas de políticas; incorporar checkpoints tras C-04; exportar insumos del replicador sin importar `analysis/` ni `viz/` |

### D: experimentos y análisis — Persona 4

| ID | Estado | Depende de | Entregable y aceptación |
|---|---|---|---|
| D-01 | Pendiente | C-01 | Métricas puras registradas en `METRICS`; empezar por supervivencia, emisión y transmisiones con resultados conocidos en fixtures; completar causas, ataques preventivos, refugios y estimación de información mutua |
| D-02 | Parcial | G-01 | Verificar CI existente y añadir comprobación de formato; sustituir pruebas vacías por aserciones con sus propietarios, retirar `xfail` resueltos y mostrar pendientes sin confundirlos con cobertura |
| D-03 | Parcial | A-02, C-02 | Completar ejecutor del barrido y nombres por celda; validar cada configuración antes de entrenar; distinguir seeds exploratorias y finales, preservar `--dry-run` y comprobar conteos sin entrenar |
| D-04 | Pendiente | B-02, C-01 | Definir protocolo H1–H2, comparadores, umbrales de clasificación, incertidumbre y criterio de evidencia antes de revisar resultados; documentar tratamiento de ausencia de frontera y de denominadores cero |
| D-05 | Pendiente | C-06, D-01 | Construir matriz de pagos desde evaluaciones guardadas; implementar dinámica del replicador y puntos fijos con pruebas; `analysis/` no ejecuta el simulador |
| D-06 | Pendiente | D-01, D-04 | Agregar `results.json` y generar diagramas λ×c y distribución por seed para H5; distinguir celdas incompletas, incertidumbre y ausencia de transición sin inventar resultados |
| D-07 | Pendiente | A-04, C-06, D-04 | Ablaciones de canal y mensajes barajados con configuraciones comparables; comparar H3 únicamente tras A-06; documentar seeds, presupuesto y cambios de cada comparación |
| D-08 | Pendiente | G-01, C-04 | Notebook de Colab que use módulos del repositorio; medir compilación y ejecución por separado, tiempo por paso/corrida, memoria y estabilidad; registrar hardware y estimación del barrido |
| D-09 | Pendiente | I-04, D-03 | Barridos secundarios de un factor (`p_kill`, `cost_atk`, `costly_signals`, `n_civs`), manteniendo el resto en la base y ≥5 seeds si se reportan; señales costosas requieren A-06 y coaliciones se coordinan con X-01; priorizar H1–H2 dentro del presupuesto |

### E: visualización y demo — Persona 5

La animación pasa de extensión a entregable principal, conforme a la propuesta.

| ID | Estado | Depende de | Entregable y aceptación |
|---|---|---|---|
| E-01 | Pendiente | C-01 | Notebook para seleccionar episodios y consultar metadatos; consume funciones de módulos y señala datos sintéticos sin duplicar simulación ni modelo |
| E-02 | Pendiente | C-01 | Implementar `render_episode` con Matplotlib: posiciones, vivos/muertos, señales, proyectiles y estallidos cuando existan; reproducir un fixture y luego un episodio de C-03 sin ejecutar el entorno |
| E-03 | Pendiente | E-02, D-06 | Demo con galaxia y diagrama de fases; manejar celdas sin resultados; exportar video reproducible y figuras legibles con config, seed y unidades identificables |
| E-04 | Pendiente | E-02 | Documentar exportación de PNG/SVG y video, dependencias de los escritores y almacenamiento; conservar productos generados en `runs/` y solo figuras pequeñas seleccionadas en `docs/assets/` |

## Fases 2 y 3: integración y resultados

| ID | Estado | Responsable | Depende de | Entregable y aceptación |
|---|---|---|---|---|
| I-01 | Pendiente | C, revisión E | A-05, B-02, C-03, D-01, E-02 | Ejecutar episodios con cada arquetipo y política aleatoria; guardar, recargar, calcular métricas básicas y reproducir con dos civilizaciones; smoke adicional con cuatro |
| I-02 | Pendiente | C, revisión B | I-01, A-04, B-04, C-04 | Integrar red con entorno real y anomalía; actualización sin NaNs y pruebas de shapes/carry; activar casos de anomalía de A-05 |
| I-03 | Pendiente | D, ejecución C | I-02, C-05, C-06, D-04 | Ejecutar `make sanity` con el entorno real y revisar H1; separar invariantes, calidad del aprendizaje y evidencia empírica; registrar diagnóstico de cualquier resultado inesperado antes de ampliar el barrido |
| I-04 | Pendiente | D, apoyo C | I-03, D-03, D-06, D-08 | Barrido exploratorio 3×3 con tres seeds y primeros diagramas; revisar confiabilidad, tiempos, memoria y estabilidad; dejar decisión escrita sobre presupuesto final |
| I-05 | Pendiente | D, apoyo A–C | I-04 | Consolidar H1–H2 con ≥5 seeds por configuración reportada; 5×5 si cabe o 3×3 con ≥5 seeds; incorporar D-05/D-07/D-09 y H5 según viabilidad; conservar resultados y limitaciones |
| I-06 | Pendiente | C, revisión A | D-08 | Optimizar `n_envs`, velocidad y estabilidad con mediciones comparables; depurar NaNs y formas sin alterar recompensa ni introducir aleatoriedad global |

### Puertas de calidad y reproducibilidad

- **Antes de entrenar:** invariantes, recompensa y PRNGKeys explícitas; pruebas con aserciones
  reales. La preparación de G-01 puede mostrar fallos esperados del esqueleto, pero eso no
  habilita entrenamientos.
- **Antes del barrido:** revisar H1 con entorno real y controles. La redacción original de
  la propuesta se conserva; operativamente H1 es una hipótesis empírica. Un resultado
  inesperado puede indicar un error, aprendizaje insuficiente o falta de apoyo a la hipótesis.
  Se investiga y documenta, sin modificar la recompensa para forzar el resultado.
- **Antes de reportar:** al menos cinco seeds por configuración, config resuelta y seed
  guardadas, comparación con modelos nulos y distinción entre datos exploratorios y finales.
- **Presupuesto:** 3×3×3 = 27 corridas exploratorias; 3×3×5 = 45 y 5×5×5 = 125 corridas para
  el barrido principal reportable, sin contar evaluaciones, ablaciones ni barridos secundarios.
  Si una corrida supera 10 minutos, aplicar la mitigación de la propuesta y reevaluar el 5×5.
- El YAML actual declara cinco seeds para ambos grids. El uso explícito de tres seeds
  exploratorias se implementará en D-03; hoy `make sweep-coarse` no implica tres seeds.

## Fase 4: informe y presentación

| ID | Estado | Responsable | Depende de | Entregable y aceptación |
|---|---|---|---|---|
| F-01 | Pendiente | A–E, integración D | I-05 | Cada persona entrega descripción y limitaciones de su módulo; D integra metodología, resultados y conclusiones respaldadas; verificar referencias originales antes de citarlas |
| F-02 | Pendiente | E, revisión D | F-01, E-03, E-04 | Informe y presentación coherentes con resultados, figuras legibles y video reproducible; distinguir hipótesis apoyadas, no apoyadas y no evaluadas |
| F-03 | Pendiente | E; participan A–D | F-02 | Ensayo, revisión de enlaces y disponibilidad local de materiales; cierre el 8 de noviembre para presentar del 9 al 13 |

## Extensiones fuera del camino crítico

| ID | Estado | Responsable | Depende de | Entregable y aceptación |
|---|---|---|---|---|
| X-01 | Pendiente | D, apoyo A/E | I-05 | H4 con seis a ocho civilizaciones: grafo de respuestas y comunidades; solo si termina antes del cierre sin desplazar informe/video |
| X-02 | Pendiente | A, revisión B | A-04, I-05 | J-04: interferencia con configuración, acciones y pruebas; no necesaria para H1–H2 |
| X-03 | Pendiente | B, revisión C | I-05 | Variantes de IPPO/MAPPO, nuevos arquetipos o acciones, con comparación controlada; no sustituir el baseline sin evidencia |
| X-04 | Pendiente | E, revisión D | F-03 | A-01: visualización artística del comportamiento colectivo |

El pulido de figuras y la animación base permanecen en E-02 a E-04 y F-02; no dependen de
estas extensiones. La reputación de J-02 es una ampliación de X-01, no un requisito del núcleo.

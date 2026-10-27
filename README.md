# RL-DarkForest

Proyecto de aprendizaje por refuerzo multiagente en JAX sobre un juego parcialmente
observable del bosque oscuro. La especificación plantea las hipótesis H1–H5 sin imponer
cooperación ni agresión mediante recompensas adicionales.

**Presentación: 9 al 13 de noviembre de 2026. Cierre de materiales: 8 de noviembre.**

## Empezar por los fundamentos

La primera iteración de Sergio reúne configuración validada, decodificación de acciones y
comprobaciones de las estructuras inmutables. El recorrido está en
[fundamentos.ipynb](notebooks/fundamentos.ipynb) y la guía en
[PRIMEROS_PASOS.md](docs/PRIMEROS_PASOS.md). Los ejemplos de observaciones son sintéticos;
la simulación de episodios y el entrenamiento se construirán en las siguientes entregas.

Desde la raíz del repositorio:

```bash
uv sync --locked --group notebooks
uv run --group notebooks pytest tests/test_config.py tests/test_actions.py tests/test_types.py
uv run --group notebooks jupyter lab notebooks/fundamentos.ipynb
```

En Jupyter, reinicia el kernel y ejecuta todas las celdas. Podrás cargar la configuración de
dos civilizaciones, aplicar overrides, inspeccionar las acciones y comprobar errores y
máscaras. El notebook versionado se conserva sin salidas; las copias ejecutadas de
verificación van en `runs/notebook_checks/`.

## Documentación y seguimiento

| Documento | Contenido |
|---|---|
| [PROPUESTA.md](docs/PROPUESTA.md) | Transcripción científica: once secciones, hipótesis, ecuaciones y referencias |
| [TAREAS.md](docs/TAREAS.md) | Alcance, dependencias, criterios de aceptación y calendario operativo |
| [ARQUITECTURA.md](docs/ARQUITECTURA.md) | Contratos, límites entre módulos y almacenamiento previsto |
| [TODO.md](docs/TODO.md) | Única fuente de avance: tarea, responsable, estado y fecha |
| [PRIMEROS_PASOS.md](docs/PRIMEROS_PASOS.md) | Preparación, recorrido de Sergio, reproducción y aprendizaje futuro |
| [AGENTS.md](AGENTS.md) | Reglas de implementación y reproducibilidad |

El PDF original `rl-proyecto.pdf` no está incluido en esta copia del repositorio; la
transcripción se conserva en `docs/PROPUESTA.md`. Los documentos de trabajo están en `docs/`;
README, AGENTS y CLAUDE permanecen en la raíz.

El equipo tiene **cinco integrantes en total**. Sergio realiza esta primera iteración común.
Los frentes posteriores son entorno, políticas/modelo, entrenamiento/persistencia,
experimentos/análisis y visualización/demo. Sus asignaciones se completarán en TODO.

TODO admite únicamente `listo`, `pendiente` y `en proceso`. Se actualiza la fecha al cambiar
realmente el estado, usando `YYYY-MM-DD`. Una tarea futura sin asignar conserva responsable,
estado y fecha vacíos. Crear un archivo preparado no completa su funcionalidad. TAREAS
conserva las especificaciones y los IDs técnicos; no duplica estados ni personas.

## Organización

| Carpeta | Uso |
|---|---|
| `darkforest/` | Configuración, tipos, entorno, anomalía, políticas y red GRU |
| `train/` | Ejecución, IPPO, autojuego, evaluación y barridos |
| `analysis/` | Métricas, replicador y diagramas sobre archivos guardados |
| `viz/` | Reproducción de episodios y exportación con Matplotlib |
| `configs/` | YAML base, H1 y barridos |
| `tests/` | Pruebas funcionales y archivos preparados para los demás frentes |
| `notebooks/` | `fundamentos.ipynb`; espacios futuros de entrenamiento, resultados y demo |
| `docs/` | Documentación; `docs/assets/` reservado para imágenes pequeñas seleccionadas |
| `runs/` | Salidas locales ignoradas por Git, creadas cuando haya un productor real |

Los notebooks importan los módulos del proyecto. Los ejecutables siguen
`uv run python -m paquete.modulo`. Se conservan las implementaciones parciales y firmas
existentes para que los cinco frentes trabajen sobre los mismos contratos.

## Comprobaciones y siguientes entregas

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Los `xfail` y los `skip` del simulador todavía pendiente no acreditan una simulación
funcional. Los archivos preparados para pruebas futuras contienen solamente un docstring.
La guía distingue la validación de fundamentos de las puertas de calidad para entrenar.

El siguiente bloque cerrará las dimensiones físicas y construirá un episodio con dos
civilizaciones. Después, una GRU compartida producirá acciones y valores; rollouts, GAE y
PPO actualizarán el modelo, el autojuego aportará oponentes congelados y las evaluaciones
guardadas alimentarán el análisis y la demo. La recompensa seguirá siendo la diferencia de
`log1p` de recursos.

`make sanity` y el ejecutor de barridos requieren implementaciones posteriores. El barrido
actual incluye combinaciones con proyectiles más rápidos que las señales: deben corregirse
antes de entrenar. Se priorizan H1–H2; toda configuración presentada como resultado requiere
al menos cinco seeds, aunque la exploración inicial use tres.

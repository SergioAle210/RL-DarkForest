# RL-DarkForest

**Dark Forest: necesitar a los otros, temer a los otros.** Proyecto final de CC3092 (Deep
Learning y Sistemas Inteligentes). Aprendizaje por refuerzo multiagente sobre un juego
estocástico parcialmente observable (POSG) en JAX.

**Presentación: 9 al 13 de noviembre de 2026. Cierre de materiales: 8 de noviembre.**

## Por dónde empezar

1. Leer [PROPUESTA.md](docs/PROPUESTA.md): problema, entorno, hipótesis H1–H5 y metodología,
   transcritos del PDF con las fórmulas y tablas restauradas.
2. Consultar [TAREAS.md](docs/TAREAS.md): estado real, reparto entre cinco personas, dependencias,
   criterios de aceptación y calendario operativo actualizado.
3. Revisar [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md): contratos de módulos, trayectorias,
   imágenes, checkpoints, notebooks y demo.
4. Seguir [AGENTS.md](AGENTS.md): reglas de desarrollo y reproducibilidad.

El cronograma de la sección 7 de la propuesta se conserva como antecedente histórico. Para
trabajar, usar las fechas de TAREAS.

## Estado al 26 de octubre de 2026

El repositorio contiene un esqueleto con implementación parcial. Ya existen dataclasses de
configuración, carga YAML con overrides, tipos inmutables, el protocolo de políticas y la
expansión del grid con `--dry-run`. El entorno, la anomalía, las políticas concretas, la GRU,
el entrenamiento, las métricas y la animación siguen principalmente pendientes.

Hay tests de configuración y un workflow de CI, pero varios tests del entorno son vacíos o
aceptan `NotImplementedError` mediante `xfail`. Eso no demuestra que el simulador funcione.
En esta revisión no se ejecutaron pytest ni ruff porque **`uv` no está disponible en la
sesión**. El arranque documental no modifica Python, YAML, dependencias ni CI.

## Equipo

| Persona | Frente |
|---|---|
| 1 / A | Entorno, tipos, configuración y anomalía |
| 2 / B | Políticas, arquetipos y modelo GRU |
| 3 / C | Ejecución, entrenamiento y persistencia |
| 4 / D | Experimentos, métricas, replicador, CI y Colab |
| 5 / E | Visualización, notebooks de demo, figuras y presentación |

Cada persona documenta su módulo. La primera integración consiste en simular un episodio
con dos civilizaciones, guardarlo, analizarlo y reproducirlo; aún no requiere demostrar H1
ni completar el entrenamiento. Los responsables y revisores de cada entrega están en TAREAS.

## Estructura

| Ruta | Contenido |
|---|---|
| `darkforest/` | Entorno, anomalía, configuración, tipos, políticas, arquetipos y redes |
| `train/` | Ejecución, IPPO, autojuego, evaluación, logging y barridos |
| `analysis/` | Métricas, replicador y diagramas desde resultados guardados |
| `viz/` | Animación de trayectorias guardadas con Matplotlib |
| `configs/` | YAML de configuración base, H1 y barridos |
| `tests/` | Pruebas unitarias y de integración, varias todavía pendientes |
| `notebooks/` | Exploración, Colab y demo; actualmente solo contiene `.gitkeep` |
| `docs/` | Arquitectura y acuerdos técnicos |
| `docs/assets/` (previsto) | Imágenes pequeñas seleccionadas para documentación |
| `runs/<config>/<seed>/` (previsto, ignorado) | Config resuelta, metadatos, trayectorias, checkpoints, resultados, figuras y videos |

Se conserva esta organización. La UI será Matplotlib y notebooks que consumen los módulos.
Las imágenes y videos generados se guardarán dentro de las corridas; los datos y checkpoints
no se subirán a Git. Los formatos y sus tareas de implementación están en la arquitectura.

## Preparación y comprobaciones

Requisitos declarados: Python 3.10 o superior y `uv` disponible. Desde la raíz:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Estos son comandos de preparación, no resultados verificados. Revisar fallos, skips, xfails
y XPASS; las pruebas vacías deben completarse antes de entrenar. Si se dispone de `make`,
`make test` y `make lint` agrupan las comprobaciones correspondientes. En PowerShell pueden
usarse directamente los comandos anteriores sin instalar `make`.

Para inspeccionar el barrido sin entrenar, una vez preparado el entorno:

```bash
uv run python -m train.sweep --config configs/sweep.yaml --grid coarse_grid --dry-run
```

Actualmente el grid grueso usa las cinco seeds del YAML: enumera 45 combinaciones. La
selección de tres seeds exploratorias y la validación de velocidades están pendientes en
D-03/A-02. La expansión del grid no certifica que las configuraciones sean físicamente válidas.

## Entrenamiento previsto

Los puntos de entrada existen, pero todavía alcanzan funciones sin implementar. Se usarán
después de cerrar los invariantes e integrar el entorno y las políticas:

```bash
uv run python -m train.ippo --config configs/h1_sanity.yaml
uv run python -m train.sweep --config configs/sweep.yaml --grid coarse_grid
uv run python -m train.sweep --config configs/sweep.yaml --grid grid
```

Equivalen a `make sanity`, `make sweep-coarse` y `make sweep`. La CLI de entrenamiento acepta
overrides `clave=valor`, por ejemplo `train.seed=1`. Cada corrida deberá guardar la
configuración resuelta y la seed.

La prioridad científica es H1–H2. Las tres seeds iniciales serán exploratorias; todo resultado
reportado tendrá al menos cinco seeds por configuración. H1 se evaluará con controles y se
documentarán los resultados inesperados, sin alterar la recompensa para forzar estrategias.

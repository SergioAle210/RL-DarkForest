# Arquitectura, interfaces y archivos del proyecto

Este documento describe las interfaces existentes y los acuerdos para completarlas. **Los
formatos de persistencia y el flujo integrado aquí descritos están pendientes de
implementación.** Su avance se sigue por ID en [TAREAS.md](TAREAS.md).

La especificación científica está en [PROPUESTA.md](PROPUESTA.md) y las reglas de desarrollo
en [AGENTS.md](../AGENTS.md). Presentación: **9 al 13 de noviembre de 2026**; cierre: **8 de
noviembre**. El calendario operativo vive únicamente en TAREAS.

## 1. Módulos y límites

| Ubicación | Responsabilidad | Propietario |
|---|---|---|
| `darkforest/config.py` | Configuración tipada, YAML y overrides; fuente de hiperparámetros | A |
| `darkforest/types.py` | Estado, observación y diagnósticos inmutables | A |
| `darkforest/env.py`, `darkforest/anomaly.py` | Transición del POSG, señales, proyectiles y anomalía | A |
| `darkforest/policies.py`, `darkforest/archetypes.py` | Interfaz intercambiable, modelo nulo y heurísticas | B |
| `darkforest/networks.py` | Actor-crítico GRU; no contiene reglas del entorno | B |
| `train/` | Ejecución, entrenamiento, evaluación, autojuego y persistencia | C; D mantiene el barrido |
| `analysis/` | Métricas, matriz de pagos, replicador y diagramas sobre resultados guardados | D |
| `viz/` | Reproducción de trayectorias y exportación de animaciones | E |
| `configs/` | Configuraciones base, H1 y barridos | A; D mantiene el diseño experimental |
| `tests/` | Pruebas de cada módulo y de integración | Cada propietario; D coordina CI |
| `notebooks/` | Exploración y demo (E), ejecución y mediciones en Colab (D) | D/E |
| `docs/` | Acuerdos técnicos; `docs/assets/` para imágenes pequeñas seleccionadas | E; revisión de todos |

Se mantiene esta estructura. No se crea una carpeta genérica de scripts ni se trasladan las
redes a un segundo paquete de modelos. Los ejecutables Python siguen el patrón
`uv run python -m paquete.modulo`; los notebooks importan funciones y no duplican lógica.

Dependencias permitidas:

- `train/`, `analysis/` y `viz/` pueden consumir `darkforest/`.
- `darkforest/` no importa los paquetes de entrenamiento, análisis ni visualización.
- `train/` no importa `analysis/` ni `viz/`.
- `analysis/` y `viz/` trabajan sobre archivos o arrays ya cargados; no ejecutan el entorno.
- Las políticas reciben `Obs`, no el estado completo ni los diagnósticos.

Flujo de datos previsto:

```text
YAML + overrides + seed
          |
          v
train/ usa entorno + políticas de darkforest/
          |
          v
runs/<config>/<seed>/
  config + metadatos + checkpoints + trayectorias
          |                         |
          v                         v
analysis/ -> results.json        viz/ -> animación
          |
          v
figuras + agregados -> notebook/demo
```

C genera evaluaciones contra políticas y arquetipos. D calcula métricas científicas,
clasificaciones y pagos a partir de esas evaluaciones; el entrenamiento puede registrar sus
propias pérdidas y retornos sin importar el análisis. La firma actual de `payoff_matrix`
sugiere simular desde `analysis/`: C-06 y D-05 deben corregir esa separación antes de implementar
el replicador. La intención actual de `train.ippo.evaluate` de devolver un régimen también
debe ajustarse a esta división.

## 2. Interfaces existentes

### Entorno

| Interfaz | Entrada y salida declaradas | Condición |
|---|---|---|
| `DarkForestEnv(cfg)` | Recibe `EnvConfig` | Parámetros estáticos durante la compilación |
| `reset(key)` | Devuelve `(Obs, EnvState)` | Inicialización reproducible con clave explícita |
| `step(key, state, actions)` | Devuelve `StepOut(obs, state, reward, done, info)` | Función pura, compatible con `jit` y `vmap` |
| `get_obs(key, state)` | Devuelve `Obs` | Observación parcial; no filtra θ ni posiciones no detectadas |
| `decode_action(action)` | Devuelve tipo de acción y argumento | Compatible con acciones discretas parametrizadas |
| `reward_fn(resources_t, resources_t1, alive_t)` | Recompensa por agente | Diferencia de `log1p`; cero para agentes ya eliminados |

`actions`, `reward` y `done` tienen forma `(N,)`. `done` indica eliminación o final del
episodio para cada civilización. El orden declarado del paso es acciones → señales y
proyectiles → detección → anomalía → recompensa. La implementación aún está pendiente.

Hay **siete familias** en `ACTIONS`, pero el espacio discreto incluye los argumentos de
`transmitir(k)` y `atacar(j)`:

```text
n_discrete = 2 + 1 + K + 1 + M + 1 = 5 + K + M
```

`env.n_actions` actualmente cuenta familias; la política y los logits de la red necesitan
`env.n_discrete`. Con `K=4` y `M=4` son 13 acciones, no siete. A-01 fijará y probará la
codificación y sus límites. Aunque AGENTS menciona `Action`, actualmente no hay una clase
`Action` en `types.py`: las acciones son arrays enteros. Tampoco hay un campo `n_actions` en
la dataclass de configuración; no debe asumirse que esas interfaces ya existen.

### Estado, observación y diagnósticos

| Estructura | Contenido actual | Uso permitido |
|---|---|---|
| `EnvState` | Tiempo, posiciones, recursos, vivos, emisión, refugio, θ, mediciones y colas | Simulación; copia de diagnóstico para análisis y visualización |
| `Obs` | Recursos/emisión propios, detecciones enmascaradas, medición propia, tiempo desde estallido, vivo e identificador | Entrada exclusiva de las políticas |
| `StepInfo` | Causa de muerte, atacante, ataques preventivos, refugios correctos y estallidos | Diagnóstico y métricas; nunca entrada adicional de la política |

Las estructuras ya son pytrees inmutables de Flax. Se conservan formas estáticas y máscaras
de validez. `N` es el número de civilizaciones, `M` las ranuras de detección y `K` los símbolos.
Quedan por cerrar en A-01 las dimensiones de θ (`P`), medición (`D`), detección (`F`), colas
(`Qs`, `Qp`) y la persistencia de detecciones/historial necesaria para ataques y métricas.

### Políticas y red

```text
Policy.init_carry(n_agents) -> carry
Policy.act(key, carry, obs) -> (nuevo_carry, acciones)

ActorCriticGRU(carry, obs_flat, agent_id) -> (nuevo_carry, logits, valor)
```

Una política sin memoria puede usar `None` como carry. B implementa y prueba el adaptador
del actor-crítico a `Policy`, porque la red por sí sola no implementa ese protocolo. El
entrenamiento conserva logits, valores, probabilidades y carry necesarios para PPO; las
heurísticas no necesitan saber cómo se entrena la red. Los agentes muertos y el reinicio de
episodios requieren máscaras y reinicio de carry verificados en B-04/C-04.

Toda fuente aleatoria recibe una clave JAX explícita y deriva claves con `split`. No se
reutilizan claves para eventos independientes ni se usa `np.random` global. La recompensa
se mantiene exclusivamente como `log(1+r_{t+1}) - log(1+r_t)`, sin bonos por estrategias.

## 3. Persistencia acordada

Esta distribución se implementará en C-01/C-02; las carpetas no se crean hasta que exista
un productor real de sus archivos.

```text
runs/
└── <config>/                    # nombre único de configuración resuelta
    └── <seed>/
        ├── config.yaml          # valores resueltos, incluidos overrides y seed
        ├── metadata.json        # identidad, procedencia y versiones
        ├── checkpoints/         # parámetros, optimizador y estado para continuar
        ├── trajectories/
        │   ├── episode_000001.npz
        │   └── episode_000001.json
        ├── results.json         # resumen científico producido desde evaluaciones
        ├── figures/             # PNG/SVG de la corrida
        └── videos/              # animaciones de la corrida
```

- `<config>` identifica la configuración resuelta sin la seed; cada celda del barrido y cada
  ablación debe tener identidad distinta. Usar un nombre legible y un sufijo determinista de
  los parámetros evita que todas las celdas compartan el nombre `base`.
- La seed define el siguiente nivel. Repetir una corrida existente exige una reanudación
  explícita o una identidad nueva; no se sobrescriben resultados silenciosamente.
- `metadata.json` registra versión del formato, config ID, seed, revisión Git y estado de
  cambios locales, versiones relevantes, hardware, presupuesto y tipo de ejecución
  (`sintetica`, `exploratoria` o `final`).
- Guardar cada configuración resuelta hace reproducibles los overrides. La ruta YAML original
  no basta: actualmente la CLI pasa `args.config` como nombre y `run_dir` lo concatena; C-02
  debe normalizar esa identidad y evitar subrutas accidentales como `configs/base.yaml/0`.
- Los checkpoints guardan también el estado necesario para reanudar de forma reproducible,
  incluido PRNG y población cuando corresponda. No basta con los pesos de la red.

### Contrato mínimo de episodios

C-01 versionará los nombres exactos de claves y publicará un fixture compartido antes de que
D/E implementen lectores. El contrato debe cubrir:

| Datos | Necesidad |
|---|---|
| Config ID, seed, episode ID y políticas participantes | Trazabilidad y matriz de pagos |
| Observaciones, acciones, recompensas y `done` por agente | Evaluación y reconstrucción temporal |
| Tiempo, posiciones, recursos, vivos, emisión y refugio | Métricas y reproducción visual |
| Señales/proyectiles, sus tiempos y máscaras de validez | Animar propagación sin volver a simular |
| Mediciones y θ de diagnóstico | Evaluar información y predicción, sin acceso de la política |
| Eventos de detección/recepción con emisor y receptor | Medir interacción y construir el grafo de H4 |
| `StepInfo` y máscaras de agentes/pasos válidos | Causas de muerte, ataques preventivos y refugios acertados |

Para un episodio con `T` transiciones, conservar `T+1` estados/observaciones (incluido el
inicial) y `T` acciones, recompensas, `done` y diagnósticos. El índice `t` de una transición
une el estado `t` con el `t+1`. El JSON del episodio describe dimensiones, campos, versión y
tipo de datos; el NPZ contiene arrays numéricos o booleanos, sin objetos que requieran pickle.
Las máscaras distinguen relleno de observaciones válidas y evitan contar pasos posteriores
a la muerte como actividad.

Los datos del rollout de entrenamiento pueden tener dimensiones adicionales de entornos y
carry; no obligan a exportar todos los rollouts. C guardará episodios de evaluación suficientes
para métricas y demo. El estado completo guardado es solo diagnóstico y no se concatena a
`Obs` ni se utiliza como entrada encubierta de las políticas.

`results.json` identificará config, seed, episodios evaluados, políticas, métricas, régimen
y versión del protocolo de clasificación. D define ese protocolo antes de observar resultados
(D-04); un dato ausente o una frontera no detectada no se convierten en cero ni en éxito.

### Agregados e imágenes

Los agregados entre seeds/configuraciones se guardarán en `runs/aggregates/<experimento>/`,
con un manifiesto de corridas fuente y subcarpetas `figures/` y `videos/`. Es una salida de
análisis, no una corrida: los lectores deben descubrir corridas por sus metadatos y omitir
esa carpeta al buscar `results.json` por seed.

`runs/` ya está ignorado por Git. También hay exclusiones globales para NPZ, MP4 y GIF. No se
versionan checkpoints, datos crudos ni logs. `docs/assets/` se reserva para PNG/SVG pequeños
seleccionados, con pie que identifique origen, config y seed o agregado. No se crean imágenes
ni resultados ficticios en esta entrega documental.

## 4. UI y notebooks

La interfaz acordada es **Matplotlib y notebooks**. La demo seleccionará un episodio guardado,
mostrará metadatos y permitirá reproducirlo; la vista combinada mostrará la galaxia junto al
diagrama de fases. Los controles y las figuras no ejecutarán pasos del entorno durante la
reproducción. Colab podrá lanzar entrenamiento mediante `train/`, separado de la reproducción.

E comienza con el fixture de C-01 y lo identifica visualmente como sintético. Al llegar C-03,
usa el mismo lector para episodios reales. D/E representan celdas pendientes como pendientes,
no como resultados. Las figuras mostrarán nombres de parámetros, unidades y seeds incluidas.

La exportación de video exige un escritor disponible; E-04 documentará y verificará sus
dependencias. Los notebooks de ejecución/exploración se guardan sin resultados pesados;
las salidas se regeneran desde las corridas. La animación base forma parte de M1, mientras
que la capa artística permanece como extensión.

## 5. Bloqueos concretos y validación

| Hallazgo de la revisión | Resolución planificada |
|---|---|
| `uv` no disponible en esta sesión | G-01: preparar herramientas y registrar ejecución real de checks |
| Dimensiones, capacidad de colas, recursos iniciales y reglas de acciones inválidas sin cerrar | A-01/A-02: documentar y configurar antes de integrar consumidores |
| `proj_speed=0.05`, pero el barrido incluye `signal_speed=0.01` y `0.0316` | A-02/D-03: validar todas las combinaciones y corregir el diseño físico antes de entrenar |
| TensorBoard predeterminado sin dependencia declarada | C-02: añadir/verificar el escritor elegido en una implementación posterior |
| `payoff_matrix(env_cfg, ...)` sugiere simulación dentro de análisis | C-06/D-05: evaluar en `train/` y calcular pagos en `analysis/` desde archivos |
| `costly_signals` sin mecanismo definido | A-06/D-07: especificar y probar costo/verificabilidad antes de H3 |
| Pruebas con `...` y `xfail(strict=False)` | A-05/D-02: aserciones reales y retirada de marcadores al implementar |
| Cinco seeds en el YAML también para el grid grueso | D-03: selección explícita de tres para exploración; ≥5 para resultados reportados |

Antes de entrenar se requieren recursos no negativos, muertos inactivos, velocidad correcta
de señales, recompensa auditada, reproducibilidad y `jit`/`vmap`. B verifica el protocolo y
carry; C prueba GAE y actualización PPO; D verifica métricas con ejemplos conocidos y E
comprueba alineación temporal de la reproducción.

La prueba de integración I-01 recorre simulación → guardado → carga → métricas → reproducción.
H1 es una evaluación empírica posterior; una ejecución sin errores no confirma la hipótesis.
Los resultados inesperados se investigan sin cambiar la recompensa para imponer estrategias.

Este arranque modifica documentación. No implementa los contratos pendientes ni cambia
Python, YAML, dependencias o CI.

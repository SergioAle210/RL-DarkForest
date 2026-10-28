# Arquitectura, interfaces y archivos del proyecto

Este documento describe las interfaces existentes y los acuerdos para completarlas. El
formato de episodios v1, la simulación, IPPO inicial y la reproducción ya tienen una primera
implementación; el autojuego y los experimentos completos siguen pendientes.
El avance se registra en [TODO.md](TODO.md); los criterios y dependencias
por ID están en [TAREAS.md](TAREAS.md).

La especificación científica está en [PROPUESTA.md](PROPUESTA.md) y las reglas de desarrollo
en [AGENTS.md](../AGENTS.md). Presentación: **9 al 13 de noviembre de 2026**; cierre: **8 de
noviembre**. El calendario operativo vive únicamente en TAREAS.

## 1. Módulos y límites

| Ubicación | Responsabilidad | Frente |
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
Los frentes A–E cubren cinco integrantes en total. Sergio construye los fundamentos comunes
de esta iteración; las asignaciones futuras se completarán únicamente en TODO.

### Configuración y fundamentos

Las dataclasses congeladas validan también la construcción directa. `load_config` conserva
YAML, herencia `extends` de un nivel y overrides por claves punteadas; la precedencia es
valores predeterminados → YAML base → YAML hijo → overrides. Los errores identifican el
parámetro en español: `KeyError` para claves desconocidas, `TypeError` para tipos incorrectos
y `ValueError` para valores inválidos.

Los conteos son enteros positivos, `n_civs` pertenece a `{2, 4, 6, 8}` y las seeds están entre
0 y `2**32 - 1`, conforme al dominio de claves de 32 bits. Los números deben ser finitos;
probabilidades y mezclas están en `[0, 1]`, costos y coeficientes son no negativos, y tasa de
aprendizaje, margen PPO y velocidades son positivos. Se exige
`proj_speed <= signal_speed`, sin corregir valores silenciosamente. El algoritmo inicial
es `ippo`, el backend es `tensorboard` o `wandb` y el directorio de salida debe ser válido.

La validación evita empezar con parámetros incoherentes. No implementa logging ni aprueba
las combinaciones físicas del barrido actual: `--dry-run` expande el producto cartesiano,
construye la configuración de cada corrida y falla si alguna es inválida. A-02/D-03 deben
corregir los YAML para que todas las celdas pasen esa validación.

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
episodio para cada civilización. El orden del paso es acciones → propagación y detección →
impactos → anomalía → recompensa. Si coinciden impacto y anomalía, registramos la muerte
como ataque para no contar dos veces la causa. El final de tiempo es un corte para el
bootstrap del entrenamiento; la muerte es una terminación real.

Hay **siete familias** en `ACTIONS`, pero el espacio discreto incluye los argumentos de
`transmitir(k)` y `atacar(j)`:

```text
n_discrete = 2 + 1 + K + 1 + M + 1 = 5 + K + M
```

`env.n_actions` cuenta familias; la política y los logits de la red necesitan
`env.n_discrete`, derivado de `EnvConfig.n_discrete`. Con `K=4` y `M=4` son 13 acciones.
Las acciones son arrays enteros; no hay una clase `Action` en `types.py` ni un campo
`n_actions` en la dataclass de configuración.

| ID discreto | Familia | Argumento |
|---|---|---|
| `0`, `1`, `2` | crecer, ocultarse, observar | `-1` |
| `3` a `K+2` | transmitir | símbolo de `0` a `K-1` |
| `K+3` | compartir | `-1` |
| `K+4` a `K+M+3` | atacar | ranura de detección de `0` a `M-1` |
| `K+M+4` | refugiarse | `-1` |

`decode_action` acepta escalares o arrays enteros y devuelve arrays de familia y argumento
con la misma forma. Los IDs fuera de rango devuelven `(-1, -1)` y las entradas no enteras
producen `TypeError`. La decodificación se comprueba en ejecución normal, `jit` y `vmap`.
En `step`, acciones inválidas, inasequibles o ataques sin contacto válido son inactivas:
no consumen recursos ni disparan efectos. Compartir exige haber medido antes.

### Estado, observación y diagnósticos

| Estructura | Contenido actual | Uso permitido |
|---|---|---|
| `EnvState` | Tiempo, posiciones, recursos, vivos, emisión, refugio, θ, mediciones y colas | Simulación; copia de diagnóstico para análisis y visualización |
| `Obs` | Recursos/emisión propios, detecciones enmascaradas, medición propia, tiempo desde estallido, vivo e identificador | Entrada exclusiva de las políticas |
| `StepInfo` | Causa de muerte, atacante, ataques preventivos, refugios correctos y estallidos | Diagnóstico y métricas; nunca entrada adicional de la política |

Las estructuras ya son pytrees inmutables de Flax. Se conservan formas estáticas y máscaras
de validez. `N` es el número de civilizaciones, `M` las ranuras de detección y `K` los símbolos.
La versión operativa usa `P=5`, `D=5` y `F=10`. Las colas se dimensionan a partir de la
velocidad y de la máxima distancia del toro; no sobrescriben señales o proyectiles en vuelo.

### Decisiones físicas de la primera implementación

Estas decisiones concretan aspectos que el esqueleto dejaba abiertos. Se mantienen visibles
para revisión científica del equipo; no se presentan como resultados de las hipótesis.

| Aspecto | Regla de la versión inicial |
|---|---|
| Recursos | `initial_resources=1`; crecimiento multiplicativo; costos absolutos; al morir pasan a cero |
| Detección | `p_det(e,d) = detection_e * exp(-detection_decay*d)`; baja 0.05, alta 0.9 y decaimiento 1 |
| Emisiones | Un frente por civilización viva y turno; crecer, transmitir y compartir ponen emisión alta; ocultarse la pone baja |
| Canal | Las emisiones pasivas revelan contacto sin mensaje; `channel_enabled=false` elimina contenido pero conserva el fenómeno físico |
| Cola de señales | `N * (ceil(sqrt(0.5)/signal_speed) + 1)`; una recepción por cruce de frente |
| Cola de proyectiles | La misma fórmula con `proj_speed`; intentos fallidos se consumen al llegar |
| Contactos | Última detección por emisor; hasta M más recientes; TTL de 20 turnos desde emisión; último mensaje persistente |
| Rumbo y distancia | Desplazamiento mínimo exacto del toro después de detectar; el ruido se aplica a mediciones de anomalía |
| Refugio | Protege el paso siguiente por defecto; `refuge_duration=1`; se consume al estallar o al vencer |
| Anomalía | θ = fase, periodo, centro x/y y radio; periodo configurado, fase/centro aleatorios, radio 0.25 |
| Extensiones | Señales costosas, interferencia y barajado todavía producen un error explícito al iniciar |

La proyección es `z = (cos(φ)*fase + sin(φ)*periodo, cos(φ)*cx + sin(φ)*cy)`.
Una dirección deja el centro indeterminado; dos direcciones linealmente independientes lo
recuperan sin ruido. Direcciones paralelas no bastan. El radio es un parámetro conocido de
configuración. Evaluar el valor estratégico de combinar mediciones queda para H2.

La medición almacenada es `[válida, cos(φ), sin(φ), z0, z1]`; su bandera distingue ausencia
de una medición de valor cero. Compartimos esa medición completa. El contenido de señal
es `[tipo, medición...]`: tipo `-1` para emisión pasiva, `0..K-1` para símbolos y `K` para
medición compartida. La detección contiene `[dx, dy, distancia, contenido..., antigüedad]`.
La antigüedad corresponde al mensaje retenido, o a la detección si aún no llegó un mensaje.

`EnvState` incorpora emisión de cada señal, tiempos y contenido de contactos, vencimiento
del refugio e historial de ataques. `StepInfo` añade familias ejecutadas, recepciones y
lanzamientos. Esos diagnósticos se guardan para análisis; no se concatenan a `Obs`.

### Políticas y red

```text
Policy.init_carry(n_agents) -> carry
Policy.act(key, carry, obs) -> (nuevo_carry, acciones)

ActorCriticGRU(carry, obs_flat, agent_id) -> (nuevo_carry, logits, valor)
```

Una política sin memoria usa `None` como carry. `NetworkPolicy` adapta el actor-crítico a
`Policy`; la red por sí sola no implementa ese protocolo. El
entrenamiento conserva logits, valores, probabilidades y carry necesarios para PPO; las
heurísticas no necesitan saber cómo se entrena la red. Los agentes muertos y el reinicio de
episodios requieren máscaras y reinicio de carry verificados en B-04/C-04.

Toda fuente aleatoria recibe una clave JAX explícita y deriva claves con `split`. No se
reutilizan claves para eventos independientes ni se usa `np.random` global. La recompensa
se mantiene exclusivamente como `log(1+r_{t+1}) - log(1+r_t)`, sin bonos por estrategias.

## 3. Persistencia acordada

`train.episodes` y `train.ippo` producen esta distribución. No se crean carpetas con
resultados ficticios. Las corridas existentes se rechazan para evitar sobrescrituras.

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
  (`mode`, inicialmente exploratorio) y la bandera `synthetic`.
- Guardar cada configuración resuelta hace reproducibles los overrides. La ruta YAML original
  no basta: `run_dir` normaliza el nombre y añade un hash de la configuración resuelta,
  excluyendo seed y directorio de salida. La seed ocupa el siguiente nivel.
- Los checkpoints guardan también el estado necesario para reanudar de forma reproducible,
  incluido PRNG y población cuando corresponda. No basta con los pesos de la red.

### Contrato mínimo de episodios

El formato v1 usa nombres de campos de `EnvState`, prefijos `obs_` e `info_`, y claves
`actions`, `rewards`, `done`. `darkforest.trajectories.load_episode` es el lector compartido.
Los lectores pueden trabajar con episodios reales guardados; un fixture científico estable
para los siguientes análisis sigue siendo parte de C-01. El contrato cubre:

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

`analysis.metrics` produce `results.json` con procedencia y métricas básicas. El régimen y
su protocolo de clasificación aún no están implementados. D define ese protocolo antes de observar resultados
(D-04); un dato ausente o una frontera no detectada no se convierten en cero ni en éxito.

### Agregados e imágenes

Los agregados entre seeds/configuraciones se guardarán en `runs/aggregates/<experimento>/`,
con un manifiesto de corridas fuente y subcarpetas `figures/` y `videos/`. Es una salida de
análisis, no una corrida: los lectores deben descubrir corridas por sus metadatos y omitir
esa carpeta al buscar `results.json` por seed.

`runs/` ya está ignorado por Git. También hay exclusiones globales para NPZ, MP4 y GIF. No se
versionan checkpoints, datos crudos ni logs. `docs/assets/` se reserva para PNG/SVG pequeños
seleccionados, con pie que identifique origen, config y seed o agregado. No se crean imágenes
ni resultados ficticios en esta iteración.

## 4. UI y notebooks

`notebooks/fundamentos.ipynb` recorre configuración, acciones y observaciones sintéticas.
`entrenamiento.ipynb` entrena o consulta un checkpoint existente y muestra pérdidas y
evaluaciones. `demo.ipynb` reproduce episodios guardados sin llamar al entorno.
`resultados.ipynb` queda preparado para el análisis experimental posterior. Las dimensiones
ilustrativas de fundamentos no sustituyen el formato físico documentado arriba.

JupyterLab, ipykernel y nbconvert están separados en el grupo `notebooks`. La copia ejecutada
de verificación se guarda en `runs/notebook_checks/`; el notebook versionado no tiene salidas.
Los pasos están en [PRIMEROS_PASOS.md](PRIMEROS_PASOS.md).

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
| Preparación de herramientas por equipo | G-01: reproducir la instalación y las comprobaciones de la guía en cada máquina |
| Decisiones físicas concretadas en la versión inicial | A-01/A-02: revisar su adecuación científica antes de los experimentos finales |
| `proj_speed=0.05`, pero el barrido incluía `signal_speed=0.01` y `0.0316` | A-02/D-03: el mínimo de `signal_speed` pasó a 0.05 y una prueba valida todas las celdas; el régimen de señales más lentas queda sin explorar |
| Reanudación de entrenamiento por CLI y backend W&B | C-02: completar estas rutas; TensorBoard local y checkpoint para inferencia ya funcionan |
| `payoff_matrix(env_cfg, ...)` sugiere simulación dentro de análisis | C-06/D-05: evaluar en `train/` y calcular pagos en `analysis/` desde archivos |
| `costly_signals` sin mecanismo definido | A-06/D-07: especificar y probar costo/verificabilidad antes de H3 |
| Con periodo fijo y `since_burst` público, cooperar no compensa (aporte real de -0.20 a -0.24) | A-07: `anomaly.period_spread` lo lleva a +0.18 a +0.50; detalle en DIAGNOSTICO_DISENO.md |
| El log de entrenamiento no guarda el retorno por iteración | C-07: registrar retorno y supervivencia para tener curva de aprendizaje |
| CI y protocolo empírico H1 | D-02/D-04: verificar CI y definir comparadores; los invariantes locales ya tienen aserciones reales |
| Cinco seeds en el YAML también para el grid grueso | D-03: selección explícita de tres para exploración; ≥5 para resultados reportados |

Antes de entrenar se requieren recursos no negativos, muertos inactivos, velocidad correcta
de señales, recompensa auditada, reproducibilidad y `jit`/`vmap`. B verifica el protocolo y
carry; C prueba GAE y actualización PPO; D verifica métricas con ejemplos conocidos y E
comprueba alineación temporal de la reproducción.

La prueba de integración I-01 recorre simulación → guardado → carga → métricas → reproducción.
H1 es una evaluación empírica posterior; una ejecución sin errores no confirma la hipótesis.
Los resultados inesperados se investigan sin cambiar la recompensa para imponer estrategias.

La primera ruta integrada y sus límites se describen en [EPISODIOS.md](EPISODIOS.md).
IPPO inicial usa `rho=0`, secuencias recurrentes completas y varias épocas por rollout.
Autojuego ficticio, torneos, barridos y clasificación científica siguen en las tareas futuras.

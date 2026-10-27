# Episodios y primer aprendizaje

Continuamos el trabajo del esqueleto original usando sus módulos, `Policy`, `Obs`,
`EnvState`, GRU e IPPO. Primero verificamos los fundamentos; ahora podemos simular,
guardar, analizar y reproducir episodios, y ejecutar un entrenamiento corto en CPU.
Los criterios generales siguen en [TAREAS.md](TAREAS.md) y el avance en [TODO.md](TODO.md).

## Qué ejecutamos y para qué

| Actividad | Qué ocurre | ¿Cambia los pesos? |
|---|---|---|
| Simular un episodio | Una política toma decisiones en un mundo nuevo | No |
| Entrenar | Generamos experiencias y aplicamos actualizaciones de PPO a una GRU | Sí |
| Evaluar un checkpoint | La red guardada actúa en un episodio nuevo | No |
| Reproducir un episodio | Leemos los estados previamente guardados | No |

No tenemos que ejecutar episodios manualmente antes de cada entrenamiento. El entrenador
los genera mientras recoge fragmentos de experiencia. Una actualización utiliza un rollout;
un rollout puede terminar antes que el episodio o contener el inicio de otro.

## Preparación

Desde la raíz del repositorio, en PowerShell:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync --locked --group notebooks
uv run pytest
```

CPU basta para los ejemplos cortos. TensorBoard ya forma parte de las dependencias.
Jupyter, los módulos de Python y las pruebas utilizan el mismo entorno de uv.

## Nuestro primer episodio

```bash
uv run python -m train.episodes --config configs/aprendizaje_dos.yaml --policy random --name episodio-dos
```

La consola muestra la ruta del NPZ. Para cuatro civilizaciones usamos
`configs/aprendizaje_cuatro.yaml` y otro nombre, por ejemplo `episodio-cuatro`.
También podemos seleccionar `hidden`, `transmit`, `reciprocal` o `attacker`.
Estas son heurísticas; no son modelos aprendidos.

Con la configuración de esta entrega, el episodio de dos se guarda en
`runs/episodio-dos-cf926437c3/0/trajectories/episode_000001.npz`.
El sufijo depende de los parámetros resueltos; si cambiamos un parámetro, usamos la nueva
ruta que imprime la consola. Un nombre y una configuración ya existentes no se sobrescriben.
Para otra ejecución elegimos otro `--name` o una seed diferente mediante `train.seed=1`.

Para verlo abrimos el notebook, seleccionamos un índice de su lista y ejecutamos sus celdas:

```bash
uv run --group notebooks jupyter lab notebooks/demo.ipynb
```

También podemos analizarlo y exportar un GIF desde archivos:

```bash
uv run python -m analysis.metrics --episode runs/episodio-dos-cf926437c3/0/trajectories/episode_000001.npz
uv run python -m viz.galaxy --episode runs/episodio-dos-cf926437c3/0/trajectories/episode_000001.npz --fps 8
```

La animación muestra posiciones fijas, señales, proyectiles, estallidos y recursos.
Los puntos representan civilizaciones vivas y las cruces representan eliminadas.
Esta vista global usa diagnósticos guardados; la red solo recibe observaciones parciales.

## Entrenamos con dos y cuatro civilizaciones

El recorrido guiado está en [entrenamiento.ipynb](../notebooks/entrenamiento.ipynb):

```bash
uv run --group notebooks jupyter lab notebooks/entrenamiento.ipynb
```

Comenzamos con `config_name = "aprendizaje_dos"`. La celda de entrenamiento guarda el modelo;
si el checkpoint ya existe, lo consulta y no vuelve a entrenar. Para cuatro cambiamos el
nombre a `aprendizaje_cuatro`, reiniciamos el kernel y ejecutamos todas las celdas.

Los comandos equivalentes son:

```bash
uv run python -m train.ippo --config configs/aprendizaje_dos.yaml
uv run python -m train.ippo --config configs/aprendizaje_cuatro.yaml
```

| Parámetro | Valor inicial | Significado |
|---|---|---|
| `n_civs` | 2 o 4 | Civilizaciones dentro de cada mundo |
| `n_envs` | 4 | Mundos independientes recogidos en paralelo |
| `max_steps` | 48 | Límite de transiciones de cada episodio |
| `rollout_len` | 16 | Pasos consecutivos de cada fragmento |
| `total_steps` | 2048 | Pasos de entorno sumados entre los cuatro mundos |
| `update_epochs` | 3 | Pasadas de optimización sobre cada fragmento |
| `gru_hidden` | 32 | Tamaño de la memoria recurrente |
| `rho` | 0 | Entrenamos la política actual sin población de snapshots |
| Letalidad | 0 | Control inicial sin muertes por anomalía |

Este presupuesto produce 32 rollouts y 96 pasos del optimizador. Las acciones no reciben
bonos por cooperar, atacar o esconderse: la recompensa es únicamente la diferencia de
`log1p` de recursos. Al morir, los recursos pasan a cero y la civilización deja de actuar.

La GRU conserva memoria entre fragmentos del mismo episodio. Al morir un agente o reiniciar
un mundo, reiniciamos su memoria. GAE corta la acumulación al acabar el episodio; en un
límite de tiempo mantiene el valor estimado del último estado, y en una muerte no lo mantiene.
La pérdida se calcula sobre agentes vivos controlados por la política actual.

El primer entrenador usa secuencias completas de cada rollout y varias épocas PPO,
sin minibatches ni población de oponentes. Pedir `rho > 0` produce un error explícito.
El presupuesto debe ser múltiplo de `n_envs * rollout_len`; no lo ampliamos silenciosamente.

## Qué guardamos

Cada corrida queda en `runs/<nombre>-<hash>/<seed>/`:

| Archivo | Contenido |
|---|---|
| `config.yaml` | Parámetros resueltos, incluida la seed |
| `metadata.json` | Versión del formato, revisión Git, cambios locales y entorno |
| `training.jsonl`, `tensorboard/` | Pérdida, entropía, error del crítico y norma del gradiente |
| `evaluation.json` | Cambio de parámetros y comparación exploratoria antes/después |
| `checkpoints/final.msgpack` | Pesos, optimizador, clave aleatoria, memoria y estado de los mundos |
| `trajectories/episode_000001.*` | Episodio de la red inicial, antes de entrenar |
| `trajectories/episode_000002.*` | Episodio de la red después de entrenar |
| `results.json`, `videos/` | Métricas y animaciones generadas posteriormente |

Los dos episodios de evaluación utilizan la misma clave para facilitar su comparación.
El NPZ tiene arrays numéricos, T acciones/recompensas y T+1 estados/observaciones; no usa
pickle. Su JSON asociado registra formas, tipos, política, configuración y seed.
`load_episode` comprueba ese contrato antes de entregar datos al análisis o la visualización.

Para generar otro episodio con pesos ya aprendidos:

```bash
uv run python -m train.episodes --checkpoint runs/aprendizaje_dos-cf926437c3/0/checkpoints/final.msgpack --name evaluacion-dos --seed 1
```

Para ver curvas locales de entrenamiento:

```bash
uv run tensorboard --logdir runs
```

Al evaluar un checkpoint elegimos otra seed sin alterar sus pesos. Guardamos el hash del
checkpoint y su seed de entrenamiento en los metadatos. La salida va al `runs` local;
`--out-dir` permite elegir otro destino, incluso si el archivo recibido conserva una ruta
de salida de la máquina de un compañero.

El checkpoint incluye estado para una futura reanudación, pero esta entrega implementa
su carga para inferencia. La CLI para continuar un entrenamiento interrumpido queda pendiente.
No cambiamos el tamaño del mundo al cargar un checkpoint: una comparación de generalización
de dos a cuatro civilizaciones requiere su propio protocolo.

## Cómo lo compartimos entre compañeros

Después de fusionar el PR, cada integrante actualiza `main` e instala el mismo lock:

```bash
git switch main
git pull --ff-only
uv sync --locked --group notebooks
uv run pytest
```

Compartimos código, configuraciones, notebooks y documentación mediante Git. Compartimos
las carpetas de corridas por el medio acordado por el equipo, fuera de Git, conservando
la estructura `runs/<config>/<seed>/` y sus metadatos. No enviamos solo el NPZ: el lector
necesita también el JSON correspondiente.

- Para analizar o reproducir, copiamos las trayectorias y sus metadatos; no entrenamos otra vez.
- Para generar episodios con un modelo existente, copiamos también el checkpoint y `config.yaml`.
- Para verificar reproducibilidad o realizar un experimento nuevo, ejecutamos el entrenador.
- Repartimos configuraciones y seeds entre compañeros; no hace falta duplicar cada corrida.

Cada integrante puede ejecutar una prueba corta de instalación. Los resultados científicos
requieren al menos cinco seeds por configuración reportada. Ejecutar una demo no satisface
ese requisito.

## Alcance, comprobaciones y siguientes entregas

El 27 de octubre de 2026 se ejecutaron 605 pruebas sin fallos, omitidos ni fallos esperados.
Incluyen física, recompensa, pytrees, GAE calculado a mano, memoria recurrente, actualización
real, recarga del checkpoint y simulación → guardado → análisis → GIF para dos y cuatro.
La inicialización también se comprueba con seis y ocho; no se realizaron experimentos H4.

Las dos corridas exploratorias de seed 0 completaron 2048 pasos y 96 actualizaciones cada una:

| Civilizaciones | Norma del cambio de pesos | Retorno medio inicial | Retorno medio final |
|---|---|---|---|
| 2 | 0.6320 | -0.3009 | -0.1512 |
| 4 | 0.7151 | -0.6110 | -0.6188 |

La red de cuatro cambió sus pesos y obtuvo un retorno ligeramente peor en el episodio
evaluado. No ocultamos ese resultado ni lo interpretamos como un fallo del código por sí solo.
Estas comprobaciones muestran que el ciclo funciona; no establecen que una estrategia haya
convergido ni que H1/H2 estén apoyadas. Las corridas conservan `git_dirty: true` porque se
ejecutaron durante la integración; los experimentos reportables deberán partir de un commit limpio.

La propuesta contempla 2, 4, 6 y 8 civilizaciones. Priorizamos 2 y 4; dejamos los experimentos
de 6 y 8 para escala y coaliciones H4, sin comprometer el cierre del 8 de noviembre y la
presentación del 9 al 13. No entrenamos cuatro tamaños obligatoriamente en cada cambio.

Siguen pendientes el autojuego ficticio, torneos contra arquetipos, reanudación por CLI,
barajado de mensajes, señales costosas, barridos válidos, replicador, información mutua y
clasificación de regímenes. Conservamos la evaluación empírica de H1 separada de los tests.

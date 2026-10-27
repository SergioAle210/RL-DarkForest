# Primera iteración: fundamentos de Sergio

Esta entrega permite comprobar la configuración y el contrato de acciones antes de
construir el simulador. Se conservaron las interfaces y estructuras existentes para que
entorno, políticas, entrenamiento, análisis y visualización puedan compartirlas.
El equipo tiene cinco integrantes en total; Sergio realiza esta base común.

## Qué contiene y por qué

| Componente | Qué hace | Motivo |
|---|---|---|
| Configuración | Valida dataclasses, YAML heredados y overrides; distingue errores de clave, tipo y valor | Detectar parámetros incoherentes antes de una corrida |
| Acciones | Deriva `n_discrete` y convierte cada ID entero en familia y argumento | Dar a políticas y entorno una codificación común |
| Tipos y pruebas | Comprueba inmutabilidad, pytrees, máscaras y ejecución JAX con datos sintéticos | Proteger los contratos antes de agregar dinámica física |
| `fundamentos.ipynb` | Recorrido reproducible sobre los módulos reales | Aprender a usarlos y comprobar cambios pequeños |
| Archivos preparados | Notebooks futuros válidos y módulos de prueba con docstring | Reservar lugares claros para los siguientes frentes |
| Documentación | Propuesta, contratos, calendario y seguimiento único en `docs/` | Mantener decisiones y avance separados del código |

`reset`, `step`, la anomalía, las políticas, la GRU, IPPO, la persistencia y la animación
siguen como entregas posteriores. Esta primera iteración no genera episodios, imágenes de
simulación, checkpoints ni un modelo entrenado.

## Preparar el entorno

Abre una terminal en la raíz del repositorio. La preparación local utiliza uv 0.13.6 y
Python 3.12.10; `uv.lock` fija las dependencias del proyecto.

```bash
uv --version
uv sync --locked --group notebooks
uv run python --version
uv run --group notebooks pytest tests/test_config.py tests/test_actions.py tests/test_types.py
uv run ruff check .
uv run ruff format --check .
```

Si acabas de instalar uv y PowerShell no lo encuentra, abre una terminal nueva. También
puedes incorporar el directorio de instalación del usuario a la sesión actual:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

El grupo `notebooks` añade JupyterLab, ipykernel y nbconvert. No se instalan dependencias
desde las celdas. La instalación editable del proyecto permite importar `darkforest` desde
la raíz o desde `notebooks/`, sin modificar `sys.path`.

## Tu recorrido en fundamentos

```bash
uv run --group notebooks jupyter lab notebooks/fundamentos.ipynb
```

1. Selecciona el kernel Python del entorno creado por uv; reinícialo y ejecuta todas las celdas.
2. Comprueba las versiones y los dispositivos JAX mostrados. CPU es suficiente para este bloque.
3. Carga `configs/h1_sanity.yaml`: debe tener dos civilizaciones y letalidad cero.
4. Revisa los overrides: cambian la copia cargada sin alterar el YAML ni la configuración previa.
5. Inspecciona las 13 acciones con cuatro símbolos y cuatro ranuras; los argumentos empiezan
   en cero y las acciones sin argumento usan `-1`.
6. Ejecuta los errores esperados. Las celdas los capturan y verifican su clase; no deben
   interrumpir el recorrido.
7. Observa las máscaras sintéticas: `False` indica relleno, y `Obs` no incluye θ ni diagnósticos.
   `.replace` produce una estructura nueva sin cambiar la original.
8. En el ejercicio, usa dos símbolos y verifica que el catálogo tenga 11 acciones. Después
   prueba seis civilizaciones y una seed distinta dentro de `0..4294967295`.
9. Reinicia el kernel y vuelve a ejecutar todo. Antes de versionar cambios, limpia las salidas.

Las dimensiones de medición y detección del ejemplo son ilustrativas. Las dimensiones
físicas, las capacidades de colas, las detecciones persistentes y las reglas del refugio
se decidirán al comenzar el entorno.

## Reproducir la comprobación completa

Desde la raíz, esta ejecución usa un kernel nuevo y guarda otra copia, sin escribir salidas
en el notebook versionado:

```bash
uv run --group notebooks jupyter nbconvert --to notebook --execute notebooks/fundamentos.ipynb --output fundamentos-ejecutado.ipynb --output-dir runs/notebook_checks --ExecutePreprocessor.timeout=180
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Verificación realizada el **27 de octubre de 2026**, con Python 3.12.10 y uv 0.13.6:

| Comprobación | Resultado |
|---|---|
| Instalación con `uv sync --locked --group notebooks` | Correcta; se conservaron las versiones previamente fijadas |
| Pruebas de configuración, acciones y tipos | 564 aprobadas |
| Suite completa | 565 aprobadas, 3 omitidas y 4 fallos esperados; sin fallos inesperados ni `XPASS` |
| Ruff y formato | Correctos en los 41 archivos revisados |
| Notebook desde kernel nuevo | Ocho celdas de código ejecutadas sin errores |
| Documentación y preparación | 26 enlaces locales válidos, cuatro notebooks válidos y cinco módulos de prueba con solo docstring |
| Seguimiento | 46 IDs conservados en 63 tareas; 16 de Sergio y 47 futuras sin asignación |

La prueba adicional de la suite solo comprueba que los arquetipos exponen los métodos de
`Policy`; no demuestra que sus acciones estén implementadas. Los tres casos omitidos
corresponden a agentes eliminados, recursos y señales. Los cuatro fallos esperados cubren
`reset`, `step`, recompensa y el marcador de H1. Ninguno acredita un episodio funcional.

La copia ejecutada está en `runs/notebook_checks/fundamentos-ejecutado.ipynb`; el resumen
de comprobaciones documentales está en `runs/notebook_checks/delivery-checks.json`.
Ambos permanecen fuera de Git. Los cinco módulos de pruebas futuros no generan casos
aprobados. El historial de esta entrega se puede consultar con:

```bash
git log --oneline --reverse 7bea76d..HEAD
```

## Cómo llevar el seguimiento

[TODO.md](TODO.md) es la única fuente de avance. Para las tareas de Sergio, usa `pendiente`,
`en proceso` o `listo`, y cambia la fecha solo al cambiar realmente el estado. Una tarea
termina cuando cumple el criterio de [TAREAS.md](TAREAS.md). Las tareas futuras sin asignar
conservan responsable, estado y fecha vacíos. Preparar un notebook o archivo de pruebas
no significa implementar su módulo.

Los 46 identificadores de TAREAS se conservan. A-01 se divide entre acciones/tipos y diseño
físico pendiente; A-02 entre validación y parámetros físicos/barrido; A-03 entre
decodificación y simulación. G-01 distingue preparar esta máquina de preparar las demás.
C-02 separa persistencia y TensorBoard; C-04 separa recolección y actualización. El resto
mantiene su cobertura, con nombres descriptivos en TODO y detalle técnico en TAREAS.

## Próximo bloque: un episodio reproducible

1. Cerrar dimensiones, recursos iniciales, colas, detecciones y duración del refugio.
2. Implementar `reset`, `step`, observaciones parciales y recompensa para dos civilizaciones.
3. Probar recursos no negativos, muertos inactivos, propagación, recompensa,
   reproducibilidad y compatibilidad con `jit`/`vmap`.
4. Implementar política aleatoria y arquetipos a través de `Policy`.
5. Guardar configuración, seed y trayectoria, calcular métricas y reproducirla con Matplotlib.

La salida de ese bloque será **simulación → guardado → análisis → reproducción**. Solo
entonces habrá un episodio real para abrir en `demo.ipynb`. Los notebooks
`entrenamiento.ipynb` y `resultados.ipynb` se completarán cuando sus módulos estén listos.
La visualización lee archivos guardados y no vuelve a ejecutar el entorno.

El barrido actual conserva combinaciones con `signal_speed < proj_speed`; el cargador las
rechaza. `train.sweep --dry-run` todavía solo expande combinaciones. Antes de entrenar hay
que revisar ese diseño, completar el ejecutor y distinguir tres seeds exploratorias de
al menos cinco por configuración reportada. TensorBoard sigue pendiente de implementar
y verificar como parte de la persistencia.

## Cómo aprenderá el modelo

1. El entorno producirá una observación parcial por civilización y recibirá sus acciones.
   Su recompensa será únicamente `log1p(recursos_nuevos) - log1p(recursos_previos)`,
   con el tratamiento de agentes eliminados previsto en la arquitectura.
2. Una red compartida con GRU mantendrá memoria de lo observado. El actor producirá logits
   para las acciones discretas y el crítico estimará el retorno futuro.
3. El ejecutor recogerá secuencias de varios entornos. Una actualización puede usar un
   fragmento de episodio; terminar un rollout no equivale a terminar un episodio.
4. GAE estimará qué acciones resultaron mejores o peores de lo esperado; PPO actualizará
   los parámetros con esas ventajas, valores y probabilidades.
5. La memoria se conservará entre fragmentos del mismo episodio y se reiniciará al
   terminar o cuando corresponda por agente. Se distinguirán terminaciones reales y
   cortes de recolección para calcular retornos y máscaras correctamente.
6. El autojuego mezclará la política actual con versiones congeladas como oponentes.
   Solo los agentes de la política actual contribuirán a su actualización.
7. Se evaluarán checkpoints contra políticas conocidas y arquetipos. Las trayectorias
   guardadas alimentarán métricas, replicador, diagramas de fases y reproducción visual.

Cooperar o atacar no recibe premios extra. H1 será evidencia empírica separada de los
tests deterministas; un resultado inesperado se investigará sin cambiar la recompensa
para forzar la hipótesis. Se priorizan H1–H2 y se mantiene el cierre del 8 de noviembre
para presentar del 9 al 13 de noviembre de 2026.

# Diagnóstico de diseño con políticas fijas

Este documento justifica los cambios al entorno de la rama y fija el criterio para decidir
otros. Todas las cifras salen de dos módulos del repositorio, con políticas escritas a mano y
sin entrenar. Sirven para saber si las reglas dejan algo que aprender. No son evidencia de
aprendizaje ni reemplazan a IPPO: los resultados finales deben salir de agentes entrenados.

Medidas: retorno acumulado medio por civilización, dos civilizaciones, 200 pasos, 400 episodios
por seed, 5 seeds; el ± es el error estándar entre seeds. Fecha: 27 de octubre de 2026.

## Cómo reproducir

```bash
uv run python -m train.valor_cooperacion --world periodo_fijo
uv run python -m train.valor_cooperacion --world periodo_aleatorio
uv run python -m train.valor_cooperacion --world periodo_aleatorio --lethality 0
uv run python -m train.valor_cooperacion --world periodo_aleatorio \
    --signal-speed 0.1 0.3 1.0 --cost-tx 0.02 0.01 0
uv run python -m train.matriz_pagos --lethality 0
uv run python -m train.matriz_pagos --lethality 0 --cancel --p-kill 0.75
uv run python -m train.matriz_pagos --lethality 0 --cancel --p-kill 1.0
```

Las pruebas con valores calculables están en `tests/test_diagnostico.py`.

## Políticas comparadas

| Política | Qué sabe o hace |
|---|---|
| ignorar | Solo crece |
| solo | Mide una vez y usa su propia medición y los estallidos que ve por `since_burst` |
| coop_ideal | Combina las dos mediciones al instante y sin costo (cota superior) |
| coop_real | Comparte con la acción `compartir`: paga `cost_tx`, la señal tarda según `signal_speed` y se entrega con probabilidad; si no recibe, actúa como solo |
| oraculo | Conoce θ exacto (techo) |

El margen de la zona se elige en una seed de ajuste distinta de las cinco de evaluación.

## Hallazgo 1: con periodo fijo cooperar no compensa

Con el periodo fijo en 20 y `since_burst` público, una civilización sola ya despeja la fase con
su propia medición, así que casi no queda nada que ganar compartiendo.

| Mundo | λ | solo | coop_ideal | coop_real | aporte ideal | aporte real |
|---|---|---|---|---|---|---|
| periodo fijo | 0.25 | 8.308 | 8.459 | 8.105 | +0.151 ± 0.032 | -0.203 ± 0.028 |
| periodo fijo | 0.5 | 8.233 | 8.387 | 8.013 | +0.154 ± 0.032 | -0.219 ± 0.025 |
| periodo fijo | 1.0 | 8.138 | 8.302 | 7.897 | +0.164 ± 0.038 | -0.241 ± 0.023 |
| periodo aleatorio | 0.25 | 7.764 | 8.391 | 7.946 | +0.627 ± 0.065 | +0.183 ± 0.060 |
| periodo aleatorio | 0.5 | 7.272 | 8.307 | 7.765 | +1.035 ± 0.086 | +0.493 ± 0.081 |
| periodo aleatorio | 1.0 | 6.954 | 8.209 | 7.456 | +1.255 ± 0.075 | +0.502 ± 0.063 |

"Aporte" es la diferencia contra solo. Con periodo fijo el aporte real es negativo: compartir
cuesta más de lo que informa. Con periodo aleatorio por episodio (`anomaly.period_spread: 8`,
periodo entre 12 y 28) un agente solo no puede deducir cuándo es el estallido sin sufrir los
dos primeros, y el aporte real pasa a ser positivo y creciente con λ hasta 0.5. De 0.5 a 1.0
queda plano dentro del error.

El techo del oráculo es 8.962 con periodo fijo y 8.955 con periodo aleatorio.

## Hallazgo 2: el aporte depende de la velocidad de señal y del costo de transmitir

Aporte de coop_real sobre solo en el mundo de periodo aleatorio, media de 5 seeds. Este barrido
no calcula errores estándar; en la tabla anterior el error del aporte real fue de 0.06 a 0.08,
así que diferencias de 0.05 entre celdas vecinas están dentro del ruido.

| `signal_speed` | `cost_tx` | λ = 0.25 | λ = 0.5 | λ = 1.0 |
|---|---|---|---|---|
| 0.1 (actual) | 0.02 (actual) | +0.183 | +0.493 | +0.502 |
| 0.1 | 0.01 | +0.236 | +0.545 | +0.552 |
| 0.1 | 0 | +0.286 | +0.595 | +0.600 |
| 0.3 | 0.02 | +0.282 | +0.629 | +0.751 |
| 0.3 | 0.01 | +0.325 | +0.671 | +0.792 |
| 0.3 | 0 | +0.365 | +0.711 | +0.831 |
| 1.0 | 0.02 | +0.317 | +0.688 | +0.843 |
| 1.0 | 0.01 | +0.359 | +0.730 | +0.884 |
| 1.0 | 0 | +0.400 | +0.769 | +0.923 |

- Bajar `cost_tx` suma de forma pareja cerca de +0.05 por cada 0.01 menos.
- Subir `signal_speed` ayuda más con λ alto (de +0.50 a +0.84 en λ = 1.0) que con λ bajo
  (de +0.18 a +0.32 en λ = 0.25).
- Con señal rápida el aporte crece con λ en las tres columnas; con la señal actual queda
  plano entre 0.5 y 1.0.
- Es la forma cualitativa que predice H2: más demora vuelve menos valiosa la cooperación y más
  letalidad la vuelve más valiosa. No es una confirmación de H2, que requiere agentes entrenados
  y la comparación contra el miedo.
- `signal_speed` es un eje del diagrama de fases (λ por c), así que no se fija en un valor
  para mejorar el resultado.
- Incluso en el mejor caso (señal 1.0, sin costo) el aporte real queda por debajo de la cota
  ideal con λ = 1.0 (+0.92 contra +1.26). No se aisló la causa; los candidatos son los pasos de
  medir y compartir sin crecer, la entrega probabilística y el primer estallido, que puede
  llegar antes del mensaje.

## Hallazgo 3: los pagos del ataque cierran el juego

Matriz con λ = 0, crecedor puro, silencio puro y un atacante que dispara una vez al primer
contacto y vuelve a crecer. Retorno de la fila contra la columna, entre paréntesis la
supervivencia de la fila.

Reglas actuales (`projectiles_cancel_on_death` apagado), `p_kill` 0.75:

| fila / columna | crecer | silencio | ataque único |
|---|---|---|---|
| crecer | +9.065 (1.00) | +9.065 (1.00) | +1.707 (0.25) |
| silencio | 0.000 (1.00) | 0.000 (1.00) | -0.512 (0.26) |
| ataque único | +8.935 (1.00) | +8.981 (1.00) | +1.680 (0.25) |

Crecer es la mejor respuesta a las tres columnas. Contra un crecedor, atacar cuesta 0.13
(`cost_atk`) y no da nada. Contra un atacante, atacar y crecer empatan dentro del error. El
silencio pierde incluso contra un atacante seguro, porque la detección baja (0.05 por la
distancia) descubre a casi todos en 200 pasos. Con estas reglas el resultado ya está en las
reglas: nadie tiene motivo para callarse ni para atacar.

Causa de fondo: con recompensa logarítmica y crecimiento exponencial, sobrevivir 200 pasos
creciendo vale cerca de 9 y morir cuesta menos de 1 (retorno total de -log 2), así que
renunciar a crecer solo conviene si la probabilidad de sobrevivir creciendo es inferior a
cerca de 7%.

Con `projectiles_cancel_on_death` activado (el primer impacto cancela los proyectiles en vuelo
del eliminado):

| `p_kill` | crecer contra ataque | ataque contra ataque | umbral de probabilidad de atacante |
|---|---|---|---|
| 0.75 | +1.707 | +3.062 (0.39) | 9% |
| 1.0 | -0.693 (0.00) | +1.848 (0.26) | 5% |

El umbral es la probabilidad de que el otro sea atacante a partir de la cual atacar primero
rinde más que crecer: q = (pérdida de atacar contra un crecedor) / (esa pérdida más la ganancia
de atacar contra un atacante), calculado con las celdas de las tablas. Con la regla activada
hay dos equilibrios, crecimiento mutuo y ataque mutuo, y ninguna estrategia domina. Es la
estructura de un juego de coordinación, pero no es un objetivo de diseño: es lo que ocurre al
cambiar esa regla.

## Hallazgo 4: el costo de cooperar depende de cuánto se ve quien crece

Hasta aquí cooperar nunca costaba exposición, porque crecer ya emitía en alta. El parámetro
`env.detection_growth` fija la probabilidad de detección de quien crece (sin valor, igual a
`detection_high`, el comportamiento anterior). Transmitir y compartir siguen emitiendo en alta
solo en el paso en que se hacen. Se midieron tres escenarios con λ = 0.5, `p_kill` 0.75 y la
cancelación de proyectiles apagada. Estrategias: solo (crece y se protege de la anomalía),
coop_real (comparte mediciones) y ataque_solo (como solo, pero dispara una vez al primer
contacto). Retorno de la fila contra la columna; la columna "coop" es un compañero que
también coopera.

| Escenario | solo contra solo | coop contra coop | solo contra atacante | coop contra atacante | atacante contra atacante |
|---|---|---|---|---|---|
| actual (crecer emite en alta) | +7.196 | +7.754 | +1.435 | +1.340 | +1.421 |
| `detection_growth` 0.05 | +7.196 | +7.754 | +2.052 | +1.331 | +3.455 |
| `detection_low` y `detection_growth` 0.001 | +7.196 | +7.754 | +6.539 | +1.340 | +6.557 |

Probabilidad de atacante a partir de la cual cooperar deja de compensar, calculada con las
celdas anteriores: q = (aporte de cooperar con un compañero que coopera) / (ese aporte más lo
que se pierde al cooperar con un atacante). El aporte es 7.754 - 7.196 = 0.558.

| Escenario | Pérdida al cooperar con un atacante | Umbral de probabilidad de atacante |
|---|---|---|
| actual | 0.095 | 85% |
| `detection_growth` 0.05 | 0.721 | 44% |
| 0.001 | 5.199 | 10% |

- Con las reglas actuales cooperar casi no expone: es lo que hacía trivial la pregunta del
  proyecto. Bajar la detección de quien crece hace que compartir sea el único acto visible, y
  el umbral de confianza necesario para cooperar baja de 85% a 10%. Es el "miedo" que la
  propuesta pide comparar con el valor de la cooperación.
- Con detección muy baja casi nadie descubre a nadie (solo contra atacante rinde +6.539, cerca
  de +7.196) y cooperar equivale a gritar: el atacante detecta los envíos y mata. Es la
  estructura del bosque oscuro.
- Con 0.05 aparece una ventaja de primer golpe sin cancelar proyectiles: atacar contra un
  atacante rinde +3.455 y sobrevive 0.44, contra +2.052 y 0.29 de un crecedor. Los atacantes
  se detectan en momentos muy distintos y quien detecta primero mata al otro antes de que
  detecte y dispare (0.60 disparos por episodio contra 0.80 si solo uno ataca). Con detección
  alta ambos se detectan casi a la vez y no hay ventaja.
- No se eligió el valor de `detection_growth`: son tres puntos, no un barrido. Qué valores entran
  al barrido principal queda por decidir y debe reportarse con sensibilidad.

## Comprobación de la H1 reformulada

H1 dice que con λ = 0 la cooperación no tiene valor. Con políticas fijas:

| Mundo | ignorar | solo | coop_ideal | coop_real |
|---|---|---|---|---|
| periodo fijo | +9.065 | +8.837 | +8.902 | +8.552 |
| periodo aleatorio | +9.065 | +8.863 | +8.896 | +8.548 |

Ninguna política que usa el canal supera a ignorar (+9.065), y compartir cuesta cerca de 0.5.
Que coop_ideal supere a solo por +0.03 o +0.07 no contradice H1: solo se refugia sin necesidad
con λ = 0 y la política con información exacta lo hace menos veces. La comparación válida es
contra ignorar. Falta la comprobación con agentes entrenados (ablación del canal con λ = 0).

## Principio de diseño

1. Las reglas del juego son decisión nuestra. Los comportamientos que aparezcan (cooperar,
   callar, atacar, coaliciones) deben salir del aprendizaje, no de la elección de reglas.
2. Los arreglos de fidelidad, es decir, hacer que el código cumpla lo que la propuesta ya
   dice, no son sesgo. Ejemplo pendiente: la propuesta dice que transmitir y compartir "revelan
   la posición" y hoy no exponen más que crecer.
3. Las mecánicas que la propuesta no especifica (qué pasa con los recursos de la víctima, con
   los proyectiles de un eliminado, si crecer emite en alto) no se eligen por su efecto.
   Se implementan como parámetros, con el valor actual por defecto, y se reportan con
   sensibilidad: resultados con y sin cada regla.
4. Todo cambio posterior a ver datos se declara en el informe. Los cambios de esta rama
   fueron motivados por estos diagnósticos.

Cambios implementados, todos apagados o neutros por defecto:

| Parámetro | Efecto | Valor por defecto y motivo |
|---|---|---|
| `anomaly.period_spread` | El periodo se sortea por episodio en [period - k, period + k] | 0 en el código (comportamiento anterior) y 8 en `configs/base.yaml`. Es un arreglo de fidelidad: la propuesta define θ con el periodo como parámetro oculto (sección 3.2) y prevé el riesgo "el canal se ignora", que ocurre si θ es identificable con una sola proyección. Con periodo fijo lo era. El valor 8 es libre |
| `env.projectiles_cancel_on_death` | El proyectil de una civilización eliminada no impacta | False: un proyectil ya lanzado sigue su camino aunque muera quien lo lanzó. Una civilización muerta nunca lanza ataques nuevos en ningún caso |
| `env.detection_growth` | Probabilidad de detección de quien crece; transmitir y compartir siguen emitiendo en alta en su paso | Sin valor (igual a `detection_high`), el comportamiento anterior. Con un valor bajo crecer es silencioso y cooperar expone |

Pendientes de decisión: qué valores de `detection_growth` y `detection_low` entran al barrido
(por defecto no cambia nada, pero la sección 3.3 de la propuesta dice que crecer pone la
emisión en alta, y bajarla es un cambio de esa regla que debe declararse), el botín al atacar y
la detección garantizada al transmitir o compartir. Con 200 pasos casi todo emisor es descubierto
si emite a 0.05 por la distancia (cerca de 99.9%), por eso la exposición de cooperar solo pesa
con una detección base muy baja.

## Límites

- Son políticas fijas, no equilibrios aprendidos. Un entrenamiento puede encontrar
  estrategias condicionales o de tiempo que estas no ven.
- Solo dos civilizaciones, `p_kill` 0.75 y 1.0, y una configuración de detección.
- coop_real es una política hecha a mano (observa una vez, comparte hasta recibir o hasta
  agotar 6 pasos); otra mejor podría acercarse más a coop_ideal.
- El atacante único dispara al primer contacto; otras políticas de ataque pueden dar otros
  pagos.
- No se modeló botín ni defensa contra proyectiles.

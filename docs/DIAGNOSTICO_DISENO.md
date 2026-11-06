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
uv run python -m train.pagos_rejilla --out-dir runs/pagos --set env.detection_growth=0.05
uv run python -m train.ablacion_canal --out-dir runs/ablacion_canal
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

## Hallazgo 5: el replicador y las masas críticas

Se midió una matriz de pagos por celda (λ de 0.25, 0.5 y 1.0 por c de 0.05, 0.22 y 1.0) con cuatro
estrategias: oculta, solo, coop_real y ataque_solo, con `detection_growth` 0.05, 3 seeds y 200
episodios por pareja (menos que las otras tablas, así que el ruido es mayor). No son los cuatro
arquetipos del repositorio, que son toscos (el que siempre transmite no crece dos de cada tres
pasos); se usan las estrategias de los diagnósticos anteriores.

```bash
uv run python -m train.pagos_rejilla --out-dir runs/pagos --set env.detection_growth=0.05
```

En las nueve celdas el replicador deja tres resultados estables: todos solos, todos cooperan y
todos atacan. Cuál se alcanza depende de la mezcla inicial, de modo que es un juego de
coordinación y no de dominio. La fracción de poblaciones iniciales aleatorias que terminan en
ataque (entre 70% y 88%) no es un resultado: depende solo de que una mezcla uniforme incluye
muchos atacantes. Por eso se reporta la masa crítica, es decir, la fracción de una estrategia a
partir de la cual vence a otra en el juego de esas dos. Mediana y rango del 5% al 95% al propagar
el error de cada pago:

| λ | c | Cooperadores necesarios frente a solos | Atacantes necesarios frente a cooperadores |
|---|---|---|---|
| 0.25 | 0.05 | 0.64 [0.00, 1.00] | 0.13 [0.00, 0.24] |
| 0.5 | 0.22 | 0.33 [0.00, 0.53] | 0.30 [0.22, 0.38] |
| 1.0 | 1.0 | 0.26 [0.01, 0.44] | 0.34 [0.28, 0.40] |

Los atacantes necesarios para vencer a quienes solo crecen son cerca de 5% en todas las celdas,
pero no están determinados (el rango llega de 0 a unos 0.3): la ventaja de solo sobre atacar es de
unos 0.07 de retorno, dentro del ruido. La figura completa está en `docs/assets/masas_criticas.png`.

- Con λ y c más altos hace falta menos masa de cooperadores para que cooperar se imponga y la
  cooperación resiste mejor a los atacantes. Es la forma que predice H2, y se apoya en comparar
  esquinas de la rejilla, no celdas vecinas.
- Los rangos son anchos: a señal lenta la masa de cooperadores va de 0 a 1.
- No sustituye a H2 con agentes entrenados. Son estrategias fijas, parejas de dos y masas críticas
  de juegos de dos estrategias, que ignoran la tercera.

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

Pendiente, para quien tenga una máquina libre (5 seeds por variante, unos 15 min por corrida y
5 en paralelo saturan una laptop; sirve también el kernel de Kaggle): entrenar
`configs/h1_sanity.yaml` con `train.seed` de 0 a 4, `train.self_play.rho=0` y
`train.total_steps=5242880` (debe ser múltiplo de `n_envs * rollout_len`), una vez con el canal
y otra con `env.channel_enabled=false`. Evaluar cada `final.msgpack` con
`train.evaluar_checkpoint --baselines` y comparar retorno y supervivencia entre variantes y
contra ignorar. Si el canal importa con λ = 0 hay una fuga de información o un error del entorno.

### Comprobación con agentes entrenados (I-03)

Dos corridas, GPU (Kaggle, 2×T4), 5 seeds por variante, `configs/h1_sanity.yaml`,
`train.self_play.rho=0`. Fecha: 6 de noviembre de 2026.

**Primer intento, `train.total_steps=5242880` (320 actualizaciones):**

| Variante | retorno medio | supervivencia media | muertes por ataque | acción atacar |
|---|---|---|---|---|
| con canal | +0.79 | 0.18 | 0.82 | 0.056 |
| sin canal | +0.43 | 0.14 | 0.86 | 0.081 |

Con solo 320 actualizaciones la política no podó el impulso de atacar (4%-12% de las acciones),
así que ambas copias de la misma red (comparten parámetros) se eliminaban entre sí antes de que
el retorno de crecer se acumulara. Inconcluso: se repitió con el doble de pasos.

**Repetición, `train.total_steps=10485760` (640 actualizaciones, el mismo presupuesto que A-09):**

| Variante | retorno medio | retorno por seed | supervivencia media |
|---|---|---|---|
| con canal | +1.87 | 1.854, 1.872, 1.981, 1.825, 1.834 | 0.355 |
| sin canal | +2.05 | 2.299, 2.303, 1.549, 2.219, 1.863 | 0.369 |

Baselines fijos (iguales en todas las variantes, como corresponde con λ = 0): ignorar +9.065,
solo +8.866, coop_real +8.318, oráculo +8.953, supervivencia 1.000 en las tres.

La diferencia entre variantes se invirtió de signo respecto al primer intento (ahí ganaba con
canal por +0.36, aquí gana sin canal por +0.18, más chico que la dispersión entre seeds de
`sin_canal` solo) y la diferencia de +1.87 contra +2.05 queda dentro del error esperado entre
seeds. Es la firma de ruido, no de un efecto sistemático del canal: **no hay evidencia de fuga
de información cuando λ = 0**, lo que es consistente con H1.

Dicho eso, ninguna de las dos variantes se acerca a ignorar ni con 640 actualizaciones:
supervivencia 35%-37% y 60%-69% de las muertes siguen siendo por ataques entre las dos copias
de la misma política, que todavía dedica 9%-28% de las acciones a ocultarse y 1.8%-3.7% a
atacar en vez de crecer. Repite el mismo patrón que A-09 (la red con 10.5M pasos y radio 0.45
tampoco superó a ignorar): con los presupuestos de cómputo probados hasta ahora, IPPO no
converge a la estrategia trivialmente óptima de solo crecer, ni siquiera cuando no hay ninguna
anomalía que evitar.

**Conclusión para H1:** evidencia parcial a favor. Descarta una fuga de información grande y
obvia por el canal, pero no es una confirmación limpia porque ninguna red juega cerca de bien
todavía — el cuello de botella sigue siendo el mismo que en A-09: el entrenamiento no converge
a "solo crecer" en los presupuestos de cómputo probados.

## Costo de ignorar (A09)

Con el radio 0.25 de la zona, una civilización que nunca se refugia sobrevive el 79% de los
episodios: la letalidad casi no importa porque la posición es fija y, con unos nueve estallidos
por episodio, cualquier civilización dentro de la zona muere con λ ≥ 0.5. Lo que decide cuánto
cuesta ignorar es el área de la zona, no λ. Sin tocar la recompensa:

| radio | crecer (ignorar) | solo | coop_real | oráculo | aporte de coop_real sobre ignorar |
|---|---|---|---|---|---|
| 0.25 | +7.03 | +6.82 | +7.81 | +8.95 | +0.78 |
| 0.40 | +4.02 | +3.86 | +6.97 | +8.79 | +2.95 |
| 0.55 | +0.45 | +0.40 | +6.18 | +8.59 | +5.73 |

Dos civilizaciones, λ = 1, c = 1, 3 seeds, 200 episodios. Con el radio base cooperar mejora el
retorno cerca de 11%, que coincide con lo observado en las corridas de la red. Las rejillas 3×3
completas con radios 0.25, 0.35 y 0.45 están en `runs/pagos_r*`. `solo` rinde igual que ignorar
en todos los radios: medir sin compartir no alcanza para anticipar el estallido.

Corrida corta con `configs/costo_ignorar.yaml` (radio 0.45, 2.1 M pasos, CPU): retorno +2.22 y
supervivencia 0.33, crece el 95% del tiempo. Las políticas fijas obtienen ignorar +2.67 (0.34),
solo +3.53, coop_real +6.88 y oráculo +8.71. Con 2 M pasos la red no supera a ignorar. Esta comparación exigió corregir `baseline_returns`, que no pasaba el radio y
medía las políticas fijas en el mundo de radio 0.25.

Cinco seeds de 10.5 M pasos con la misma config (Kaggle, CPU, `kaggle/run.py`; corridas en
`runs/costo_ignorar-f8b2146ed7/`), evaluadas con 400 episodios:

| seed | retorno | supervivencia | crecer | refugiarse | atacar |
|---|---|---|---|---|---|
| 0 | +1.88 | 0.36 | 0.72 | 0.23 | 0.001 |
| 1 | +1.46 | 0.26 | 0.82 | 0.13 | 0.005 |
| 2 | +1.90 | 0.34 | 0.78 | 0.15 | 0.002 |
| 3 | +2.30 | 0.35 | 0.91 | 0.06 | 0.001 |
| 4 | +1.93 | 0.34 | 0.78 | 0.17 | 0.001 |
| media ± error estándar | +1.89 ± 0.13 | 0.33 | | | |

Ninguna seed alcanza a ignorar (+2.77, supervivencia 0.36), solo (+3.71) ni coop_real (+6.94).
Las cinco aprenden a refugiarse (6% a 23% del tiempo) sin saber cuándo, y transmitir o compartir
quedan por debajo del 2%. Con radio 0.45 y 10.5 M pasos H1 y H2 no tienen evidencia a favor ni en
contra: la red no usa el canal. Falta saber si más pasos o una GPU cambian el patrón.

## Ablación del canal con políticas fijas

`train.ablacion_canal` mide ignorar, solo y coop_real con `env.channel_enabled` encendido y
apagado. Dos civilizaciones, periodo aleatorio, c = 1, margen 0.2 para coop_real, 5 seeds y
200 episodios. La diferencia se calcula por seed (mismas semillas con y sin canal) y el ± es el
error estándar. "Aporte" es coop_real con canal menos la mejor de ignorar y solo.

| λ | radio | coop_real con canal | sin canal | diferencia por apagar | aporte del canal |
|---|---|---|---|---|---|
| 0 | 0.25 | +8.383 | +8.337 | +0.046 ± 0.007 | -0.682 ± 0.005 |
| 0 | 0.45 | +8.215 | +8.258 | -0.043 ± 0.004 | -0.850 ± 0.004 |
| 0.25 | 0.25 | +8.061 | +7.315 | +0.746 ± 0.106 | +0.378 ± 0.113 |
| 0.25 | 0.45 | +7.678 | +5.240 | +2.438 ± 0.115 | +2.495 ± 0.150 |
| 0.5 | 0.25 | +7.917 | +6.840 | +1.076 ± 0.112 | +0.677 ± 0.128 |
| 0.5 | 0.45 | +7.254 | +3.643 | +3.611 ± 0.143 | +3.543 ± 0.153 |
| 1 | 0.25 | +7.778 | +6.508 | +1.270 ± 0.098 | +0.709 ± 0.106 |
| 1 | 0.45 | +6.681 | +2.503 | +4.178 ± 0.139 | +3.900 ± 0.151 |

- **No hay fuga de información.** Ignorar y solo dan retornos idénticos con y sin canal en las
  ocho celdas, seed por seed. Lo que difiere es solo lo que usa el canal.
- **H1 con políticas fijas.** Con λ = 0 apagar el canal cambia coop_real en menos de 0.05, con
  signo distinto según el radio, es decir cerca de 0.5% del retorno y no sistemático. No es
  exactamente cero: coop_real se refugia distinto cuando recibe una medición, aunque con λ = 0
  refugiarse no sirve de nada. Lo que sí se cumple es que el canal no vale nada: ignorar rinde
  entre 0.68 y 0.85 más que coop_real. Con agentes entrenados falta la misma ablación.
- **Con λ > 0 el canal sí importa y su aporte crece con λ y con el radio.** En radio 0.25 sube
  de +0.38 a +0.71, y en radio 0.45 de +2.50 a +3.90. Sin canal coop_real queda cerca de solo
  o por debajo (paga el costo de compartir y no recibe nada). Es la forma que predice H2 para
  estas estrategias, sin ser evidencia de aprendizaje.
- **Lectura para el entrenamiento.** El valor del canal que una red podría descubrir es +0.7
  con el radio base y +3.9 con 0.45 en λ = 1. Las cinco redes entrenadas con radio 0.45
  (+1.89) quedaron por debajo de ignorar, así que el techo existe pero no lo encuentran.
- **Mensajes barajados** (`env.shuffle_messages`, implementado en esta rama). El mensaje conserva
  emisor, símbolo, llegada y ángulo de observación, pero su medición se cambia por la de un θ
  independiente, como si viniera de otro episodio. Isola si importa que el mensaje sea verdadero
  o solo que llegue. Retorno de coop_real:

| λ | radio | con canal | sin canal | barajado | quita barajar |
|---|---|---|---|---|---|
| 0 | 0.25 | +8.383 | +8.337 | +8.430 | -0.047 ± 0.009 |
| 0 | 0.45 | +8.215 | +8.258 | +8.267 | -0.052 ± 0.003 |
| 0.25 | 0.25 | +8.061 | +7.315 | +7.310 | +0.751 ± 0.093 |
| 0.25 | 0.45 | +7.678 | +5.240 | +5.860 | +1.818 ± 0.127 |
| 0.5 | 0.25 | +7.917 | +6.840 | +6.880 | +1.037 ± 0.097 |
| 0.5 | 0.45 | +7.254 | +3.643 | +4.110 | +3.144 ± 0.155 |
| 1 | 0.25 | +7.778 | +6.508 | +6.588 | +1.190 ± 0.104 |
| 1 | 0.45 | +6.681 | +2.503 | +2.607 | +4.074 ± 0.149 |

  Barajar quita casi todo lo que quita apagar el canal: lo que vale es que el mensaje sea
  verdadero, no que llegue. En dos celdas (λ de 0.25 y 0.5 con radio 0.45) el barajado recupera
  entre 0.5 y 0.6 sobre apagar; es probable que refugiarse en un lugar al azar proteja algo
  cuando la zona cubre más de la mitad del toro, pero no lo verifiqué. Con λ = 0 los tres
  casos quedan a menos de 0.06 entre sí. Ignorar y solo siguen idénticos en las tres variantes.
- **Límites.** Son políticas fijas; la misma ablación con redes entrenadas queda pendiente y
  hoy ninguna red usa el canal. Las señales costosas siguen sin implementar, así que H3
  queda pendiente.

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
| `env.detection_transmit` | Probabilidad de detección en el paso en que se transmite o comparte | Sin valor (igual a `detection_high`), el comportamiento anterior. Con un valor bajo compartir deja de exponer |
| `env.loot` | Fracción de los recursos de la víctima que gana quien la elimina con un ataque | 0: sin botín, el comportamiento anterior. La recompensa no cambia, solo crecen los recursos del atacante |

Bajar `detection_growth` o `detection_transmit` cambia la regla de la sección 3.3 de la
propuesta (crecer, transmitir y compartir emiten en alta) y debe declararse. Los valores de
barrido propuestos están en la sección de sensibilidad (A-08). Con 200 pasos casi todo emisor es descubierto
si emite a 0.05 por la distancia (cerca de 99.9%), por eso la exposición de cooperar solo pesa
con una detección base muy baja.

## Sensibilidad a la detección de quien crece (A-08)

Se midieron matrices de pagos con radio 0.45, λ en {0.5, 1}, c en {0.22, 1}, 3 seeds por celda,
para `detection_growth` en {0.1, 0.05, 0.01, 0.001} (en 0.01 y 0.001 se bajó también
`detection_low` al mismo valor, porque el crecimiento no puede quedar fuera de [low, high]). La
referencia es `runs/pagos_r0.45` con la detección por defecto (crecer en alta, 0.9; 5 seeds).

| `detection_growth` | Dominante | Vértices estables | Cooperadores vs solos | Atacantes vs cooperadores |
|---|---|---|---|---|
| 0.9 (defecto) | coop_real | solo, coop_real (ataque_solo solo en c=0.22 con λ=1 y λ=0.25) | 0.04-0.13 | 0.97-0.99 |
| 0.1 | coop_real | solo, coop_real, ataque_solo | 0.04-0.06 | 0.78-0.82 |
| 0.05 | coop_real | solo, coop_real, ataque_solo | 0.04-0.06 | 0.70-0.79 |
| 0.01 | coop_real | solo, coop_real, ataque_solo | 0.04-0.06 | 0.60-0.70 |
| 0.001 | coop_real | solo, coop_real | 0.04-0.06 | 0.54-0.67 |

Lectura:

- La masa crítica de cooperadores frente a solos no depende de la detección (0.04-0.06): cooperar
  vale por evitar la anomalía, no por cómo se ve quien crece.
- La detección sí mueve al atacante: cuanto menos se ve quien crece, menos hay que atacar y la
  masa crítica de atacantes frente a cooperadores baja de 0.98 a 0.55. El vértice ataque_solo
  aparece como estable en valores intermedios y desaparece en 0.001.
- El régimen dominante no cambia en ningún valor, así que la conclusión principal (cooperar
  domina cuando la anomalía es costosa) no depende de este parámetro.

Propuesta para el barrido secundario (el equipo decide): `detection_growth` en {defecto, 0.05,
0.001}. Los extremos cubren con y sin ataque_solo estable; 0.1 y 0.01 quedan entre ellos y no
añaden regímenes nuevos. Lo medido son políticas fijas con 3 seeds; no se añade a
`configs/sweep.yaml` hasta que el equipo lo apruebe.

### Botín y detección al transmitir

Mismas celdas y 3 seeds, con `env.loot` en {0.5, 1.0} y `env.detection_transmit` en {0.3, 0.05}
(`detection_low` 0.05, así que 0.05 hace que compartir sea tan visible como callar).

| Ajuste | Dominante | Vértices estables | Cooperadores vs solos | Atacantes vs cooperadores |
|---|---|---|---|---|
| defecto | coop_real | solo, coop_real (más ataque_solo en λ=1, c=0.22) | 0.04-0.07 | 0.98-0.99 |
| `loot` 0.5 | coop_real | coop_real, ataque_solo (en λ=0.5, c=1 solo coop_real) | 0.04-0.06 | 0.98-0.99 |
| `loot` 1.0 | coop_real | coop_real, ataque_solo | 0.04-0.06 | 0.97-0.98 |
| `detection_transmit` 0.3 | coop_real | solo, coop_real | 0.06-0.10 | 0.97-1.00 |
| `detection_transmit` 0.05 | coop_real | solo, coop_real | 0.23-0.39 | 1.00 |

- El botín no mueve las masas críticas, pero cambia la estabilidad: el vértice solo deja de ser
  estable porque un atacante invade a los solos, y ataque_solo pasa a ser estable. Cooperar
  sigue dominando.
- La detección al transmitir solo pesa cuando compartir casi no emite: con 0.3 la masa crítica
  de cooperadores sube poco y con 0.05 pasa de 0.04-0.07 a 0.23-0.39. Con valores cercanos al
  defecto no hay efecto, porque a 200 pasos casi todo emisor es descubierto de todos modos.

Propuesta para el barrido secundario (el equipo decide): `loot` en {0, 0.5} y
`detection_transmit` en {defecto, 0.05}. El 1.0 repite el resultado del 0.5 y el 0.3 queda entre
el defecto y el 0.05 sin cambiar el régimen.

## Límites

- Son políticas fijas, no equilibrios aprendidos. Un entrenamiento puede encontrar
  estrategias condicionales o de tiempo que estas no ven.
- Solo dos civilizaciones, `p_kill` 0.75 y 1.0, y una configuración de detección.
- coop_real es una política hecha a mano (observa una vez, comparte hasta recibir o hasta
  agotar 6 pasos); otra mejor podría acercarse más a coop_ideal.
- El atacante único dispara al primer contacto; otras políticas de ataque pueden dar otros
  pagos.
- Sin `env.loot` no hay botín; tampoco se modeló defensa contra proyectiles.

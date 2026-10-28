# Dark Forest: necesitar a los otros, temer a los otros

Proyecto final de CC3092 (Deep Learning y Sistemas Inteligentes). Aprendizaje por refuerzo
multiagente sobre un juego estocástico parcialmente observable.

Presentación: semana del **9 al 13 de noviembre de 2026**.

> Nota de edición: transcripción de `rl-proyecto.pdf` (fuente original no incluida en el
> repositorio). Se corrigieron
> duplicaciones de fórmulas y defectos de impresión, sin cambiar el contenido científico.
> La fecha de presentación fue actualizada por el equipo. El cronograma original de la
> sección 7 se conserva como referencia histórica; el calendario operativo vigente está en
> [TAREAS.md](TAREAS.md), con cierre de materiales el **8 de noviembre de 2026**.

## 1. Origen de la idea

La propuesta nace del escenario C de **Rogue Agents**:

Un universo simulado donde varios agentes reciben un desafío que ninguno puede resolver
solo. Al inicio, cada agente desconoce que existen los demás y su percepción del universo es
parcial y estrictamente individual. Explorando, pueden descubrir indicios de otras
inteligencias. Los agentes desarrollan objetivos incompatibles, compiten por recursos
virtuales y pueden eliminar a otros participantes dentro del juego. Se busca observar si
emergen coaliciones, conflicto, negociación, equilibrio estratégico o colapso colectivo.

A eso se suma la hipótesis del bosque oscuro de Liu Cixin: en un universo donde la intención
de otra civilización no se puede verificar, esconderse o atacar primero puede ser lo racional.

## 2. El problema

En el bosque oscuro clásico, el contacto no trae ningún beneficio, solo riesgo. Si el entorno
se construye así, esconderse o atacar domina de forma trivial: el resultado ya está en las
reglas y no emerge nada.

La idea original aporta el contrapeso: **un desafío cósmico que solo se resuelve combinando
observaciones de varias civilizaciones.** Comunicarse vale algo, pero comunicarse revela la
posición. Ese es el dilema central:

> Necesito a los otros para sobrevivir, pero los otros pueden destruirme si me encuentran.

**Pregunta de investigación.** ¿Cuánto debe valer la cooperación para vencer al miedo, y cómo
cambia ese umbral con el retraso de las señales, la letalidad de los ataques y la posibilidad
de enviar señales costosas?

**Resultado principal esperado.** Un diagrama de fases que muestre en qué condiciones emerge
cada régimen: cooperación, silencio mutuo, ataque preventivo, coaliciones o colapso.

## 3. Especificación del entorno

### 3.1 Formalismo

Juego estocástico parcialmente observable (POSG):

$$
\langle N, S, \{A_i\}, \{O_i\}, T, \{R_i\}, \gamma \rangle
$$

- $N = \{1, \ldots, N\}$: civilizaciones, con $N \in \{2, 4, 6, 8\}$.
- Cada civilización ve solo su observación $o_i$. No conoce $N$ ni las posiciones de las demás.
- Los agentes comparten parámetros de la red y se distinguen por un identificador.

### 3.2 Estado $s_t$

| Componente | Descripción |
|---|---|
| $p_i \in [0,1]^2$ | Posición fija de cada civilización en un toro 2D |
| $r_i \geq 0$ | Recursos |
| $v_i \in \{0,1\}$ | Viva o eliminada |
| $e_i \in \{\text{baja}, \text{alta}\}$ | Nivel de emisión (cuánto "brilla" ante las demás) |
| $\theta$ | Parámetros ocultos de la anomalía (fase, periodo, centro del estallido) |
| Cola de señales | `(origen, t_emisión, contenido)`; cada señal se expande como un anillo de radio $c(t - t_{\text{emisión}})$ |
| Cola de proyectiles | `(origen, destino, t_lanzamiento)`; viajan a velocidad $v_p \leq c$ |

### 3.3 Acciones $A_i$ (discretas)

| Acción | Efecto | Costo |
|---|---|---|
| `crecer` | $r_i \mathrel{+}= g \cdot r_i$ | Pone la emisión en alta |
| `ocultarse` | Pone la emisión en baja | Sin crecimiento ese turno |
| `observar` | Recibe una medición $z_i = h_i(\theta) + \varepsilon$ | $c_{\text{obs}}$ |
| `transmitir(k)` | Emite el símbolo $k \in \{1, \ldots, K\}$ a todas direcciones y revela la posición | $c_{\text{tx}}$ |
| `compartir` | Emite la última medición $z_i$ y revela la posición | $c_{\text{tx}}$ |
| `atacar(j)` | Lanza un proyectil a la posición detectada en la ranura $j$ | $C_{\text{atk}}$ |
| `refugiarse` | Protege del próximo estallido | $c_{\text{ref}}$ |

Extensión J-04 (Los Extraños): acción `interferir`, que degrada las mediciones de la anomalía
dentro de un radio.

### 3.4 Detección y observación $o_i$

Una civilización $i$ detecta a $j$ cuando el anillo de emisión de $j$ la alcanza, con
probabilidad $p_{\text{det}}(e_j, d_{ij})$. La probabilidad es mayor con emisión alta y menor
con la distancia. Descubrir a otro es, literalmente, que su cono de luz te alcance.

$o_i$ contiene:

- recursos y emisión propios;
- hasta $M$ ranuras de detección, con rumbo y distancia estimada del emisor, contenido
  recibido y antigüedad (con relleno si hay menos);
- la última medición propia de la anomalía y las mediciones recibidas de otros;
- los turnos desde el último estallido y si sobrevivió.

### 3.5 La anomalía: el desafío compartido

Un fenómeno periódico con parámetros ocultos $\theta$. Cada civilización mide una proyección
ruidosa $h_i(\theta)$ que depende de su ángulo de observación. **$\theta$ solo es identificable
combinando al menos dos proyecciones distintas.**

En cada estallido, toda civilización dentro de la zona afectada que no esté refugiada muere
con probabilidad $\lambda$. Refugiarse siempre cuesta, así que hacerlo en todos los turnos no
es viable: hay que predecir cuándo y dónde.

### 3.6 Recompensa

$$
R_{i,t} = \log(1 + r_{i,t+1}) - \log(1 + r_{i,t})
$$

La eliminación termina el episodio para esa civilización, que deja de acumular recompensa.
**No hay recompensa por ocultarse, atacar ni cooperar.** Cualquier estrategia que aparezca
es instrumental (criterio de no imposición).

### 3.7 Parámetros del barrido

| Parámetro | Símbolo | Rango inicial | Rol |
|---|---|---|---|
| Letalidad de la anomalía | $\lambda$ | 0, 0.25, 0.5, 0.75, 1.0 | Valor de la cooperación ($\lambda = 0$ es el bosque oscuro clásico) |
| Velocidad de las señales | $c$ | 5 valores en escala logarítmica | Retraso de la información |
| Probabilidad de que un ataque elimine | $p_{\text{kill}}$ | 0.5 - 1.0 | Letalidad del conflicto |
| Costo de atacar | $C_{\text{atk}}$ | bajo / alto | Barrera al ataque preventivo |
| Señales costosas | — | sí / no | Verificabilidad de la intención (H3) |
| Número de civilizaciones | $N$ | 2, 4, 6, 8 | Escala y coaliciones (H4) |

## 4. Hipótesis falsables

| ID | Hipótesis | Cómo se prueba |
|---|---|---|
| **H1** (control) | Con $\lambda = 0$ la cooperación no tiene valor: anular el canal no cambia el retorno ni la supervivencia. | Ablación del canal con $\lambda = 0$ y comparación contra las políticas que no lo usan. Si el canal importa con $\lambda = 0$, hay una fuga de información o un error del entorno. Es la prueba de cordura. |
| **H2** (umbral) | Existe un $\lambda^*$ a partir del cual emerge la comunicación, y $\lambda^*$ crece con el retraso y con $p_{\text{kill}}$. | Barrido $\lambda \times c$; la frontera de fase es el resultado principal. |
| **H3** (señales costosas) | Permitir señales caras de falsificar reduce $\lambda^*$. | Mismo barrido con y sin señales costosas. |
| **H4** (coaliciones) | Con $N \geq 6$ y detección local, emergen grupos que comparten información entre sí y guardan silencio ante los demás. | Grafo de quién responde a quién, más detección de comunidades. |
| **H5** (biestabilidad) | Cerca de $\lambda^*$, unas seeds convergen a cooperación y otras a silencio. | Distribución de resultados por seed en la frontera. |

**Nota sobre H1 (reformulada el 27 de octubre de 2026).** La redacción original decía: "Con
$\lambda = 0$, el silencio o el ataque preventivo dominan para cualquier retraso". Se cambió
después de medir políticas fijas en el entorno implementado (`docs/DIAGNOSTICO_DISENO.md`): con
$\lambda = 0$ crecer fue la mejor respuesta a todas las estrategias medidas, porque con
recompensa logarítmica y crecimiento exponencial renunciar a crecer cuesta mucho más que el
riesgo de ser atacado. H1 era una predicción interna de diseño, no un resultado. El cambio es
posterior a ver datos y se reporta como tal. La pregunta original se conserva como pregunta
exploratoria: si en alguna región de parámetros el silencio o el ataque preventivo dominan con
$\lambda = 0$. No se usa como prueba de cordura.

## 5. Cómo distinguir emergencia genuina

| Criterio | Prueba |
|---|---|
| No imposición | Auditar la recompensa: solo crecimiento y supervivencia. |
| Necesidad funcional | Anular el canal (`transmitir` y `compartir` sin efecto) y barajar los mensajes; medir la caída de supervivencia. |
| Modelo nulo | Comparar con agentes aleatorios y con los cuatro arquetipos heurísticos. |
| Robustez | 5 seeds o más por configuración. |
| Dependencia de condiciones | El barrido de parámetros: el régimen cambia en una frontera identificable. |

### Métricas

- Tasa de supervivencia, total y por causa: anomalía o ataque.
- Fracción del tiempo en emisión baja.
- Transmisiones y observaciones compartidas por episodio.
- Ataques preventivos: ataques contra una civilización que nunca ha atacado.
- Precisión de predicción de la anomalía: refugios correctos sobre el total de estallidos.
- $I(M; \theta)$: cuánto informan los mensajes sobre los parámetros de la anomalía.
- Régimen del episodio: cooperación, silencio mutuo, ataque preventivo, coalición o colapso.

## 6. Implementación

### 6.1 Algoritmo

- **IPPO / MAPPO** con parámetros compartidos e identificador de agente.
- Red con **GRU**, por la observación parcial y el retraso de las señales.
- **Autojuego ficticio:** se entrena contra una población de versiones pasadas de los propios
  agentes, para no quedar atrapados en un solo equilibrio.
- **Línea base analítica:** dinámica del replicador sobre cuatro arquetipos (siempre oculto,
  siempre transmite, recíproco, atacante preventivo), con la matriz de pagos medida en el
  mismo simulador.

### 6.2 Pseudocódigo del ciclo de entrenamiento

```text
para cada configuración (λ, c, p_kill, ...) en el barrido:
  para cada seed en 1..S:
    inicializar política π_θ (compartida) y población P ← {π_θ}
    repetir hasta presupuesto de pasos:
      para cada entorno en paralelo (vmap):
        asignar a cada civilización una política: π_θ, o una muestra de P con prob. ρ
        simular T pasos → trayectorias (o, a, r, h_GRU)
      calcular ventajas con GAE solo sobre las civilizaciones controladas por π_θ
      actualizar π_θ con PPO (clip, entropía, valor)
      cada K iteraciones: P ← P ∪ {copia congelada de π_θ}
    evaluar π_θ contra sí misma y contra los arquetipos → métricas, régimen
  agregar métricas por configuración → celda del diagrama de fases
```

### 6.3 Stack

- **JAX** para vectorizar miles de partidas en paralelo; el entorno sigue el estilo de
  JaxMARL.
- **Flax/Optax** para las redes; **W&B** o TensorBoard para el registro.
- Visualización con matplotlib para el análisis y una animación de la galaxia para la demo.
- **Cómputo:** se espera que quepa en Colab con GPU; se confirma midiendo el tiempo de
  entrenamiento en la semana 1.

### 6.4 Esqueleto del repositorio

```text
dark-forest/
├── PROPUESTA.md         ← este documento
├── configs/             ← un YAML por configuración del barrido
├── darkforest/
│   ├── env.py           ← POSG en JAX: estado, transición, detección, anomalía
│   ├── anomaly.py       ← θ, proyecciones h_i(θ), estallidos
│   ├── archetypes.py    ← agentes heurísticos y modelo nulo
│   └── networks.py      ← actor-crítico con GRU
├── train/
│   ├── ippo.py          ← ciclo de entrenamiento y autojuego ficticio
│   └── sweep.py         ← lanza el barrido de parámetros
├── analysis/
│   ├── metrics.py       ← métricas y clasificación de régimen
│   ├── replicator.py    ← línea base de dinámica del replicador
│   └── phase_diagram.py ← diagrama de fases
├── viz/
│   └── galaxy.py        ← animación: conos de luz, proyectiles, estallidos
└── notebooks/
```

### 6.5 Demo

Galaxia con conos de luz expandiéndose, proyectiles en tránsito y estallidos de la anomalía.
Al lado, el diagrama de fases $\lambda \times c$ llenándose celda por celda.

## 7. Plan de trabajo

> Cronograma original del PDF, conservado como antecedente. **No es el calendario vigente.**
> Las fechas actualizadas, responsables y criterios de salida están en [TAREAS.md](TAREAS.md).

| Semana original | Entregable |
|---|---|
| 23 - 29 oct | Entorno en JAX, visualización, arquetipos y prueba de cordura (H1) con 2 a 4 civilizaciones. Medir tiempo por paso. |
| 30 oct - 5 nov | IPPO entrenando, anomalía activa y primer barrido grueso 3×3 con 3 seeds. |
| 6 - 12 nov | Barrido completo 5×5 con 5 seeds, ablaciones de canal, señales costosas (H3) y línea base del replicador. |
| 13 - 19 nov | Coaliciones con 6 a 8 civilizaciones (H4) si hay tiempo, video de la demo, informe y presentación. |

## 8. Riesgos y mitigaciones

| Riesgo | Señal temprana | Mitigación |
|---|---|---|
| Las civilizaciones nunca se descubren y todo colapsa en silencio | Cero detecciones por episodio | Currículo: galaxia pequeña y detección fácil al inicio, luego alejarlas |
| Convergencia a un solo equilibrio | Todas las seeds idénticas | Autojuego ficticio y entropía más alta |
| El canal se ignora | La ablación no cambia la supervivencia | Verificar que $\theta$ no sea identificable con una sola proyección |
| El barrido no cabe en el cómputo | Más de 10 min por configuración | Reducir a un barrido 3×3 y priorizar H1–H2 |

## 9. Extensiones (módulos del paquete)

- **J-04 Los Extraños:** acción de interferencia para bloquear la observación de la anomalía
  a otros. Pregunta: ¿cuándo es racional ocultar conocimiento?
- **J-02 Rogue Civilizations:** escala a 6–8 civilizaciones con reputación. Es el escenario
  natural para H4.
- **A-01 The Painting That Thinks:** capa de visualización donde el comportamiento colectivo
  se registra como pintura.

## 10. Trabajos relacionados

- Liu Cixin, *El bosque oscuro* (2008): la hipótesis original.
- Leibo et al. (2017), *Multi-agent Reinforcement Learning in Sequential Social Dilemmas*.
- Leibo et al. (2021), *Melting Pot*: evaluación de la generalización social.
- Baker et al. (2019), *Emergent Tool Use from Multi-Agent Autocurricula*.
- Lanctot et al. (2017), PSRO: aprendizaje de equilibrios con poblaciones.
- Zahavi (1975) y Grafen (1990): el principio del handicap y las señales costosas.
- Lowe et al. (2019), *On the Pitfalls of Measuring Emergent Communication*.
- Rutherford et al. (2023), JaxMARL.

Verificar títulos, años y venues en la fuente original antes de citarlos en el informe.

## 11. Alcance ético

Proyecto exclusivamente académico y computacional. Todo ocurre dentro de una simulación
cerrada, sin conexión con sistemas externos.

Tarjeta de una métrica con su etiqueta, su valor y una nota. Se usa para el retorno medio, la supervivencia, el número de mensajes o de mentiras. Es `ob-metric` con `ob-metric__label`, `ob-metric__value` (mono, 28 px) y `ob-metric__note`.

**El consumidor aporta** la etiqueta corta, el valor ya formateado («+5.97») y una nota que dice de qué es el valor («promedio de 40 episodios»). Marca `ob-metric--hl` en una sola tarjeta por pantalla: la que responde a la pregunta del momento.

**Estados.** Normal (borde `line`) y destacada (borde `sun`, valor `sun` y `shadow-glow`). No tiene estado interactivo.

**Accesibilidad.** Agrupa etiqueta y valor en el mismo bloque para que el lector los lea juntos («Retorno, más cinco punto noventa y siete»). Los signos van en el texto, no en un icono.

| Haz | No hagas |
|---|---|
| Di siempre la base del valor en la nota. | Mostrar una cifra sin decir si es de un episodio o un promedio. |
| Destaca una sola tarjeta. | Destacar todas las tarjetas. |

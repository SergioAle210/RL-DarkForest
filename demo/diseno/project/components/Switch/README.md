Interruptor de una opción binaria del mundo: «Engaño», «Señales costosas», «Red determinista». Es una casilla `<input type="checkbox" role="switch">` dentro de `ob-switch`, seguida de la pista `ob-switch__track` y el texto.

**El consumidor aporta** el texto, siempre una afirmación que describe el estado encendido («Engaño (mentir)»), y el valor inicial.

**Estados.** Apagado (pista `raised` con borde `line-strong` y pulgar `muted`); encendido (pista `sun` y pulgar `on-sun`); `:focus-visible` con anillo `focus` alrededor de la pista; deshabilitado con opacidad reducida.

**Accesibilidad.** La casilla real recibe el foco y se activa con Espacio. El estado se comunica por posición del pulgar y por relleno, nunca solo por color; el lector anuncia «activado/desactivado».

| Haz | No hagas |
|---|---|
| Un texto que se lea bien encendido: «Engaño (mentir)». | Un texto negativo («Sin engaño») que se invierte al encender. |
| Aplica el cambio al pulsar Simular, no al instante. | Re-simular en cada clic del interruptor. |

Control deslizante para un parámetro continuo del mundo: letalidad λ, ruido de la medición, costo del refugio, probabilidad de que un ataque mate. Es un `<input type="range">` con la clase `ob-range`, dentro de un `ob-field` cuya cabecera muestra la etiqueta a la izquierda y el valor en `ob-field__value` (mono, `sun`).

**El consumidor aporta** `min`, `max`, `step`, el valor y la etiqueta. Actualiza el valor mostrado en cada cambio y añade `aria-valuetext` si la unidad no es obvia («0.50»).

**Estados.** Reposo (pista `line-strong`, pulgar `sun`); `:focus-visible` con anillo `focus`; deshabilitado con opacidad reducida.

**Accesibilidad.** Flechas, Inicio y Fin cambian el valor. El valor numérico siempre está a la vista; no lo dejes solo en la posición del pulgar.

| Haz | No hagas |
|---|---|
| Muestra el valor numérico a la derecha de la etiqueta. | Depender solo de la posición del pulgar. |
| Un `step` que dé valores redondos (0.05). | Un `step` de 0.001 que obliga a arrastrar con precisión de píxel. |

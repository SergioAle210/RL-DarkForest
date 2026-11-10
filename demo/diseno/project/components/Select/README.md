Selector de una opción de una lista, para elegir la política de cada civilización, el número de civilizaciones o un escenario. Envuelve un `<select>` nativo en `ob-select` para dibujar el caret.

**El consumidor aporta** una etiqueta visible (`<label for>`), las opciones y, si son muchas, `<optgroup>` con nombre: «Estrategias» y «Redes entrenadas».

**Estados.** Reposo con borde `line-strong`; `:focus-visible` con anillo `focus`; deshabilitado con opacidad reducida. El valor mostrado va en `ink` sobre `raised`.

**Accesibilidad.** Es nativo: se abre con Espacio o Alt+Flecha y se recorre con las flechas. La etiqueta siempre es visible; no uses un texto interno como etiqueta.

| Haz | No hagas |
|---|---|
| Agrupa más de siete opciones con `optgroup`. | Mezclar estrategias y redes en una lista plana. |
| Escribe cada opción en lenguaje del proyecto: «Coopera: comparte y usa lo que recibe». | Mostrar identificadores internos como `fijo:coop_real`. |

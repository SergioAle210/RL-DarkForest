Línea de tiempo de un episodio: un deslizador sobre los turnos, marcas de los estallidos debajo y una lectura del turno actual. Combina un `ob-range` dentro de `ob-timeline__track`, `ob-timeline__marks` con una marca `<b>` por estallido (posición en porcentaje, color `--c: var(--burst)`) y `ob-timeline__read`.

**El consumidor aporta** el número total de turnos, el turno actual y la lista de turnos con estallido. La marca de cada estallido se coloca con `left: (turno / total) * 100%`.

**Estados.** Reposo; `:focus-visible` con anillo `focus` en el deslizador; en reproducción, el pulgar avanza solo. Con `prefers-reduced-motion`, la reproducción empieza en pausa.

**Accesibilidad.** El deslizador lleva `aria-label="Turno"` y `aria-valuetext="turno 38 de 200"`. Las marcas son decorativas; dilas también en texto: «estallidos en los turnos 12, 31, 57».

| Haz | No hagas |
|---|---|
| Lee el turno como «38/200» en mono. | Mostrar solo el porcentaje. |
| Marca los estallidos con la palabra en la leyenda. | Marcar con un color sin leyenda. |

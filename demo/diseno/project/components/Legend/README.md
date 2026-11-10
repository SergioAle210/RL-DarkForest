Leyenda de las marcas de la arena: compartir, símbolo, mentira, estallido y refugio. Es `ob-legend` con un `ob-legend__item` por marca; la muestra `<i>` toma su color de `--c` y es una línea de 2 px, o un punto con `data-shape="dot"`.

**El consumidor aporta** la lista de marcas que de verdad pueden aparecer en la escena actual; omite «mentira» si el engaño está apagado. Cada muestra lleva su palabra al lado.

**Estados.** Estática. Sin interacción.

**Accesibilidad.** La palabra y la muestra juntas son la señal: el color no se usa solo. Las muestras son decorativas (`aria-hidden="true"`) y el texto las describe.

| Haz | No hagas |
|---|---|
| Colócala junto a la arena, siempre visible. | Esconderla tras un botón. |
| Muestra solo las marcas posibles en esa escena. | Listar «mentira» cuando el engaño está apagado. |

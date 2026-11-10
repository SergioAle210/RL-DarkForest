Medidor de la escala de Kardashev. Es una barra de 0 a III con marcas en I, II y III que muestra cuánto ha crecido una civilización, junto al nombre de su etapa y el valor numérico.

**La escala es ilustrativa.** Los recursos del entorno son adimensionales: no son vatios ni energía. El nivel se calcula como `3 · ln(max(recursos, 1)) / ln(12)`, acotado entre 0 y 3 (`OB.kardashev.nivel`), de modo que 12 recursos equivalen al nivel III. Es una metáfora para leer el crecimiento relativo, no una medida física. Rotula siempre la escala como «ilustrativa» cerca del medidor.

**El consumidor aporta** el nivel (`OB.kardashev.nivel(recursos)`), la etapa (`OB.kardashev.etapa(nivel)`, con `id` «0», «I», «II» o «III», nombre y descripción) y el color de la civilización en `--c` (por defecto `share`; usa `civ-*` si el medidor va junto a una civilización concreta). En el interior de la demo, `OB.kardashev.medidorSvg(nivel, ancho)` produce una versión SVG equivalente.

| Clase | Contenido |
|---|---|
| `ob-kard` | Contenedor. Recibe `--nivel` (0 a 3) y `--c`. Lleva el rol `meter`. |
| `ob-kard__cabecera` | Civilización, etapa y nombre a la izquierda; el valor en `data` a la derecha. |
| `ob-kard__pista` | Pista con borde `line-strong`. Contiene el relleno y las marcas. |
| `ob-kard__relleno` | Barra de ancho `nivel / 3`. |
| `ob-kard__marca` | Marca vertical en I, II y III (`--m` = 1, 2 o 3), de 2 px en `ink`. |
| `ob-kard__escala` y `ob-kard__etiqueta` | Rótulos 0, I, II y III bajo la barra; los alcanzados van en `ink` y en negrita. |

| Etapa | Nivel | Nombre |
|---|---|---|
| 0 | de 0 a menos de 1 | Planetaria incipiente |
| I | de 1 a menos de 2 | Planetaria |
| II | de 2 a menos de 3 | Estelar |
| III | 3 | Galáctica |

**Estados.** El medidor no tiene hover. Cambia con el turno: la barra crece con los recursos, y el nivel se mantiene en 3 al llegar al máximo. Una civilización eliminada conserva el último nivel que alcanzó, con el relleno en `dead` y la palabra «eliminada» en la cabecera.

**Accesibilidad.** Usa `role="meter"` con `aria-valuemin="0"`, `aria-valuemax="3"`, `aria-valuenow` y un `aria-valuetext` que dice etapa, nivel y «Escala ilustrativa»: «Nivel 1.67 de 3: etapa I, planetaria». Los rótulos 0, I, II y III son `aria-hidden` porque el texto de la cabecera ya da la etapa. No es interactivo y no recibe foco. Las marcas `ink` cumplen 3:1 sobre la pista y sobre el relleno en ambos temas, y la etapa se lee por palabra, no por color.

| Haz | No hagas |
|---|---|
| Escribe la etapa y su nombre junto al valor. | Dejar que el relleno de color sea la única señal. |
| Indica «escala ilustrativa» en la leyenda o la nota. | Presentar el nivel como energía o potencia real. |
| Usa `--c` con el color `civ-*` de la civilización medida. | Usar `sun` para el relleno: el acento es para lo accionable. |
| Muestra el valor con dos decimales en `data`. | Redondear el nivel a la etapa y perder el valor. |

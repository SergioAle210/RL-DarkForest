Observatorio del Bosque Oscuro es el lenguaje visual de la demo de RL-DarkForest, un laboratorio de civilizaciones que deciden entre esconderse, cooperar o atacar. La interfaz es un instrumento de observación: calma, precisa y sin adornos, para leer una escena y compararla con otra.

## Fundamentos del contenido

- Escribe en español, de tú, con frases cortas y en indicativo. Describe lo que se ve («Compartir verificado», «Estallido en el turno 38»), no lo que se promete.
- Nombra las cosas como el proyecto: *civilización*, *estallido*, *refugio*, *mensaje*, *mentira*. No uses «jugador», «enemigo» ni «aliado».
- Las cifras llevan su unidad o contexto: «+5.97 de retorno», «40 episodios», «turno 38/200». Usa punto decimal y dos decimales en retornos.
- Sé honesto con la incertidumbre. Junto a un episodio suelto escribe «Un episodio no es representativo» y ofrece el promedio. Un resultado negativo se muestra igual que uno positivo.
- Sin emojis y sin signos de exclamación. Los verbos de acción en los botones van en infinitivo: «Simular», «Reproducir», «Pausa».
- Mayúsculas de frase en títulos y botones; las etiquetas de sección y las cabeceras de tabla van en mayúsculas pequeñas con la clase `label`.

## Fundamentos visuales

- **Dos temas.** `noche` es el de referencia: vacío casi negro (`void`), paneles `surface` y controles `raised`, con texto `ink` de blanco cálido. `carta` es la versión de papel de carta estelar para imprimir o proyectar de día. Los componentes solo leen variables de `tokens.css`; nunca escribas un hex en un componente.
- **Un solo acento.** `sun`, un ámbar de sol lejano, marca lo accionable y lo activo: botón principal, valor de un deslizador, cursor de la línea de tiempo. Úsalo en una cosa por pantalla a la vez. Texto sobre un relleno `sun` va en `on-sun`.
- **El color significa algo en la arena.** `civ-1` a `civ-8` identifican civilizaciones; `share` es un compartir verdadero, `symbol` un símbolo, `lie` una mentira, `burst` un estallido, `refuge` un refugio, `dead` una civilización eliminada. Cada marca va con su palabra en la leyenda y en la tabla; el color nunca es la única señal.
- **Tipografía.** Los títulos van en `title-xl`, `title-lg` o `title-md` (Fraunces, un serif suave de aire de bitácora); el texto en `body` y `body-sm`; las etiquetas en `label`; todas las cifras en `data` o `data-lg` (IBM Plex Mono, cifras tabulares). Nunca mezcles una cifra en la familia de texto.
- **Bordes, no sombras.** Separa paneles con un borde de 1 px `line`. La sombra `shadow-panel` es solo para un panel flotante sobre el lienzo, y `shadow-glow` para el único elemento destacado de la pantalla.
- **Radios.** `radius-md` en controles, `radius-lg` en paneles y tarjetas, `radius-pill` en el interruptor y la pista del deslizador. Los discos de las civilizaciones son círculos exactos.
- **Espaciado.** Escala de 4 px: `space-2` entre etiqueta y control, `space-3` entre controles, `space-4` de relleno de panel, `space-5` entre secciones.
- **Foco.** Todo elemento interactivo muestra un anillo sólido de 2 px `focus` con separación de 2 px. Nunca lo quites.
- **Movimiento.** La arena anima el tiempo del episodio (anillos que se expanden a la velocidad `c`); el resto de la interfaz no se mueve. Con `prefers-reduced-motion` la reproducción arranca en pausa.
- **Lienzo.** Sobre `void`, dibuja la arena con un borde `line-strong`, los anillos de señal con trazo de 2 px y alfa decreciente con la distancia, y los anillos ambientales (crecer) a alfa 0.12. El estallido es un disco de anillos concéntricos `burst` que se apaga en cuatro turnos.

## Iconografía

No hay conjunto de iconos. Usa texto y glifos tipográficos mínimos (`◀`, `▶`), siempre con una etiqueta accesible. Las marcas de la arena son formas geométricas simples: disco, anillo, cruz y rombo.

## Reglas de uso

- Compón cada pantalla con `ob-panel` sobre `void`; dentro, `ob-section` para titular cada grupo de controles.
- Cada campo es etiqueta, control y ayuda opcional, en ese orden y con `ob-field`.
- Usa `ob-btn--primary` una vez por panel («Simular»); el resto son `ob-btn` o `ob-btn--ghost`.
- Muestra siempre la leyenda (`ob-legend`) junto a la arena y la tabla (`ob-table`) junto al episodio.
- Contraste: texto 4.5:1 y marcas 3:1 en ambos temas; comprueba cada color nuevo en los dos antes de añadirlo.

## Narrativa: galaxia, sistemas y astronautas

La demo cuenta cada episodio como una historia espacial sin cambiar los datos que lo sostienen.

- Trata la arena como una galaxia: el toro de la simulación es su mapa. Cada civilización es un sistema estelar con su planeta, y lo identifica el color `civ-*` de siempre.
- Dale a cada civilización el astronauta de su estrategia (`Astronauta`): ocho pieles con silueta, insignia y colores propios. Los colores de las pieles son arte, no tokens de interfaz; no los uses fuera del astronauta.
- Muestra a la civilización seleccionada con `FichaCivilizacion`: escáner, nombre del sistema, rasgos, datos y última acción.
- Muestra el crecimiento con `MedidorKardashev`, una escala de Kardashev ilustrativa: los recursos son adimensionales y la etapa no mide energía. Rotúlala siempre como «ilustrativa».
- Cuenta los hechos con `Bitacora`, en frases cortas, con una forma y una palabra por tipo.

**Decoración frente a dato.**

- Es dato lo que sale de `ep`: posiciones, recursos, acciones, señales, proyectiles, estallidos, muertes y la etapa derivada de los recursos. Dibújalo exactamente como llega.
- Es decoración el aspecto del planeta, el mundo, el nombre del sistema, el tamaño de la flota y la piel. Márcala como tal: borde discontinuo y la palabra «decorativo» en los rasgos, y un comentario en el código.
- Ofrece siempre un modo científico sin pieles: sin astronautas, planetas ni flotas, con discos, anillos, leyenda y tabla, y la política escrita como texto. El modo científico es el de referencia para comparar escenas y para imprimir.
- Ninguna decoración puede decir algo que los datos no digan: no apagues, inclines ni armes a un astronauta sin que `ep` lo indique.

**La luz de emisión significa riesgo de detección.** Una civilización que crece, transmite o comparte emite luz y se hace detectable; una que se oculta o se refugia no. Dibuja esa luz solo desde `emision[t][i]` (baja, activa, alta), acompáñala de su palabra («Emisión alta») y no la uses como adorno ni para destacar una civilización.

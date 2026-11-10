Astronauta de cuerpo entero que representa a una civilización según su estrategia. Hay ocho pieles; cada una combina una silueta propia, un traje, un acento, un visor y una insignia en el pecho. Es decoración con significado: la piel dice qué política juega la civilización, pero no aporta ningún dato del episodio.

**El consumidor aporta** el estilo (`OB.estilos.deEstrategia(ep.politicas[i])`, con `id`, `traje`, `acento`, `visor`, `insignia`) y el estado (`{viva, refugio, accion, …}`). Con ellos llama a `OB.astronauta.svg(estilo, estado, tam)` y coloca el resultado dentro de la ficha o de la arena. Un SVG insertado necesita su texto alternativo (`<title>`), que el módulo ya incluye. Las clases `ob-astro`, `ob-astro__nombre`, `ob-astro__insignia` y `ob-astro__estado` solo sirven para la figura con su pie.

**Colores de las pieles.** Los colores de traje, acento y visor son arte de las pieles. No son tokens de interfaz: no están en `tokens.json`, no cambian con el tema y no se reutilizan en controles, texto ni marcas de la arena. Los trazos del contorno, el escudo, los pulsos y las grietas sí usan tokens (`line-strong`, `refuge`, `share`, `dead`, `ink`).

| Piel | Silueta | Traje | Acento | Visor | Insignia |
|---|---|---|---|---|---|
| Cooperadora | Casco redondo, brazos abiertos | `#eef3f8` | `#2f9bd6` | `#8fdcff` | Ola |
| Recíproca | Casco cuadrado, un brazo arriba y otro abajo | `#e6efe2` | `#3d9a5b` | `#a6e8b4` | Hoja |
| Prudente | Capucha ancha, brazos cruzados | `#f2f6fb` | `#5a7fc4` | `#cfe3ff` | Cristal |
| Mentirosa | Casco en rombo, visor mitad espejo | `#3b2a4d` | `#d98bb8` | `#f0b6dd` | Espejo |
| Solitaria | Figura pequeña, sombrero de ala y farol | `#e7d7b8` | `#c2813a` | `#ffd9a0` | Duna |
| Belicosa | Casco angular con cresta, hombreras y cañones | `#4a2a26` | `#ff7a45` | `#ffb48a` | Llama |
| Callada | Cono alto sin brazos, visor en ranura | `#c9ccd6` | `#8a90a8` | `#dfe3ee` | Nube |
| Neuronal | Casco hexagonal, antena y brazos con nodos | `#dff6f0` | `#00a58e` | `#9ff4e2` | Nodos |

| Estado | Cómo se ve | Palabra |
|---|---|---|
| Viva | Visor encendido. | Viva |
| Muerta | Cuerpo inclinado a la deriva, visor apagado (`dead`) y agrietado. | Muerta |
| Refugio | Burbuja de escudo `refuge` con borde discontinuo. | Refugio |
| Comparte | Pulso de dos arcos `share` en la antena (también al transmitir). | Comparte |
| Miente | Reflejo de espejo en el visor. | Miente |
| Ataca | Brazo extendido con lanzasondas. | Ataca |

**Estados.** Un astronauta puede sumar estados: en refugio y compartiendo a la vez, por ejemplo. La muerte anula el resto. Cada estado cambia una forma, no solo un color.

**Accesibilidad.** Es una imagen (`role="img"` con `<title>`) que nombra piel y estado: «Astronauta belicosa, viva». No recibe foco ni responde al teclado; si va dentro de un botón o una fila, el control es el que lleva el foco. El contorno `line-strong` de 2 px mantiene la silueta a 3:1 sobre `surface` y `raised` en ambos temas, incluso con trajes muy claros o muy oscuros. En modo científico no se muestra (ver «Narrativa» en el README del sistema); la política se escribe como texto.

| Haz | No hagas |
|---|---|
| Muestra el nombre de la piel y la política junto al dibujo. | Dejar que el traje sea la única señal de la estrategia. |
| Usa `OB.estilos` para elegir colores e insignia. | Escribir a mano los hex de una piel en otro componente. |
| Marca el astronauta como decorativo: no representa un dato. | Inclinar o apagar un astronauta sin que la civilización haya muerto en `ep`. |
| Dibuja refugio, mensaje, mentira y ataque solo si `ep` lo dice. | Añadir pulso o lanzasondas por efecto. |

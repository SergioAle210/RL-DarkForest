Bitácora del episodio. Es una lista cronológica de los hechos que importan hasta el turno actual: estallidos, muertes, ataques, mensajes, mentiras, refugios e hitos de la escala de Kardashev. Cuenta la historia de la escena en frases cortas; no sustituye a la tabla de cifras.

**El consumidor aporta** el episodio `ep` y el turno `t`. `OB.bitacora.eventos(ep)` devuelve `[{t, tipo, civs, texto}]` ordenado por turno, y `OB.bitacora.render(contenedor, eventos, t, opciones)` pinta la lista hasta `t`, resalta la última entrada y se desplaza al final. Cada `texto` va en español claro y sin jerga («Estallido de la anomalía; C3 no estaba refugiada»). El módulo colapsa lo repetitivo: del intercambio entre dos civilizaciones solo aparece el primer mensaje y un resumen cada 25 turnos, y los refugios solo cuando empiezan.

| Clase | Contenido |
|---|---|
| `ob-bitacora` | El contenedor desplazable, con borde `line`. |
| `ob-bitacora__item` | Una entrada. Su `data-tipo` fija marca y color. La última lleva `aria-current="true"`. |
| `ob-bitacora__turno` | Turno en `data` («T31»). |
| `ob-bitacora__tipo` | Marca y palabra del tipo, en `label`. |
| `ob-bitacora__texto` | La frase. |
| `ob-bitacora__vacio` | Texto cuando aún no hay eventos: «Sin eventos hasta el turno 0.» |

| `data-tipo` | Marca | Color | Cuándo |
|---|---|---|---|
| `estallido` | Estrella de cuatro puntas | `burst` | La anomalía estalla en `ep.bursts`. |
| `muerte` | Cruz | `dead` | Una civilización muere (`ep.muertes`). |
| `ataque` | Triángulo | `danger` | Se lanza un proyectil (`ep.projectiles`). |
| `mensaje` | Disco | `share` | Primer mensaje entre dos civilizaciones y resúmenes. |
| `mentira` | Semidisco | `lie` | Una señal con `msg` 3. |
| `refugio` | Anillo | `refuge` | Una civilización empieza un refugio. |
| `hito` | Rombo | `ink` | Cambio de etapa de Kardashev (ilustrativa). |

**Estados.** Entrada ordinaria; entrada actual (fondo `raised` y filete `ink` a la izquierda); y lista vacía. La entrada del turno mostrado siempre es la última visible. No hay hover ni selección.

**Accesibilidad.** El contenedor es `role="log"` con `aria-live="off"`: la reproducción avanza turno a turno y no debe anunciar cada fotograma. Un lector de pantalla recorre la lista con las flechas cuando el usuario enfoca la bitácora. Lleva `aria-label` con el turno («Bitácora del episodio hasta el turno 40») y `tabindex="0"` para que el teclado pueda desplazarla; su anillo de foco es el de `focus`. Cada tipo tiene color, forma y palabra: el color no es la única señal. La marca más clara (`dead` en `carta`) cumple 3:1 y el texto 4.5:1 en ambos temas.

| Haz | No hagas |
|---|---|
| Escribe el tipo con su palabra, además de la marca. | Distinguir los tipos solo por el color de la marca. |
| Usa `aria-live="off"` durante la reproducción. | Anunciar cada evento nuevo mientras avanza el tiempo. |
| Colapsa el intercambio repetido de mensajes. | Registrar un mensaje por turno entre las mismas dos civilizaciones. |
| Escribe solo hechos que están en `ep`. | Inventar relatos, nombres o motivos que no vienen de los datos. |

Ficha de civilización. Presenta una civilización como un sistema estelar con su planeta: un escáner con el astronauta de su estrategia y la miniatura del planeta, el nombre del sistema, sus rasgos en fichas, unos pocos datos y su última acción. Se abre al seleccionar una civilización en la arena o en la tabla y se actualiza con el turno.

**El consumidor aporta** el episodio `ep`, el índice `i` de la civilización, el turno `t` y los tokens `T`, y llama a `OB.tarjeta.render(contenedor, ep, i, t, T)`. El nombre del sistema sale de `OB.nombres.generar(semilla)`; el astronauta, de `OB.astronauta.svg`; el planeta, de `OB.planeta.miniatura`; la etapa, de `OB.kardashev`. Si falta un módulo, la ficha pone un reemplazo de texto en su lugar y no falla.

| Clase | Contenido |
|---|---|
| `ob-ficha` | La tarjeta. Define `--c`, el color `civ-*` de la civilización (`dead` si `data-dead="true"`), para el filete superior y la mini gráfica. |
| `ob-ficha__escaner` | Retícula de escaneo con esquinas de mira. Contiene el planeta y el astronauta. |
| `ob-ficha__planeta` | Miniatura del planeta. Decorativa (`data-decorativo="true"`, `aria-hidden`). Una civilización eliminada lo muestra apagado con una cruz. |
| `ob-ficha__astronauta` | El SVG del astronauta, con su `<title>`. |
| `ob-ficha__nombre` | `<h3>` en `title-md` con el nombre del sistema y, encima en `label`, «Civilización N». |
| `ob-ficha__chips` | Lista de rasgos. Cada uno es un `ob-chip`: política, emisión, estado y mundo. |
| `ob-ficha__stat` | Un dato en una lista de definición (`dt` etiqueta, `dd` cifra en `data`): recursos con mini gráfica, etapa de Kardashev, mensajes enviados y recibidos, turno. |
| `ob-ficha__accion` | Última acción hasta el turno `t`, en una frase. |

**Variantes.**

| Variante | Cuándo |
|---|---|
| Viva | Estado ordinario: filete `civ-*` y planeta encendido. |
| Eliminada (`data-dead="true"`) | La civilización murió antes de `t`: filete, planeta y visor en `dead`, y la última acción dice cómo y cuándo. |
| `ob-chip[data-decorativo="true"]` | Rasgo decorativo (mundo): borde discontinuo y texto `muted`, además de la palabra «decorativo». |

**Estados.** La ficha es informativa; no tiene hover ni selección propia. El resplandor `shadow-glow` solo lo lleva una ficha destacada, y una por pantalla. La emisión se escribe en el chip con un glifo de tres pasos (`○` baja, `◐` activa, `●` alta) y su palabra: la luz de emisión es el riesgo de ser detectada.

**Accesibilidad.** Es un `<article>` con `aria-labelledby` apuntando al nombre. Los rasgos son una lista con `aria-label="Rasgos"`; los datos, una `<dl>`. El escáner y el planeta no se anuncian; el astronauta sí, con su título. Las mini gráficas llevan `role="img"` y `aria-label` con el rango. No hay elementos interactivos propios, así que el foco lo recibe el control que la abre. Todo el texto cumple 4.5:1 sobre `surface` y `raised` en `noche` y `carta`; la retícula y el contorno de las marcas cumplen 3:1.

| Haz | No hagas |
|---|---|
| Marca como decorativo el mundo y el nombre del sistema. | Presentar el nombre o el mundo como si salieran de la simulación. |
| Muestra la etapa con la palabra «ilustrativa». | Escribir la etapa de Kardashev como una medida física. |
| Escribe el estado con palabras en el chip («Eliminada»). | Indicar la muerte solo con el gris. |
| Limita los datos a recursos, etapa, mensajes y turno. | Añadir métricas que no vienen de `ep`. |

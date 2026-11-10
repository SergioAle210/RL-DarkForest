# Demo: visualización de episodios

La demo reproduce episodios del entorno con las políticas y mundos que se elijan. No cambia el
entorno ni el entrenamiento: solo lo simula y lo muestra.

## Interfaz web (simulación en vivo)

```bash
uv run python -m demo.servidor        # http://localhost:8765
```

Elige el mundo (letalidad λ, velocidad de la señal c, ruido, costo del refugio, probabilidad de
matar, número de civilizaciones, engaño y señales costosas), una política por civilización
(siete estrategias fijas y las redes entrenadas que haya en las carpetas `kout*` junto al repo) y
la semilla. Cada mundo nuevo recompila el simulador (unos segundos). «Promediar 40 episodios» da el
promedio, que es lo representativo: un episodio suelto no lo es.

## Vista 3D en la propia interfaz

El botón «Toro 3D» sobre la arena cambia a un toro interactivo (three.js): se gira arrastrando y se
acerca con la rueda. Dibuja lo mismo que la vista plana (civilizaciones con un disco cuyo tamaño
depende de sus recursos, o su planeta en modo narrativo, anillos de señal, proyectiles, estallido y refugios) con los colores
del tema activo. La biblioteca va incluida en `demo/web/vendor/` (three.js r128 y OrbitControls,
licencia MIT), así que funciona sin conexión y también con el catálogo.

## Modo narrativo

El botón «Modo» alterna entre narrativo y científico. En el narrativo la arena es una galaxia: cada
civilización es un sistema estelar con un planeta, un astronauta cuyo traje indica su estrategia
(ocho pieles), una escala de Kardashev ilustrativa que crece con los recursos, y sondas, flota,
escudo y supernova dibujados desde los datos del episodio. A un lado hay una ficha por civilización
(selector C1…CN) y una bitácora de eventos. El aspecto del planeta, los nombres, el tamaño de la
flota y la etapa de Kardashev son decorativos; posiciones, recursos, acciones, señales,
proyectiles, estallidos y muertes salen del episodio. El modo científico deja solo discos,
anillos, leyenda y tabla, y es el de referencia para comparar escenas. Los módulos están en
`demo/web/modulos/` (contrato en su `README.md`).

En la vista 3D narrativa cada civilización es una estrella con su planeta en una órbita punteada (decorativa) sobre un cielo estrellado. Las teclas 1 a 9 saltan con efecto warp hasta el planeta de esa civilización y lo siguen; 0 o Escape vuelven a la vista general. Arrastrar gira la cámara y la rueda acerca.

## Otras lecturas del episodio

- **Partitura de acciones:** una fila por civilización y una marca por acción y turno (compartir,
  símbolo, mentira, ataque, refugio), con los estallidos como franjas. Sirve para comparar
  estrategias y leer la cooperación sin 3D.
- **Red de comunicación:** las civilizaciones en su posición, con una flecha del emisor al receptor
  por cada mensaje explícito recibido hasta el turno actual (más gruesa cuantos más) y un borde
  ámbar para las que forman una comunidad (mensajes en ambos sentidos, al menos 3). Con más de 2
  civilizaciones permite ver si se forman grupos.

## Catálogo para presentar sin servidor

```bash
uv run python -m demo.catalogo        # escribe demo/web/catalogo/
```

Genera doce escenarios con su promedio de 40 episodios. La página los carga sola si no encuentra el
servidor, así que `demo/web/` se puede abrir desde cualquier servidor estático.

## Visor 3D (PyVista)

```bash
uv sync --locked --group visual3d
uv run python -m demo.visor3d --episodio demo/web/catalogo/escenario_01.json            # ventana
uv run python -m demo.visor3d --episodio <json> --turno 60 --out foto.png --zona        # imagen
uv run python -m demo.visor3d --episodio <json> --out video.gif --cada 2 --giro         # animación
```

La arena es un toro, así que se dibuja como un toro: las civilizaciones son esferas con un pilar
cuya altura crece con el logaritmo de sus recursos, las señales son anillos que viajan por la
superficie a la velocidad c y el estallido ilumina la zona de la anomalía. En la ventana, la barra
elige el turno y la barra espaciadora pausa. Con `--tema carta` se dibuja sobre papel claro.

## Sistema de diseño

La interfaz usa el sistema «Observatorio del Bosque Oscuro»: tokens de color en dos temas (noche y
carta), tipografía, espaciado y ocho componentes con vista previa. Sus archivos están en
`demo/diseno/project/`; `tokens.json` es la única fuente de los colores de la web y del visor 3D.

```bash
uv run python -m demo.diseno.generar_css   # tokens.json a demo/web/tokens.css y componentes.css
```

Los pares de color cumplen contraste 4.5:1 en texto y 3:1 en marcas en ambos temas, y una prueba lo
comprueba (`tests/test_demo.py`). El sistema también está publicado como artefacto de Claude
(privado) para consultarlo y editarlo.

## Formato del episodio

- `emision`: (turnos + 1) filas por civilización con la luz que la hace detectable (0 baja, 1 alta; el entorno no emite 2).
- `muertes`: una entrada `{t, civ, causa, atacante}` por civilización muerta, con `t` el turno en que ya está muerta, `causa` `anomalia` o `ataque` y `atacante` su origen o -1.
- `politicas`: texto de la política de cada civilización (`fijo:<estrategia>` o la etiqueta de la red); vacío si no se indicó.

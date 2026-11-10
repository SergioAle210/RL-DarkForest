# Módulos de la demo: contrato compartido

La demo es una página (`demo/web/index.html`) que reproduce un episodio de RL-DarkForest. Estos
módulos añaden la capa narrativa: cada civilización es un **sistema estelar con su planeta**, vive
en una **galaxia** (la arena, un toro) y se representa con un **astronauta** cuyo traje depende de
la estrategia que juega. El integrador (no los autores de módulo) cablea todo en `index.html`.

## Convenciones

- Scripts clásicos (sin `import`/`export`), cada uno asigna su API a `window.OB.<nombre>` dentro de
  una función autoejecutada, igual que `estilos.js`. Sin dependencias salvo `THREE` (three.js r128,
  global) en las funciones `three`, que lo reciben como primer argumento.
- Español en comentarios, textos de usuario y nombres de API. Sin emojis. Comentarios breves y
  útiles; JSDoc corto (`/** ... */`) antes de cada función pública.
- No toques el DOM global ni `document.body`: trabaja sobre los contenedores o contextos que te
  pasen. No mutes `ep`. Las funciones de dibujo deben ser rápidas (se llaman en cada fotograma).
- Todo es determinista: la misma semilla da el mismo resultado. Usa un generador pseudoaleatorio
  propio con semilla (por ejemplo mulberry32), nunca `Math.random()` en la apariencia.
- Colores de interfaz: recíbelos en el objeto `T` (tokens, ver abajo). Los colores de piel salen de
  `OB.estilos` (`traje`, `acento`, `visor`, `mochila`). Debe leerse bien en fondo oscuro y claro.
- **Decoración frente a dato.** Lo que dibujes con datos reales (posiciones, recursos, acciones,
  señales, proyectiles, estallidos, muertes) debe coincidir con `ep`. Lo decorativo (aspecto del
  planeta, tamaño de flota, nombres) debe estar marcado como tal en un comentario.
- Accesibilidad: todo SVG con texto alternativo (`<title>`), nada que dependa solo del color.
- **No uses el navegador integrado** (lo comparte el integrador). Verifica con `node --check` y con
  pruebas en Node (ver abajo). Si quieres una vista previa, crea una página estática
  `demo/web/prueba_<modulo>.html` que cargue los scripts por rutas relativas; el integrador la
  abrirá.

## Datos del episodio `ep`

Campos existentes: `n` (civilizaciones), `steps`, `c` (velocidad de señal), `v` (de proyectil),
`pos[i] = [x, y]` en [0, 1) (toro), `alive[t][i]` (0/1, `t` de 0 a `steps`), `res[t][i]` recursos,
`refuge[t][i]` (0/1), `kind[t][i]` acción ejecutada en el turno `t` (0 crecer, 1 ocultarse,
2 observar, 3 transmitir, 4 compartir, 5 atacar, 6 refugiarse, -1 ninguna; `t` de 0 a `steps-1`),
`act[t][i]` acción cruda, `signals[{o, t, msg}]` (`msg`: 0 ambiental, 1 símbolo, 2 compartir,
3 mentira), `projectiles[{o, t, tx, ty}]` (origen, turno de lanzamiento, destino), `bursts[t...]`
(turnos con estallido), `zone {x, y, r}`, `mensajes[[t, receptor, emisor]]`, `deception`,
`n_symbols`, `policy` (texto), `seed`, `resumen`, `promedio`.

Campos nuevos (los añade `viz/exportar_json.py`; los módulos deben tolerar que falten):
- `politicas`: lista de `n` textos, la política de cada civilización (`"fijo:coop_real"`,
  `"red ft_l0p5_c0p22 (seed 0)"`, …). Si falta, usa `OB.estilos.deEstrategia(ep.policy)`.
- `emision[t][i]`: 0 baja, 1 activa (creció), 2 alta (transmitió o compartió). Es la luz que la
  hace detectable.
- `muertes`: `[{t, civ, causa: "anomalia" | "ataque", atacante: índice o -1}]`.

Tokens `T`: objeto con colores hexadecimales: `void, surface, raised, line, line-strong, ink, muted,
sun, on-sun, focus, share, symbol, lie, burst, refuge, danger, ok, dead, civ-1 … civ-8` y `sans`
(familia tipográfica). `T["civ-" + (i % 8 + 1)]` es el color de la civilización `i`.

## Estilos (`OB.estilos`, ya escrito, no lo modifiques)

`OB.estilos.deEstrategia(politica)` devuelve `{id, nombre, rol, mundo, traje, acento, visor,
mochila, insignia}`. Mundos: `oceanico, selvatico, helado, nebuloso, desertico, volcanico, gaseoso,
cristalino`. Insignias: `ola, hoja, cristal, espejo, duna, llama, nube, nodos`.

## Estado de una civilización

`estado = {viva: bool, emite: 0|1|2, refugio: bool, accion: "crecer"|"ocultarse"|"observar"|
"transmitir"|"compartir"|"atacar"|"refugiarse"|"mentir"|"", recursos: number, nivel: 0..3}`.

## API por módulo (firmas exactas)

**`astronauta.js` → `OB.astronauta`**
- `svg(estilo, estado, tam = 96) -> string` marcado SVG (astronauta de cuerpo entero, de frente).
  Muestra: visor encendido si `viva`, apagado y agrietado si no (cuerpo inclinado, a la deriva);
  burbuja de escudo si `refugio`; pulso en la antena si `accion` es `compartir`/`transmitir`; visor
  con reflejo espejado si `accion` es `mentir`; brazo con lanzasondas si `accion` es `atacar`;
  insignia del estilo en el pecho; mochila y acentos con los colores del estilo.
- `insignia(id, tam = 24) -> string` SVG de la insignia sola.
- `canvas(ctx, estilo, estado, x, y, alto, t)` dibuja el mismo astronauta en un contexto 2D con el
  centro de los pies en `(x, y)` y altura `alto` píxeles; `t` (segundos) anima un balanceo suave.
- `three(THREE, estilo) -> THREE.Group` astronauta de baja poligonación (cápsula, casco, visor,
  mochila, extremidades) de unos 0,5 unidades de alto con el origen en los pies, y
  `grupo.userData.actualizar(estado, t)` que cambia visor, escudo, antena y postura.

**`planetas.js` → `OB.planeta`**
- `textura(mundo, semilla, ancho = 512, alto = 256) -> HTMLCanvasElement` textura equirrectangular
  procedural del mundo (océanos con continentes, dunas, lava con grietas, nubes de gas…).
- `miniatura(mundo, semilla, tam = 96) -> HTMLCanvasElement` disco sombreado con atmósfera.
- `canvas(ctx, mundo, semilla, x, y, radio, t, estado)` planeta en 2D (con `estado.viva === false`
  apagado, grietas rojas).
- `three(THREE, mundo, semilla, radio = 1) -> THREE.Group` esfera con la textura, capa de atmósfera
  aditiva y nubes opcionales; `grupo.userData.actualizar(t, estado)` rota y apaga si no `viva`.

**`kardashev.js` → `OB.kardashev`** (escala **ilustrativa**: los recursos son adimensionales)
- `nivel(recursos, referencia = 12) -> number` = `clamp(3 * ln(max(recursos, 1)) / ln(referencia),
  0, 3)`.
- `etapa(nivel) -> {id: "0"|"I"|"II"|"III", nombre, descripcion}` (umbrales 1, 2 y 3 hacia arriba:
  `id` es la etapa alcanzada; «0» hasta 1; nombres: planetaria incipiente, planetaria, estelar,
  galáctica).
- `medidorSvg(nivel, ancho = 160) -> string` barra de 0 a III con marcas I, II y III y el valor.
- `canvas(ctx, x, y, radioPlaneta, nivel, t, T)` dibuja la etapa alrededor del planeta: luces de
  ciudad y anillo orbital (I), enjambre de Dyson alrededor de la estrella (II), resplandor que se
  extiende (III).
- `three(THREE, radio) -> THREE.Group` con `userData.actualizar(nivel, t)` (anillo orbital,
  enjambre de instancias, halo).

**`efectos.js` → `OB.efectos`**
- `crear2d() -> {dibujar(ctx, ep, t, h)}` donde `h = {P(q) -> [px, py] (q = [x, y] unitario),
  wrap(v), T, civ(i), escala}` y `t` es el turno (entero). Dibuja desde los datos: sondas con estela
  y **flota decorativa** (3 a 5 naves en formación, determinista por origen y turno) por cada
  `ep.projectiles` en vuelo, impacto con onda de choque al llegar, escudo de cúpula con
  `ep.refuge`, planeta que se apaga al morir (`ep.muertes` o `ep.alive`), y la supernova de la
  anomalía en los turnos de `ep.bursts` sobre `ep.zone`.
- `crear3d(THREE, escena, h3) -> {actualizar(ep, t)}` con `h3 = {p3(x, y, alto) -> THREE.Vector3,
  n3(x, y) -> THREE.Vector3 (normal), T, civ(i)}`: la misma lógica en 3D, con instancias; limpia
  lo que crea.

**`bitacora.js` → `OB.bitacora`**
- `eventos(ep) -> [{t, tipo, civs, texto}]` ordenado por `t`. Tipos: `estallido, muerte, ataque,
  mensaje, mentira, refugio, hito` (cambio de etapa de Kardashev). Texto en español, claro y sin
  jerga («Turno 31: estallido de la anomalía; C3 no estaba refugiada y se apaga»). Colapsa lo
  repetitivo: del intercambio de mensajes entre dos civilizaciones, solo el primero y un resumen
  cada 25 turnos; los refugios, solo cuando empiezan.
- `render(contenedor, eventos, t, opciones = {})` pinta la lista hasta el turno `t` (la última
  entrada resaltada, desplazamiento al final), con `role="log"` y sin anunciar cada fotograma
  (`aria-live="off"`).

**`tarjeta.js` → `OB.tarjeta`** y **`nombres.js` → `OB.nombres`**
- `OB.nombres.generar(semilla) -> string` nombre de sistema procedural, pronunciable, determinista
  («Thalassa Prime», «Ketu-47»…).
- `OB.tarjeta.render(contenedor, ep, i, t, T)` pinta la ficha de la civilización `i` en el turno
  `t`, al estilo de las fichas de descubrimiento de los juegos de exploración espacial (retícula de
  escaneo, rasgos en fichas, nombre de sistema): astronauta (`OB.astronauta.svg`), miniatura del
  planeta (`OB.planeta.miniatura`), medidor Kardashev, política y rol, estado, recursos con mini
  gráfica, mensajes enviados y recibidos hasta `t`, y una línea con su última acción. Si falta algún
  otro módulo, muestra un reemplazo de texto en vez de fallar.
- Estilos en `modulos/tarjeta.css`, solo con variables CSS de tokens (`var(--surface)`, …).

## Propiedad de archivos y pruebas

Cada agente escribe solo los archivos de su módulo (y su prueba). Pruebas en Node:
`demo/web/modulos/pruebas/<modulo>.test.mjs`, ejecutables con `node <archivo>`; cargan el script con
`globalThis.window = globalThis` y `await import(...)`/`vm.runInThisContext`, y usan `node:assert`.
Para el dibujo, comprueba que no lanza y que llama a las operaciones esperadas con un contexto 2D
falso. Termina cada informe indicando qué funciones implementaste, qué decisiones tomaste y si te
apartaste de este contrato.

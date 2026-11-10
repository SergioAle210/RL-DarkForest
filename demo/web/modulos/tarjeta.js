/* Ficha de civilización: sistema descubierto, astronauta, planeta y datos del turno.
 * Datos reales: estado, recursos, mensajes, mentiras, acción. Decoración: nombre del sistema,
 * aspecto del planeta, retícula de escaneo. */
(function (OB) {
  "use strict";

  const ACCIONES = ["crecer", "ocultarse", "observar", "transmitir", "compartir", "atacar",
    "refugiarse"];
  const VERBOS = {
    crecer: "Crece", ocultarse: "Se oculta", observar: "Observa", transmitir: "Transmite",
    compartir: "Comparte", atacar: "Ataca", refugiarse: "Se refugia", mentir: "Miente",
  };
  const MUNDOS = {
    oceanico: "oceánico", selvatico: "selvático", helado: "helado", nebuloso: "nebuloso",
    desertico: "desértico", volcanico: "volcánico", gaseoso: "gaseoso", cristalino: "cristalino",
  };
  const EMISION = ["baja", "activa", "alta"];
  const ETAPAS = [
    { id: "0", nombre: "planetaria incipiente" }, { id: "I", nombre: "planetaria" },
    { id: "II", nombre: "estelar" }, { id: "III", nombre: "galáctica" },
  ];
  const TAM_ASTRONAUTA = 120;
  const TAM_PLANETA = 96;
  const GRAFICA = { w: 170, h: 44, margen: 3, maxPuntos: 120 };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /** Número con coma decimal; menos decimales cuanto mayor es el valor. */
  function num(x, d) {
    const v = Number(x) || 0;
    const dec = d !== undefined ? d : Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2;
    return v.toFixed(dec).replace(".", ",");
  }

  const acota = (t, lo, hi) => Math.max(lo, Math.min(hi, t));

  /** Mezcla estable de dos enteros (decorativo: semilla del nombre y del planeta). */
  function semillaCiv(ep, i) {
    let h = (Number(ep && ep.seed) | 0) ^ Math.imul((i | 0) + 1, 0x9e3779b1);
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
    return (h ^ (h >>> 16)) >>> 0;
  }

  function nivelDe(r) {
    const K = OB.kardashev;
    if (K && typeof K.nivel === "function") return K.nivel(r);
    return Math.min(3, Math.max(0, (3 * Math.log(Math.max(r, 1))) / Math.log(12)));
  }

  function etapaDe(k) {
    const K = OB.kardashev;
    if (K && typeof K.etapa === "function") {
      const e = K.etapa(k);
      if (e && e.id) return { id: e.id, nombre: e.nombre || "" };
    }
    return ETAPAS[Math.min(3, Math.floor(k))];
  }

  /** Nombre de la acción de la civilización `i` en el turno `t` ("" si no hay). */
  function accionEn(ep, i, t) {
    const fila = ep.kind && ep.kind[acota(t, 0, ep.steps - 1)];
    const k = fila ? fila[i] : -1;
    if (k === undefined || k < 0 || k >= ACCIONES.length) return "";
    if (ep.deception && k === 3) {
      const a = ep.act && ep.act[acota(t, 0, ep.steps - 1)];
      if (a && a[i] === 2 + ep.n_symbols) return "mentir";
    }
    return ACCIONES[k];
  }

  /**
   * Estado de la civilización `i` en el turno `t`, en el formato que usan los módulos hermanos.
   * @returns {{viva:boolean, emite:number, refugio:boolean, accion:string, recursos:number,
   *   nivel:number}}
   */
  function estado(ep, i, t) {
    const tt = acota(t, 0, ep.alive ? ep.alive.length - 1 : ep.steps);
    const recursos = ep.res && ep.res[tt] ? ep.res[tt][i] : 0;
    return {
      viva: !!ep.alive && ep.alive[tt][i] === 1,
      emite: ep.emision ? ep.emision[Math.min(t, ep.steps - 1)][i] : 0,
      refugio: !!ep.refuge && !!ep.refuge[tt] && ep.refuge[tt][i] === 1,
      accion: accionEn(ep, i, t),
      recursos,
      nivel: nivelDe(recursos),
    };
  }

  /** Ejecuta un módulo hermano; si falta o falla devuelve el reemplazo de texto. */
  function seguro(fn, reemplazo) {
    try {
      const r = fn();
      return r === undefined || r === null || r === "" ? reemplazo : r;
    } catch (e) {
      return reemplazo;
    }
  }

  const aviso = (txt) => '<span class="ob-ficha__reemplazo">' + esc(txt) + "</span>";

  // Retícula decorativa: anillos concéntricos y marcas de ángulo.
  function escaner() {
    let marcas = "";
    for (let a = 0; a < 360; a += 15) {
      const r = (a * Math.PI) / 180;
      const largo = a % 90 === 0 ? 12 : a % 30 === 0 ? 8 : 4;
      const x1 = (100 + Math.cos(r) * 96).toFixed(1);
      const y1 = (100 + Math.sin(r) * 96).toFixed(1);
      const x2 = (100 + Math.cos(r) * (96 - largo)).toFixed(1);
      const y2 = (100 + Math.sin(r) * (96 - largo)).toFixed(1);
      marcas += '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '"/>';
    }
    return '<svg class="ob-ficha__escaner" viewBox="0 0 200 200" '
      + 'preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">'
      + '<g fill="none" stroke="currentColor" stroke-width="1">'
      + '<circle cx="100" cy="100" r="96"/>'
      + '<circle cx="100" cy="100" r="68" stroke-dasharray="3 4"/>'
      + '<circle cx="100" cy="100" r="40"/><circle cx="100" cy="100" r="14"/>'
      + '<path d="M4 100h192M100 4v192"/>' + marcas + "</g></svg>";
  }

  // Máximo de recursos del episodio por civilización (eje vertical fijo de la gráfica).
  const maximos = new WeakMap();
  function maximoRecursos(ep, i) {
    let m = maximos.get(ep);
    if (!m) { m = []; maximos.set(ep, m); }
    if (m[i] === undefined) {
      let mx = 1;
      for (const f of ep.res || []) if (f[i] > mx) mx = f[i];
      m[i] = mx;
    }
    return m[i];
  }

  /** Mini gráfica de recursos de 0 a `t` (SVG polyline con datos reales de ep.res). */
  function grafica(ep, i, t, color, actual) {
    const { w, h, margen } = GRAFICA;
    const tt = Math.min(t, (ep.res || []).length - 1);
    const total = Math.max(ep.steps, 1);
    const mx = maximoRecursos(ep, i);
    const paso = Math.max(1, Math.ceil((tt + 1) / GRAFICA.maxPuntos));
    const px = (k) => margen + (k / total) * (w - 2 * margen);
    const py = (v) => h - margen - (Math.max(v, 0) / mx) * (h - 2 * margen);
    const pts = [];
    let ult = 0;
    const punto = (k) => px(k).toFixed(1) + "," + py(ep.res[k][i]).toFixed(1);
    for (let k = 0; k <= tt; k += paso) { pts.push(punto(k)); ult = k; }
    if (ult !== tt && tt >= 0) pts.push(punto(tt));
    const cx = px(tt < 0 ? 0 : tt).toFixed(1);
    const cy = py(actual).toFixed(1);
    return '<svg class="ob-ficha__grafica" viewBox="0 0 ' + w + " " + h + '" width="' + w
      + '" height="' + h + '" role="img"><title>Recursos de la civilización ' + (i + 1)
      + " hasta el turno " + t + ": " + num(actual) + " (máximo del episodio " + num(mx)
      + ')</title><line class="ob-ficha__base" x1="' + margen + '" y1="' + (h - margen) + '" x2="'
      + (w - margen) + '" y2="' + (h - margen) + '"/><polyline fill="none" stroke="' + color
      + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="'
      + pts.join(" ") + '"/><circle cx="' + cx + '" cy="' + cy + '" r="3" fill="' + color
      + '"/></svg>';
  }

  // Miniatura del planeta cacheada por contenedor (clave: mundo, semilla, tamaño).
  const planetas = new WeakMap();
  function miniatura(contenedor, mundo, semilla) {
    if (!(OB.planeta && typeof OB.planeta.miniatura === "function")) return null;
    const clave = mundo + "|" + semilla + "|" + TAM_PLANETA;
    const previo = planetas.get(contenedor);
    if (previo && previo.clave === clave) return previo.canvas;
    const canvas = seguro(() => OB.planeta.miniatura(mundo, semilla, TAM_PLANETA), null);
    if (canvas && canvas.setAttribute) {
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", "Planeta " + (MUNDOS[mundo] || mundo));
    }
    planetas.set(contenedor, { clave, canvas });
    return canvas;
  }

  function contar(ep, i, t) {
    let enviados = 0;
    let recibidos = 0;
    for (const m of ep.mensajes || []) {
      if (m[0] > t) continue;
      if (m[2] === i) enviados++;
      if (m[1] === i) recibidos++;
    }
    let mentiras = 0;
    if (ep.deception) {
      for (const s of ep.signals || []) if (s.msg === 3 && s.o === i && s.t <= t) mentiras++;
    }
    return { enviados, recibidos, mentiras };
  }

  /** Última acción ejecutada hasta `t`: {accion, turno} o null. */
  function ultimaAccion(ep, i, t) {
    for (let k = Math.min(t, ep.steps - 1); k >= 0; k--) {
      const a = accionEn(ep, i, k);
      if (a) return { accion: a, turno: k };
    }
    return null;
  }

  function chip(etiqueta, valor, extra) {
    return '<li class="ob-chip"' + (extra || "") + '><span class="ob-chip__etiqueta">'
      + esc(etiqueta) + '</span> <span class="ob-chip__valor">' + esc(valor) + "</span></li>";
  }

  /**
   * Pinta la ficha de la civilización `i` en el turno `t` (reemplaza el contenido).
   * @param {object} contenedor elemento (o similar con innerHTML)
   * @param {object} ep episodio
   * @param {number} i índice de la civilización
   * @param {number} t turno
   * @param {object} [T] tokens de color (se usan como respaldo de las variables CSS)
   */
  function render(contenedor, ep, i, t, T) {
    const estilo = OB.estilos ? OB.estilos.deEstrategia(ep.politicas ? ep.politicas[i] : ep.policy)
      : { nombre: "Desconocida", rol: "", mundo: "", id: "" };
    const tt = acota(Math.round(t), 0, ep.steps);
    const e = estado(ep, i, tt);
    const semilla = semillaCiv(ep, i);
    const nombre = seguro(() => OB.nombres.generar(semilla), "Sistema " + (i + 1));
    const idx = (i % 8) + 1;
    const color = "var(--civ-" + idx + (T && T["civ-" + idx] ? ", " + T["civ-" + idx] : "") + ")";
    const cuenta = contar(ep, i, tt);
    const etapa = etapaDe(e.nivel);

    const svgAstro = seguro(
      () => OB.astronauta.svg(estilo, e, TAM_ASTRONAUTA),
      aviso("Astronauta no disponible (" + estilo.nombre + ")"));
    const canvas = miniatura(contenedor, estilo.mundo, semilla);
    const mundo = MUNDOS[estilo.mundo] || estilo.mundo || "desconocido";
    const medidor = seguro(
      () => OB.kardashev.medidorSvg(e.nivel, 180),
      aviso("Nivel " + num(e.nivel) + " de 3"));
    const ult = ultimaAccion(ep, i, tt);

    let chips = chip("Mundo:", mundo) + chip("Política:", estilo.nombre)
      + chip("Rol:", estilo.rol)
      + chip("Estado:", e.viva ? "viva" : "muerta",
        ' data-estado="' + (e.viva ? "viva" : "muerta") + '"');
    if (e.viva && e.refugio) chips += chip("Refugio:", "activo", ' data-estado="refugio"');
    if (ep.emision) chips += chip("Emisión:", EMISION[e.emite] || "baja");

    let html = '<article class="ob-ficha ob-panel" data-viva="' + e.viva + '" style="--c:' + color
      + '" aria-label="Ficha de la civilización ' + (i + 1) + '">'
      + '<header class="ob-ficha__cabecera"><p class="ob-ficha__civ">'
      + '<i class="ob-ficha__marca"></i>Civilización ' + (i + 1)
      + '</p><h3 class="ob-ficha__nombre">' + esc(nombre) + "</h3></header>"
      + '<div class="ob-ficha__escena">' + escaner()
      + '<div class="ob-ficha__astronauta">' + svgAstro + "</div>"
      + '<div class="ob-ficha__planeta" data-planeta>'
      + (canvas ? "" : aviso("Planeta " + mundo)) + "</div></div>"
      + '<ul class="ob-ficha__chips" aria-label="Rasgos">' + chips + "</ul>"
      + '<div class="ob-ficha__stat"><span class="ob-ficha__etiqueta">'
      + "Escala de Kardashev (ilustrativa)</span>"
      + '<div class="ob-ficha__medidor">' + medidor + '</div>'
      + '<span class="ob-ficha__valor">Etapa ' + esc(etapa.id) + ": "
      + esc(etapa.nombre) + "</span></div>"
      + '<div class="ob-ficha__stat"><span class="ob-ficha__etiqueta">Recursos</span>'
      + '<span class="ob-ficha__valor ob-ficha__valor--grande">' + num(e.recursos) + "</span>"
      + grafica(ep, i, tt, color, e.recursos) + "</div>"
      + '<div class="ob-ficha__stat"><span class="ob-ficha__etiqueta">Mensajes</span>'
      + '<span class="ob-ficha__valor">' + cuenta.enviados + " enviados, " + cuenta.recibidos
      + " recibidos</span></div>";
    if (ep.deception) {
      html += '<div class="ob-ficha__stat"><span class="ob-ficha__etiqueta">Mentiras</span>'
        + '<span class="ob-ficha__valor">' + cuenta.mentiras + " emitidas</span></div>";
    }
    html += '<p class="ob-ficha__accion"><span class="ob-ficha__etiqueta">Última acción</span> '
      + (ult
        ? '<span class="ob-ficha__valor" data-accion="' + ult.accion + '">' + VERBOS[ult.accion]
          + " (turno " + ult.turno + ")</span>"
        : '<span class="ob-ficha__valor">Ninguna todavía</span>')
      + "</p></article>";

    contenedor.innerHTML = html;
    // La miniatura es un nodo: se vuelve a insertar tras reemplazar el contenido.
    const hueco = canvas && contenedor.querySelector && contenedor.querySelector("[data-planeta]");
    if (hueco && hueco.appendChild) hueco.appendChild(canvas);
  }

  OB.tarjeta = { render, estado };
})((window.OB = window.OB || {}));

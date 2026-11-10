/* Astronauta: la piel de cada civilización. Arte original hecho con formas simples.
 * Una sola descripción de primitivas (rectángulos, círculos, polígonos, arcos y grupos) alimenta
 * el SVG y el lienzo 2D; la versión 3D usa geometrías de baja poligonación.
 * Sistema de coordenadas 2D: 140 de alto, pies centrados en (50, 132), visible de x = -14 a 114. */
(function (OB) {
  "use strict";

  const PI = Math.PI;
  const VIOLETA = "#a67cff"; // reflejo espejado de la mentira (dato de acción, no del estilo)
  const ACCIONES = ["", "crecer", "ocultarse", "observar", "transmitir", "compartir", "atacar",
    "refugiarse", "mentir"];
  const TEXTO_ACCION = {
    crecer: "creciendo", ocultarse: "oculta", observar: "observando", transmitir: "transmitiendo",
    compartir: "compartiendo", atacar: "atacando", refugiarse: "refugiándose", mentir: "mintiendo",
  };

  /* ---------- color ---------- */

  function rgb(h) {
    let s = String(h || "#888888");
    if (s[0] === "#") {
      if (s.length === 4) s = "#" + s[1] + s[1] + s[2] + s[2] + s[3] + s[3];
      const n = parseInt(s.slice(1, 7), 16);
      if (!isNaN(n)) return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    return [136, 136, 136];
  }
  function hex(c) {
    return "#" + c.map((v) => {
      const s = Math.max(0, Math.min(255, Math.round(v))).toString(16);
      return s.length < 2 ? "0" + s : s;
    }).join("");
  }
  function mezclar(a, b, t) {
    const x = rgb(a), y = rgb(b);
    return hex([x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t]);
  }
  function lum(h) {
    const c = rgb(h);
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
  }
  function desat(h, k) {
    const l = lum(h) * 255;
    return hex(rgb(h).map((v) => v + (l - v) * k));
  }

  /** Estilo utilizable: acepta un objeto de OB.estilos, un id o nada (cae en "red"). */
  function resolver(e) {
    const tabla = (OB.estilos && OB.estilos.tabla) || {};
    if (e && typeof e === "object" && e.traje) return e;
    if (typeof e === "string" && tabla[e]) return tabla[e];
    return tabla.red || { id: "red", nombre: "Neuronal", traje: "#dff6f0", acento: "#00a58e",
      visor: "#9ff4e2", mochila: "#9fcfc4", insignia: "nodos" };
  }

  /** Estado normalizado (tolera campos ausentes). */
  function normal(est) {
    est = est || {};
    return {
      viva: est.viva !== false,
      emite: est.emite === 2 ? 2 : est.emite === 1 ? 1 : 0,
      refugio: !!est.refugio,
      accion: String(est.accion || ""),
    };
  }

  /** Paleta del estilo; apagada y desaturada si la civilización está muerta. */
  function paleta(e, viva) {
    const f = viva ? (h) => h : (h) => mezclar(desat(h, 0.85), "#20242e", 0.2);
    const traje = f(e.traje), acento = f(e.acento), visor = f(e.visor), mochila = f(e.mochila);
    const oscuro = lum(traje) < 0.3;
    return {
      traje, acento, visor, mochila,
      linea: oscuro ? mezclar(traje, "#ffffff", 0.5) : mezclar(traje, "#10141f", 0.7),
      plato: oscuro ? mezclar(traje, "#000000", 0.4) : mezclar(traje, "#ffffff", 0.6),
      tinta: oscuro ? mezclar(acento, "#ffffff", 0.25) : mezclar(acento, "#10141f", 0.25),
      vidrio: "#2b3040", grieta: "#a3abbf",
    };
  }

  /* ---------- primitivas ---------- */
  // f: relleno, s: trazo, w: grosor, o: opacidad, resp: respira, pulso: fase de onda.

  const R = (x, y, w, h, rr, o) => Object.assign({ k: "r", x, y, w, h, rr }, o);
  const C = (x, y, r, o) => Object.assign({ k: "c", x, y, r }, o);
  const E = (x, y, rx, ry, o) => Object.assign({ k: "e", x, y, rx, ry }, o);
  const P = (p, z, o) => Object.assign({ k: "p", p, z }, o);
  const A = (x, y, r, a0, a1, o) => Object.assign({ k: "a", x, y, r, a0, a1 }, o);
  const G = (rot, cx, cy, i, anim) => ({ k: "g", rot, cx, cy, i, anim });

  /* ---------- insignias (coordenadas locales de -10 a 10) ---------- */
  // f: 1 / s: 1 significan "usa el color de la insignia".

  function disenoInsignia(id) {
    const L = [];
    const tr = { s: 1, w: 2 };
    switch (id) {
      case "ola":
        for (const off of [-5.5, 0, 5.5]) {
          const pts = [];
          for (let x = -9; x <= 9.01; x += 1.5) pts.push(x, off + 2 * Math.sin(((x + 9) / 18) * PI * 3));
          L.push(P(pts, false, tr));
        }
        break;
      case "hoja": {
        const arriba = [], abajo = [];
        for (let u = 0; u <= 1.001; u += 0.1) {
          const px = -7 + 14 * u, py = 8 - 16 * u, w = 4.6 * Math.sin(PI * u) * 0.7071;
          arriba.push(px + w, py + w);
          abajo.unshift(px - w, py - w);
        }
        L.push(P(arriba.concat(abajo), true, { s: 1, w: 1.8 }));
        L.push(P([-9.5, 10.5, 6, -6], false, { s: 1, w: 1.5 }));
        break;
      }
      case "cristal":
        L.push(P([-7, -3, -3, -9, 3, -9, 7, -3, 0, 10], true, { s: 1, w: 1.8 }));
        L.push(P([-7, -3, 7, -3], false, { s: 1, w: 1.5 }));
        L.push(P([-3, -9, -2, -3, 0, 10], false, { s: 1, w: 1.3 }));
        L.push(P([3, -9, 2, -3, 0, 10], false, { s: 1, w: 1.3 }));
        break;
      case "espejo":
        L.push(E(0, -2, 6.5, 8, tr));
        L.push(P([0, 6, 0, 10.5], false, { s: 1, w: 2.2 }));
        L.push(P([-3.5, 1, 1.5, -5.5], false, { s: 1, w: 1.6 }));
        L.push(P([-0.5, 3, 3.5, -1.5], false, { s: 1, w: 1.6 }));
        break;
      case "duna": {
        const pts = [];
        for (let x = -10; x <= 10.01; x += 2) {
          pts.push(x, 6 - 8 * Math.exp(-Math.pow((x + 3) / 5, 2)) - 4 * Math.exp(-Math.pow((x - 6) / 3.5, 2)));
        }
        pts.push(10, 9, -10, 9);
        L.push(P(pts, true, { f: 1 }));
        L.push(C(6, -6, 2.6, { s: 1, w: 1.6 }));
        break;
      }
      case "llama":
        L.push(P([1, -10, 5, -4, 8, 2, 5, 9, 0, 10, -5, 9, -8, 2, -5, -1, -3, -5, -1, -3], true,
          { f: 1, s: 1, w: 1 }));
        break;
      case "nube":
        L.push(C(-5, 3, 4, { f: 1 }), C(0, -1, 6, { f: 1 }), C(5, 3.5, 4.5, { f: 1 }));
        L.push(R(-5, 3, 10, 4.5, 0, { f: 1 }));
        break;
      case "nodos": {
        const n = [[-7, -5], [6, -6], [0, 1], [-6, 7], [7, 6]];
        for (const q of [0, 1, 3, 4]) L.push(P([0, 1, n[q][0], n[q][1]], false, { s: 1, w: 1.5 }));
        for (const q of n) L.push(C(q[0], q[1], 2.3, { f: 1 }));
        break;
      }
      default:
        break;
    }
    return L;
  }

  /** Primitivas de una insignia centradas en (cx, cy), escala k y color col. */
  function primsInsignia(id, cx, cy, k, col) {
    return disenoInsignia(id).map((p) => {
      const q = Object.assign({}, p);
      if (q.f) q.f = col;
      if (q.s) q.s = col;
      if (q.w) q.w *= k;
      if (q.k === "p") q.p = p.p.map((v, i) => v * k + (i % 2 ? cy : cx));
      else {
        q.x = p.x * k + cx; q.y = p.y * k + cy;
        if (q.r !== undefined) q.r *= k;
        if (q.rx !== undefined) { q.rx *= k; q.ry *= k; }
        if (q.w !== undefined && q.k === "r") { q.w = p.w * k; q.h = p.h * k; }
      }
      return q;
    });
  }

  /* ---------- escena 2D ---------- */

  // Arco superior de las ondas: de -145 a -35 grados.
  const ARCO0 = (-145 * PI) / 180, ARCO1 = (-35 * PI) / 180;

  /** Lista de primitivas del astronauta para un estilo y un estado (no depende de t). */
  function construir(e, est) {
    const c = paleta(e, est.viva), id = e.insignia || "";
    const viva = est.viva;
    const pulsa = viva && (est.accion === "compartir" || est.accion === "transmitir");
    const miente = viva && est.accion === "mentir";
    const ataca = viva && est.accion === "atacar";
    const trazo = { s: c.linea, w: 2 };
    const out = [];

    // Halo de emisión (dato: la luz que la hace detectable).
    if (viva && est.emite > 0) {
      out.push(C(50, 68, 47, { f: c.acento, o: est.emite === 2 ? 0.34 : 0.16, resp: true }));
      if (est.emite === 2) out.push(C(50, 68, 47, { s: c.acento, w: 2.2, o: 0.7 }));
    }

    const cuerpo = [];
    const idEst = String(e.id || "");
    let tip = [50, 12]; // base de las ondas de la antena
    const antes = [], despues = [];
    let octogono = false;

    // Rasgo de silueta propio de cada estilo.
    switch (idEst) {
      case "cooperador": // antena doble
        antes.push(P([44, 16, 39, 3], false, { s: c.linea, w: 2 }), P([56, 16, 61, 3], false, { s: c.linea, w: 2 }));
        despues.push(C(39, 3, 3, { f: c.acento, ...trazo, w: 1.5 }), C(61, 3, 3, { f: c.acento, ...trazo, w: 1.5 }));
        tip = [50, 5];
        break;
      case "reciproco": // antena con punta de hoja
        antes.push(P([50, 16, 50, 7], false, { s: c.linea, w: 2 }));
        despues.push(P([50, 8, 56, 3, 50, -1, 44, 3], true, { f: c.acento, ...trazo, w: 1.5 }));
        tip = [50, 2];
        break;
      case "prudente": // hombreras anchas y antena corta
        antes.push(P([50, 16, 50, 9], false, { s: c.linea, w: 2 }));
        despues.push(C(50, 7, 2.8, { f: c.acento, ...trazo, w: 1.5 }));
        tip = [50, 6];
        break;
      case "mentiroso": // capucha puntiaguda
        antes.push(P([38, 22, 50, 0, 62, 22], true, { f: c.traje, ...trazo }));
        tip = [50, 3];
        break;
      case "solo": // casco de ala ancha
        despues.push(R(25, 15, 50, 5.5, 2.7, { f: c.acento, ...trazo }));
        tip = [50, 11];
        break;
      case "atacante": // aletas laterales
        antes.push(P([32, 25, 12, 17, 18, 31, 31, 39], true, { f: c.acento, ...trazo }),
          P([68, 25, 88, 17, 82, 31, 69, 39], true, { f: c.acento, ...trazo }));
        tip = [50, 12];
        break;
      case "ignorar": // orejeras redondas, sin antena
        antes.push(C(27, 35, 6.5, { f: c.acento, ...trazo }), C(73, 35, 6.5, { f: c.acento, ...trazo }));
        tip = [50, 12];
        break;
      default: // red: casco facetado y antena de nodos
        octogono = true;
        antes.push(P([50, 14, 50, 8, 50, 8, 44, 3], false, { s: c.linea, w: 1.8 }),
          P([50, 8, 56, 3], false, { s: c.linea, w: 1.8 }));
        despues.push(C(50, 7, 2.3, { f: c.acento, ...trazo, w: 1.4 }),
          C(44, 3, 2.1, { f: c.acento, ...trazo, w: 1.4 }), C(56, 3, 2.1, { f: c.acento, ...trazo, w: 1.4 }));
        tip = [50, 1];
        break;
    }

    // Mochila, piernas, botas, torso.
    cuerpo.push(R(22, 46, 56, 48, 9, { f: c.mochila, ...trazo }));
    cuerpo.push(...antes.filter((p) => idEst === "ignorar" || idEst === "atacante"));
    cuerpo.push(R(36, 92, 13, 30, 5, { f: c.traje, ...trazo }), R(51, 92, 13, 30, 5, { f: c.traje, ...trazo }));
    cuerpo.push(R(34, 119, 15, 13, 5, { f: c.acento, ...trazo }), R(51, 119, 15, 13, 5, { f: c.acento, ...trazo }));
    cuerpo.push(R(34, 52, 32, 45, 11, { f: c.traje, ...trazo }));
    cuerpo.push(R(35, 86, 30, 5, 0, { f: c.acento }));
    cuerpo.push(C(50, 69, 11.5, { f: c.plato, s: c.linea, w: 1.5 }));
    cuerpo.push(...primsInsignia(id, 50, 69, 0.95, c.tinta));

    // Brazos: balanceo suave; levantados al compartir; brazo con lanzasondas al atacar.
    const brazoIzq = [R(22, 54, 13, 26, 6.5, { f: c.traje, ...trazo }), C(30, 81, 4.8, { f: c.acento, ...trazo, w: 1.6 })];
    const brazoDer = [R(65, 54, 13, 26, 6.5, { f: c.traje, ...trazo }), C(70, 81, 4.8, { f: c.acento, ...trazo, w: 1.6 })];
    if (ataca) {
      brazoDer.push(R(66.5, 84, 7, 11, 2, { f: "#5b6275", ...trazo }), R(65, 94, 10, 3.5, 1.5, { f: c.acento, ...trazo, w: 1.5 }),
        C(70, 100, 3.2, { f: c.acento, ...trazo, w: 1.5 }));
    }
    if (viva) {
      cuerpo.push(G(4, 30, 58, brazoIzq, { amp: 5, vel: 2, fase: 0 }));
      const rotDer = ataca ? -80 : pulsa ? -32 : -4;
      cuerpo.push(G(rotDer, 70, 58, brazoDer, { amp: ataca ? 2 : 5, vel: ataca ? 3 : 2, fase: PI }));
    } else {
      cuerpo.push(G(40, 30, 58, brazoIzq, { amp: 6, vel: 0.9, fase: 0 }));
      cuerpo.push(G(-28, 70, 58, brazoDer, { amp: 6, vel: 0.7, fase: 1.5 }));
    }
    if (idEst === "prudente") {
      cuerpo.push(C(30, 57, 8.5, { f: c.acento, ...trazo }), C(70, 57, 8.5, { f: c.acento, ...trazo }));
    }

    // Cuello, antena, casco.
    cuerpo.push(R(40, 50, 20, 6, 3, { f: c.mochila, ...trazo }));
    cuerpo.push(...antes.filter((p) => !(idEst === "ignorar" || idEst === "atacante")));
    if (octogono) {
      const pts = [];
      for (let k = 0; k < 8; k++) {
        const a = ((k * 45 + 22.5) * PI) / 180;
        pts.push(50 + 21.5 * Math.cos(a), 34 + 21.5 * Math.sin(a));
      }
      cuerpo.push(P(pts, true, { f: c.traje, ...trazo }));
      cuerpo.push(P([pts[8], pts[9], pts[10], pts[11], pts[12], pts[13], 50, 34], true, { f: "#ffffff", o: 0.35 }));
      cuerpo.push(P([pts[0], pts[1], 50, 34, pts[6], pts[7]], false, { s: c.linea, w: 1.2, o: 0.5 }));
    } else {
      cuerpo.push(C(50, 34, 20, { f: c.traje, ...trazo }));
    }

    // Visor: encendido con reflejo; muerto, apagado y agrietado; mentira, espejado violeta.
    const rx = idEst === "mentiroso" ? 15.5 : 14, ry = idEst === "mentiroso" ? 11.5 : 10.5;
    if (!viva) {
      cuerpo.push(E(50, 35, rx, ry, { f: c.vidrio, ...trazo, w: 1.8 }));
      cuerpo.push(P([46, 25.5, 50, 34, 45, 40, 48, 45], false, { s: c.grieta, w: 1.4 }));
      cuerpo.push(P([50, 34, 58, 37, 60, 43], false, { s: c.grieta, w: 1.4 }));
    } else if (miente) {
      cuerpo.push(E(50, 35, rx, ry, { f: VIOLETA, ...trazo, w: 1.8 }));
      cuerpo.push(P([40, 41, 52, 28], false, { s: "#efe4ff", w: 2.4, o: 0.95 }));
      cuerpo.push(P([47, 43, 58, 31], false, { s: "#efe4ff", w: 2.4, o: 0.95 }));
    } else {
      cuerpo.push(E(50, 35, rx, ry, { f: c.visor, ...trazo, w: 1.8 }));
      cuerpo.push(P([41, 30, 47, 27.5, 49, 29, 43, 32.5], true, { f: "#ffffff", o: 0.85 }));
      if (idEst === "mentiroso") { // rasgo propio: visor espejado, también en reposo
        cuerpo.push(P([52, 41, 59, 31], false, { s: "#ffffff", w: 2, o: 0.55 }));
      }
    }
    cuerpo.push(...despues);

    // Ondas en la antena.
    if (pulsa) {
      [0, 1, 2].forEach((n) => cuerpo.push(
        A(tip[0], tip[1], 8 + n * 5, ARCO0, ARCO1, { s: c.acento, w: 2.2, o: 0.9 - n * 0.25, pulso: n })));
    }

    // Cuerpo entero: leve balanceo vivo; inclinado y a la deriva si está muerto.
    out.push(viva ? G(0, 50, 132, cuerpo, { amp: 1.5, vel: 1.6, fase: 0 })
      : G(22, 50, 92, cuerpo, { amp: 4, vel: 0.8, fase: 0 }));

    // Burbuja de refugio.
    if (est.refugio) {
      out.push(E(50, 70, 52, 66, { f: c.visor, s: c.visor, w: 2.4, o: 0.2 }));
      out.push(E(50, 70, 52, 66, { s: c.visor, w: 2.4, o: 0.85 }));
      out.push(A(50, 70, 44, (200 * PI) / 180, (250 * PI) / 180, { s: "#ffffff", w: 3, o: 0.8 }));
    }
    return out;
  }

  const cache2d = new WeakMap(); // estilo -> { clave: primitivas }

  /** Escena cacheada por estilo y estado (el estado es un conjunto pequeño y finito). */
  function escena(e, est) {
    let m = cache2d.get(e);
    if (!m) { m = {}; cache2d.set(e, m); }
    let ia = ACCIONES.indexOf(est.accion);
    if (ia < 0) ia = 0;
    const clave = (est.viva ? 1 : 0) + (est.emite << 1) + (est.refugio ? 8 : 0) + ia * 16;
    return m[clave] || (m[clave] = construir(e, est));
  }

  /* ---------- salida SVG ---------- */

  const nf = (v) => String(Math.round(v * 100) / 100);
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  function atributos(p) {
    let a = ' fill="' + (p.f || "none") + '"';
    if (p.s) a += ' stroke="' + p.s + '" stroke-width="' + nf(p.w || 1) + '"';
    if (p.o !== undefined && p.o < 1) a += ' opacity="' + nf(p.o) + '"';
    return a;
  }

  function primSvg(p) {
    switch (p.k) {
      case "r":
        return '<rect x="' + nf(p.x) + '" y="' + nf(p.y) + '" width="' + nf(p.w) + '" height="' + nf(p.h) +
          '" rx="' + nf(p.rr || 0) + '"' + atributos(p) + "/>";
      case "c":
        return '<circle cx="' + nf(p.x) + '" cy="' + nf(p.y) + '" r="' + nf(p.r) + '"' + atributos(p) + "/>";
      case "e":
        return '<ellipse cx="' + nf(p.x) + '" cy="' + nf(p.y) + '" rx="' + nf(p.rx) + '" ry="' + nf(p.ry) +
          '"' + atributos(p) + "/>";
      case "p": {
        const pts = [];
        for (let i = 0; i < p.p.length; i += 2) pts.push(nf(p.p[i]) + "," + nf(p.p[i + 1]));
        return "<" + (p.z ? "polygon" : "polyline") + ' points="' + pts.join(" ") + '"' + atributos(p) + "/>";
      }
      case "a": {
        const x0 = p.x + p.r * Math.cos(p.a0), y0 = p.y + p.r * Math.sin(p.a0);
        const x1 = p.x + p.r * Math.cos(p.a1), y1 = p.y + p.r * Math.sin(p.a1);
        return '<path d="M' + nf(x0) + " " + nf(y0) + " A" + nf(p.r) + " " + nf(p.r) + " 0 0 1 " + nf(x1) +
          " " + nf(y1) + '"' + atributos(p) + "/>";
      }
      default: { // grupo con rotación
        const dentro = p.i.map(primSvg).join("");
        return p.rot ? '<g transform="rotate(' + nf(p.rot) + " " + nf(p.cx) + " " + nf(p.cy) + ')">' + dentro + "</g>"
          : "<g>" + dentro + "</g>";
      }
    }
  }

  function titulo(e, est) {
    let s = "Astronauta de la civilización " + (e.nombre || "sin estilo") + ": ";
    s += est.viva ? "viva" : "muerta, a la deriva";
    if (est.viva && TEXTO_ACCION[est.accion]) s += ", " + TEXTO_ACCION[est.accion];
    if (est.refugio) s += ", con escudo de refugio";
    if (est.viva && est.emite) s += est.emite === 2 ? ", halo fuerte" : ", halo débil";
    if (e.insignia) s += ", insignia de " + e.insignia;
    return s;
  }

  /**
   * Astronauta de cuerpo entero y de frente como marcado SVG.
   * @param {object} estilo entrada de OB.estilos; @param {object} estado estado de la civilización;
   * @param {number} tam alto en píxeles (96 por omisión).
   * @returns {string}
   */
  function svg(estilo, estado, tam = 96) {
    const e = resolver(estilo), est = normal(estado);
    const alto = Number(tam) > 0 ? Number(tam) : 96;
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-14 0 128 140" width="' + nf((alto * 128) / 140) +
      '" height="' + nf(alto) + '" role="img" stroke-linejoin="round" stroke-linecap="round"><title>' +
      esc(titulo(e, est)) + "</title>" + escena(e, est).map(primSvg).join("") + "</svg>";
  }

  /**
   * Insignia del estilo sola (usa currentColor).
   * @param {string} id ola, hoja, cristal, espejo, duna, llama, nube o nodos; @param {number} tam lado.
   * @returns {string}
   */
  function insignia(id, tam = 24) {
    const lado = Number(tam) > 0 ? Number(tam) : 24;
    const prims = primsInsignia(String(id || ""), 12, 12, 1.1, "currentColor").map(primSvg).join("");
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="' + nf(lado) + '" height="' +
      nf(lado) + '" role="img" stroke-linejoin="round" stroke-linecap="round"><title>Insignia de ' +
      esc(id || "sin nombre") + "</title>" + prims + "</svg>";
  }

  /* ---------- salida canvas 2D ---------- */

  function rrect(ctx, x, y, w, h, r) {
    r = Math.min(r || 0, w / 2, h / 2);
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function primCanvas(ctx, p, t) {
    if (p.k === "g") {
      let rot = p.rot;
      if (p.anim) rot += p.anim.amp * Math.sin(t * p.anim.vel + p.anim.fase);
      ctx.save();
      ctx.translate(p.cx, p.cy);
      ctx.rotate((rot * PI) / 180);
      ctx.translate(-p.cx, -p.cy);
      for (let i = 0; i < p.i.length; i++) primCanvas(ctx, p.i[i], t);
      ctx.restore();
      return;
    }
    let op = p.o === undefined ? 1 : p.o, r = p.r;
    if (p.resp) op *= 0.85 + 0.15 * Math.sin(t * 2.2);
    if (p.pulso !== undefined) { // onda que se expande y se desvanece
      const ph = (t * 1.1 + p.pulso / 3) % 1;
      r = p.r * (0.55 + 0.9 * ph);
      op = 1 - ph;
    }
    ctx.globalAlpha = op;
    ctx.beginPath();
    switch (p.k) {
      case "r": rrect(ctx, p.x, p.y, p.w, p.h, p.rr); break;
      case "c": ctx.arc(p.x, p.y, r, 0, 2 * PI); break;
      case "e":
        if (ctx.ellipse) ctx.ellipse(p.x, p.y, p.rx, p.ry, 0, 0, 2 * PI);
        else { ctx.save(); ctx.translate(p.x, p.y); ctx.scale(p.rx, p.ry); ctx.arc(0, 0, 1, 0, 2 * PI); ctx.restore(); }
        break;
      case "p":
        ctx.moveTo(p.p[0], p.p[1]);
        for (let i = 2; i < p.p.length; i += 2) ctx.lineTo(p.p[i], p.p[i + 1]);
        if (p.z) ctx.closePath();
        break;
      default: ctx.arc(p.x, p.y, r, p.a0, p.a1, false); break;
    }
    if (p.f) { ctx.fillStyle = p.f; ctx.fill(); }
    if (p.s) { ctx.strokeStyle = p.s; ctx.lineWidth = p.w || 1; ctx.stroke(); }
  }

  /**
   * Dibuja el astronauta en un contexto 2D con el centro de los pies en (x, y).
   * @param {CanvasRenderingContext2D} ctx; @param {object} estilo; @param {object} estado;
   * @param {number} x @param {number} y @param {number} alto altura en píxeles;
   * @param {number} t segundos, anima el balanceo y las ondas.
   */
  function canvas(ctx, estilo, estado, x, y, alto, t) {
    const prims = escena(resolver(estilo), normal(estado));
    const s = alto / 140, tt = t || 0;
    ctx.save();
    ctx.translate(x - 50 * s, y - 132 * s);
    ctx.scale(s, s);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (let i = 0; i < prims.length; i++) primCanvas(ctx, prims[i], tt);
    ctx.restore();
  }

  /* ---------- versión 3D (three.js r128 recibido como argumento) ---------- */

  const cache3d = new WeakMap(); // THREE -> geometrías unitarias compartidas

  function geometrias(T) {
    let g = cache3d.get(T);
    if (!g) {
      g = {
        esfera: new T.SphereGeometry(1, 12, 9), cil: new T.CylinderGeometry(1, 1, 1, 10, 1),
        caja: new T.BoxGeometry(1, 1, 1), cono: new T.ConeGeometry(1, 1, 6, 1),
        anillo: new T.RingGeometry(0.9, 1, 24), octa: new T.OctahedronGeometry(1, 0),
        ico: new T.IcosahedronGeometry(1, 0), arco: new T.TorusGeometry(1, 0.2, 5, 10, PI),
        toro: new T.TorusGeometry(1, 0.2, 5, 12),
      };
      cache3d.set(T, g);
    }
    return g;
  }

  /** Insignia 3D mínima sobre el pecho (formas simples reconocibles). */
  function insignia3d(T, g, id, m) {
    const grp = new T.Group();
    const add = (geo, sx, sy, sz, x, y, rz) => {
      const o = new T.Mesh(geo, m);
      o.scale.set(sx, sy, sz);
      o.position.set(x, y, 0);
      if (rz) o.rotation.z = rz;
      grp.add(o);
    };
    switch (id) {
      case "ola": add(g.arco, 0.01, 0.01, 0.012, -0.01, -0.004, 0); add(g.arco, 0.01, 0.01, 0.012, 0.01, 0.004, PI); break;
      case "hoja": add(g.esfera, 0.011, 0.022, 0.004, 0, 0, 0.7); break;
      case "cristal": add(g.octa, 0.014, 0.022, 0.006, 0, 0, 0); break;
      case "espejo": add(g.toro, 0.014, 0.017, 0.01, 0, 0.004, 0); add(g.caja, 0.004, 0.012, 0.004, 0, -0.02, 0); break;
      case "duna": add(g.cono, 0.022, 0.016, 0.006, 0, -0.004, 0); break;
      case "llama": add(g.cono, 0.012, 0.03, 0.006, 0, 0, 0); break;
      case "nube": add(g.esfera, 0.011, 0.011, 0.006, -0.012, -0.003, 0); add(g.esfera, 0.013, 0.013, 0.006, 0, 0.003, 0);
        add(g.esfera, 0.011, 0.011, 0.006, 0.012, -0.003, 0); break;
      default: add(g.esfera, 0.006, 0.006, 0.006, -0.012, -0.008, 0); add(g.esfera, 0.006, 0.006, 0.006, 0.012, -0.008, 0);
        add(g.esfera, 0.006, 0.006, 0.006, 0, 0.012, 0); add(g.caja, 0.024, 0.002, 0.002, 0, -0.008, 0); break;
    }
    return grp;
  }

  /**
   * Astronauta de baja poligonación (~0,5 de alto, origen en los pies).
   * @param {object} THREE three.js r128; @param {object} estilo entrada de OB.estilos.
   * @returns {THREE.Group} con userData.actualizar(estado, t) y userData.partes.
   */
  function three(THREE, estilo) {
    const T = THREE, g = geometrias(T), e = resolver(estilo), id = String(e.id || "");
    const viv = paleta(e, true), mue = paleta(e, false);
    const mats = []; // [material, color vivo, color apagado]
    const std = (clave, extra) => {
      const m = new T.MeshStandardMaterial(Object.assign({ color: viv[clave], roughness: 0.6, metalness: 0.1 }, extra));
      mats.push([m, new T.Color(viv[clave]), new T.Color(mue[clave])]);
      return m;
    };
    const mTraje = std("traje"), mAcento = std("acento"), mMochila = std("mochila"), mPlato = std("plato");
    const mTinta = std("tinta", { emissive: viv.tinta, emissiveIntensity: 0.15 });
    const mVisor = new T.MeshStandardMaterial({ color: viv.visor, emissive: viv.visor, emissiveIntensity: 0.8,
      roughness: 0.2, metalness: 0.3 });
    const cVisorV = new T.Color(viv.visor), cVisorM = new T.Color(mue.vidrio), cViolet = new T.Color(VIOLETA);
    const mBlanco = new T.MeshBasicMaterial({ color: 0xffffff });
    const mGrieta = new T.MeshBasicMaterial({ color: mue.grieta });

    const mesh = (geo, mat, sx, sy, sz, x, y, z, padre) => {
      const o = new T.Mesh(geo, mat);
      o.scale.set(sx, sy, sz);
      o.position.set(x, y, z);
      (padre || cuerpo).add(o);
      return o;
    };
    const capsula = (mat, r, h, x, y, padre) => {
      const gr = new T.Group();
      mesh(g.cil, mat, r, h, r, 0, 0, 0, gr);
      mesh(g.esfera, mat, r, r, r, 0, h / 2, 0, gr);
      mesh(g.esfera, mat, r, r, r, 0, -h / 2, 0, gr);
      gr.position.set(x, y, 0);
      padre.add(gr);
      return gr;
    };

    const grupo = new T.Group(), cuerpo = new T.Group();
    grupo.add(cuerpo);

    // Mochila, piernas, botas y torso.
    mesh(g.caja, mMochila, 0.12, 0.14, 0.06, 0, 0.25, -0.07);
    capsula(mTraje, 0.028, 0.09, -0.034, 0.1, cuerpo);
    capsula(mTraje, 0.028, 0.09, 0.034, 0.1, cuerpo);
    mesh(g.caja, mAcento, 0.07, 0.045, 0.095, -0.036, 0.0225, 0.012);
    mesh(g.caja, mAcento, 0.07, 0.045, 0.095, 0.036, 0.0225, 0.012);
    mesh(g.cil, mTraje, 0.06, 0.13, 0.05, 0, 0.245, 0);
    mesh(g.esfera, mTraje, 0.06, 0.04, 0.05, 0, 0.18, 0);
    mesh(g.cil, mAcento, 0.061, 0.014, 0.051, 0, 0.2, 0);
    const placa = mesh(g.cil, mPlato, 0.028, 0.008, 0.028, 0, 0.255, 0.052);
    placa.rotation.x = PI / 2;
    const ins = insignia3d(T, g, e.insignia, mTinta);
    ins.position.set(0, 0.255, 0.058);
    cuerpo.add(ins);

    // Brazos con pivote en el hombro; el derecho lleva el lanzasondas.
    const brazoIzq = new T.Group(), brazoDer = new T.Group();
    brazoIzq.position.set(-0.082, 0.3, 0);
    brazoDer.position.set(0.082, 0.3, 0);
    [brazoIzq, brazoDer].forEach((b) => {
      capsula(mTraje, 0.022, 0.07, 0, -0.055, b);
      mesh(g.esfera, mAcento, 0.028, 0.028, 0.028, 0, -0.125, 0, b);
      cuerpo.add(b);
    });
    const lanzador = new T.Group();
    mesh(g.cil, mMochila, 0.012, 0.06, 0.012, 0, -0.17, 0, lanzador);
    mesh(g.esfera, mAcento, 0.016, 0.016, 0.016, 0, -0.21, 0, lanzador);
    lanzador.visible = false;
    brazoDer.add(lanzador);

    // Casco y visor.
    const casco = id === "red" ? mesh(g.ico, mTraje, 0.09, 0.09, 0.09, 0, 0.395, 0)
      : mesh(g.esfera, mTraje, 0.085, 0.085, 0.085, 0, 0.395, 0);
    const visor = mesh(g.esfera, mVisor, 0.058, 0.046, 0.04, 0, 0.4, 0.062);
    const reflejo = mesh(g.esfera, mBlanco, 0.012, 0.008, 0.004, -0.02, 0.416, 0.098);
    const franjas = [0.012, 0.03].map((x) => {
      const f = mesh(g.caja, mBlanco, 0.006, 0.05, 0.004, x - 0.01, 0.4, 0.1);
      f.rotation.z = 0.6;
      return f;
    });
    const grietas = [0.5, -0.6].map((r) => {
      const f = mesh(g.caja, mGrieta, 0.004, 0.05, 0.004, 0, 0.4, 0.1);
      f.rotation.z = r;
      f.visible = false;
      return f;
    });

    // Rasgo de silueta y antena.
    const antena = (x, rz, h) => {
      const a = mesh(g.cil, mMochila, 0.004, h, 0.004, x, 0.475 + h / 2, 0);
      a.rotation.z = rz;
      return a;
    };
    let punta = [0, 0.53];
    switch (id) {
      case "cooperador":
        antena(-0.03, 0.3, 0.06); antena(0.03, -0.3, 0.06);
        mesh(g.esfera, mAcento, 0.011, 0.011, 0.011, -0.048, 0.535, 0);
        mesh(g.esfera, mAcento, 0.011, 0.011, 0.011, 0.048, 0.535, 0);
        punta = [0, 0.54];
        break;
      case "reciproco": antena(0, 0, 0.04); mesh(g.octa, mAcento, 0.012, 0.022, 0.006, 0, 0.53, 0); break;
      case "prudente":
        antena(0, 0, 0.03); mesh(g.esfera, mAcento, 0.009, 0.009, 0.009, 0, 0.51, 0);
        mesh(g.esfera, mAcento, 0.036, 0.036, 0.036, -0.088, 0.305, 0);
        mesh(g.esfera, mAcento, 0.036, 0.036, 0.036, 0.088, 0.305, 0);
        break;
      case "mentiroso": mesh(g.cono, mTraje, 0.05, 0.08, 0.05, 0, 0.5, 0); punta = [0, 0.54]; break;
      case "solo": mesh(g.cil, mAcento, 0.12, 0.008, 0.12, 0, 0.445, 0); punta = [0, 0.5]; break;
      case "atacante":
        [-1, 1].forEach((s) => { const f = mesh(g.cono, mAcento, 0.03, 0.07, 0.03, s * 0.105, 0.405, 0); f.rotation.z = -s * 1.35; });
        punta = [0, 0.5];
        break;
      case "ignorar":
        mesh(g.esfera, mAcento, 0.028, 0.028, 0.028, -0.09, 0.395, 0);
        mesh(g.esfera, mAcento, 0.028, 0.028, 0.028, 0.09, 0.395, 0);
        punta = [0, 0.5];
        break;
      default:
        antena(0, 0, 0.04);
        [-0.018, 0, 0.018].forEach((x) => mesh(g.esfera, mAcento, 0.009, 0.009, 0.009, x, x === 0 ? 0.525 : 0.505, 0));
        punta = [0, 0.535];
        break;
    }

    // Ondas de la antena, halo de emisión y escudo de refugio.
    const pulsos = [0, 1, 2].map(() => {
      const m = new T.MeshBasicMaterial({ color: viv.acento, transparent: true, opacity: 0, side: T.DoubleSide });
      const o = mesh(g.anillo, m, 0.03, 0.03, 0.03, punta[0], punta[1], 0.01);
      o.visible = false;
      return o;
    });
    const mHalo = new T.MeshBasicMaterial({ color: viv.acento, transparent: true, opacity: 0.15,
      side: T.BackSide, depthWrite: false });
    const halo = mesh(g.esfera, mHalo, 0.3, 0.36, 0.3, 0, 0.25, 0, grupo);
    halo.visible = false;
    const mEscudo = new T.MeshStandardMaterial({ color: viv.visor, transparent: true, opacity: 0.18,
      roughness: 0.1, metalness: 0, depthWrite: false });
    const escudo = mesh(g.esfera, mEscudo, 0.29, 0.34, 0.29, 0, 0.26, 0, grupo);
    escudo.visible = false;

    let vivaPrev = null;

    /** Actualiza visor, escudo, halo, ondas y postura; no crea objetos. */
    grupo.userData.actualizar = function (estado, t) {
      const est = normal(estado), tt = t || 0, viva = est.viva;
      const pulsa = viva && (est.accion === "compartir" || est.accion === "transmitir");
      const miente = viva && est.accion === "mentir";
      const ataca = viva && est.accion === "atacar";
      if (viva !== vivaPrev) {
        for (let i = 0; i < mats.length; i++) {
          mats[i][0].color.copy(viva ? mats[i][1] : mats[i][2]);
        }
        vivaPrev = viva;
      }
      mVisor.color.copy(!viva ? cVisorM : miente ? cViolet : cVisorV);
      mVisor.emissive.copy(!viva ? cVisorM : miente ? cViolet : cVisorV);
      mVisor.emissiveIntensity = viva ? 0.8 + 0.25 * Math.sin(tt * 2) : 0;
      reflejo.visible = viva && !miente;
      franjas[0].visible = franjas[1].visible = miente || (viva && id === "mentiroso");
      grietas[0].visible = grietas[1].visible = !viva;
      escudo.visible = est.refugio;
      halo.visible = viva && est.emite > 0;
      mHalo.opacity = est.emite === 2 ? 0.3 : 0.14;
      halo.scale.set(0.3 + 0.01 * Math.sin(tt * 2.2), 0.36 + 0.01 * Math.sin(tt * 2.2), 0.3);
      for (let i = 0; i < 3; i++) {
        pulsos[i].visible = pulsa;
        if (pulsa) {
          const ph = (tt * 1.1 + i / 3) % 1;
          pulsos[i].scale.setScalar(0.02 + 0.07 * ph);
          pulsos[i].material.opacity = 1 - ph;
        }
      }
      lanzador.visible = ataca;
      const osc = Math.sin(tt * 2);
      if (viva) {
        cuerpo.rotation.z = 0.025 * Math.sin(tt * 1.6);
        cuerpo.position.y = 0;
        brazoIzq.rotation.z = -(0.07 + 0.08 * osc);
        brazoDer.rotation.z = ataca ? 1.35 + 0.03 * osc : pulsa ? 0.55 : 0.07 - 0.08 * osc;
      } else {
        cuerpo.rotation.z = 0.5 + 0.1 * Math.sin(tt * 0.8);
        cuerpo.position.y = 0.05 + 0.015 * Math.sin(tt * 1.1);
        brazoIzq.rotation.z = -0.7 + 0.1 * Math.sin(tt * 0.9);
        brazoDer.rotation.z = 0.55 + 0.1 * Math.sin(tt * 0.7 + 1.5);
      }
    };
    grupo.userData.partes = { visor, escudo, halo, lanzador, pulsos, cuerpo, brazoIzq, brazoDer, casco, grietas };
    grupo.userData.actualizar({ viva: true, emite: 0, refugio: false, accion: "" }, 0);
    return grupo;
  }

  OB.astronauta = { svg, insignia, canvas, three };
})((window.OB = window.OB || {}));

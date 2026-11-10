/* Bitácora del observatorio: convierte un episodio en una lista de sucesos legibles.
 * Todo sale de los datos de `ep`; los textos son una redacción, no un dato nuevo. */
(function (OB) {
  "use strict";

  const PERIODO = 25; // turnos entre resúmenes de lo repetitivo
  const TOLERANCIA = 0.05; // distancia (toro unitario) para decir que un tiro apunta a una civ
  const ORDEN = { estallido: 0, ataque: 1, mentira: 2, mensaje: 3, refugio: 4, hito: 5, muerte: 6 };
  const ETAPAS = [
    { id: "0", nombre: "planetaria incipiente" },
    { id: "I", nombre: "planetaria" },
    { id: "II", nombre: "estelar" },
    { id: "III", nombre: "galáctica" },
  ];
  const ETIQUETAS = {
    estallido: "Estallido", muerte: "Muerte", ataque: "Ataque", mensaje: "Mensaje",
    mentira: "Mentira", refugio: "Refugio", hito: "Hito",
  };

  const C = (i) => "C" + (i + 1);

  function unir(civs) {
    const nombres = civs.map(C);
    if (nombres.length <= 1) return nombres.join("");
    return nombres.slice(0, -1).join(", ") + " y " + nombres[nombres.length - 1];
  }

  /** Nivel de Kardashev: usa OB.kardashev si existe y, si no, la fórmula local. */
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

  /** Distancia en el toro unitario. */
  function distToro(ax, ay, bx, by) {
    let dx = Math.abs(ax - bx);
    let dy = Math.abs(ay - by);
    dx = Math.min(dx, 1 - dx);
    dy = Math.min(dy, 1 - dy);
    return Math.hypot(dx, dy);
  }

  /** Civilización más cercana a (x, y) dentro de la tolerancia, o -1. */
  function civEn(ep, x, y) {
    let mejor = -1;
    let dm = TOLERANCIA;
    for (let k = 0; k < (ep.n || 0); k++) {
      const p = ep.pos && ep.pos[k];
      if (!p) continue;
      const d = distToro(p[0], p[1], x, y);
      if (d <= dm) { dm = d; mejor = k; }
    }
    return mejor;
  }

  function vivaEn(ep, t, i) {
    const f = ep.alive && ep.alive[t];
    return !!f && f[i] === 1;
  }

  function enRefugio(ep, t, i) {
    const f = ep.refuge && ep.refuge[t];
    return !!f && f[i] === 1;
  }

  function estallidos(ep, out) {
    const lim = ep.alive ? ep.alive.length - 1 : ep.steps;
    for (const t of ep.bursts || []) {
      const refugiadas = [];
      const expuestas = [];
      const caen = [];
      for (let i = 0; i < ep.n; i++) {
        if (!vivaEn(ep, Math.min(t, lim), i)) continue;
        if (enRefugio(ep, Math.min(t, lim), i)) { refugiadas.push(i); continue; }
        expuestas.push(i);
        if (ep.alive && ep.alive[t + 1] && !vivaEn(ep, t + 1, i)) caen.push(i);
      }
      let txt = "Turno " + t + ": estallido de la anomalía";
      if (refugiadas.length) {
        const pl = refugiadas.length > 1;
        txt += "; " + unir(refugiadas) + (pl ? " estaban refugiadas y resisten"
          : " estaba refugiada y resiste");
      }
      for (const k of expuestas) {
        txt += "; " + C(k) + " no estaba refugiada" + (caen.includes(k) ? " y se apaga" : "");
      }
      out.push({ t, tipo: "estallido", civs: refugiadas.concat(expuestas), texto: txt + "." });
    }
  }

  function ataques(ep, out) {
    for (const p of ep.projectiles || []) {
      const blanco = civEn(ep, p.tx, p.ty);
      const destino = blanco >= 0 ? "hacia " + C(blanco) : "hacia un punto del espacio";
      out.push({
        t: p.t, tipo: "ataque", civs: blanco >= 0 ? [p.o, blanco] : [p.o],
        texto: "Turno " + p.t + ": " + C(p.o) + " lanza una sonda de ataque " + destino + ".",
      });
    }
  }

  /** Atacante probable de una muerte sin registro de causa (o -1). */
  function atacanteDe(ep, i, t) {
    let mejor = -1;
    for (const p of ep.projectiles || []) {
      if (civEn(ep, p.tx, p.ty) !== i || p.t > t) continue;
      const o = ep.pos && ep.pos[p.o];
      if (ep.v > 0 && o) {
        const llega = p.t + Math.ceil(distToro(o[0], o[1], p.tx, p.ty) / ep.v);
        if (Math.abs(llega - t) > 2) continue;
      }
      mejor = p.o;
    }
    return mejor;
  }

  function muertes(ep, out) {
    let reg = ep.muertes;
    if (!Array.isArray(reg)) {
      // Sin `ep.muertes`: se deduce del primer turno con alive = 0 y de la causa más probable.
      const rafagas = new Set(ep.bursts || []);
      reg = [];
      for (let i = 0; i < ep.n; i++) {
        for (let t = 1; ep.alive && t < ep.alive.length; t++) {
          if (ep.alive[t][i] !== 0 || ep.alive[t - 1][i] !== 1) continue;
          const atacante = atacanteDe(ep, i, t);
          let causa = "desconocida";
          if (atacante >= 0) causa = "ataque";
          else if ((rafagas.has(t - 1) || rafagas.has(t)) && !enRefugio(ep, t - 1, i)) {
            causa = "anomalia";
          }
          reg.push({ t, civ: i, causa, atacante });
          break;
        }
      }
    }
    for (const m of reg) {
      let txt;
      if (m.causa === "ataque" && m.atacante >= 0) {
        txt = C(m.civ) + " es destruida por un ataque de " + C(m.atacante);
      } else if (m.causa === "ataque") {
        txt = C(m.civ) + " es destruida por un ataque";
      } else if (m.causa === "anomalia") {
        txt = C(m.civ) + " se apaga por el estallido de la anomalía";
      } else {
        txt = C(m.civ) + " se apaga";
      }
      out.push({
        t: m.t, tipo: "muerte",
        civs: m.causa === "ataque" && m.atacante >= 0 ? [m.civ, m.atacante] : [m.civ],
        texto: "Turno " + m.t + ": " + txt + ".",
      });
    }
  }

  /** Mensajes: el primero de cada pareja y un resumen cada PERIODO turnos. */
  function mensajes(ep, out) {
    const parejas = new Map();
    const orden = (ep.mensajes || []).slice().sort((a, b) => a[0] - b[0]);
    for (const [t, rec, emi] of orden) {
      const clave = Math.min(rec, emi) + "-" + Math.max(rec, emi);
      let p = parejas.get(clave);
      if (!p) {
        p = { a: Math.min(rec, emi), b: Math.max(rec, emi), cubos: new Map() };
        parejas.set(clave, p);
        out.push({
          t, tipo: "mensaje", civs: [emi, rec],
          texto: "Turno " + t + ": " + C(emi) + " envía su primer mensaje a " + C(rec) + ".",
        });
        continue;
      }
      const k = Math.floor(t / PERIODO);
      const c = p.cubos.get(k) || { n: 0, deA: 0, deB: 0, ultimo: t };
      c.n++; c.ultimo = t;
      if (emi === p.a) c.deA++; else c.deB++;
      p.cubos.set(k, c);
    }
    for (const p of parejas.values()) {
      for (const [k, c] of p.cubos) {
        out.push({
          t: c.ultimo, tipo: "mensaje", civs: [p.a, p.b],
          texto: "Turno " + c.ultimo + ": " + C(p.a) + " y " + C(p.b)
            + " siguen intercambiando mensajes (" + c.n + " en los turnos " + k * PERIODO
            + " a " + ((k + 1) * PERIODO - 1) + ": " + c.deA + " de " + C(p.a) + " y "
            + c.deB + " de " + C(p.b) + ").",
        });
      }
    }
  }

  /** Mentiras (señales con msg 3): la primera de cada civilización y un resumen por periodo. */
  function mentiras(ep, out) {
    const por = new Map();
    const senales = (ep.signals || []).filter((s) => s.msg === 3).sort((a, b) => a.t - b.t);
    for (const s of senales) {
      let p = por.get(s.o);
      if (!p) {
        por.set(s.o, { cubos: new Map() });
        out.push({
          t: s.t, tipo: "mentira", civs: [s.o],
          texto: "Turno " + s.t + ": " + C(s.o) + " emite una medición falsa por primera vez.",
        });
        continue;
      }
      p = por.get(s.o);
      const k = Math.floor(s.t / PERIODO);
      const c = p.cubos.get(k) || { n: 0, ultimo: s.t };
      c.n++; c.ultimo = s.t;
      p.cubos.set(k, c);
    }
    for (const [o, p] of por) {
      for (const [k, c] of p.cubos) {
        out.push({
          t: c.ultimo, tipo: "mentira", civs: [o],
          texto: "Turno " + c.ultimo + ": " + C(o) + " sigue mintiendo (" + c.n
            + " mediciones falsas más en los turnos " + k * PERIODO
            + " a " + ((k + 1) * PERIODO - 1) + ").",
        });
      }
    }
  }

  /** Refugios: solo el turno en que empiezan. */
  function refugios(ep, out) {
    if (!ep.refuge) return;
    for (let i = 0; i < ep.n; i++) {
      for (let t = 0; t < ep.refuge.length; t++) {
        if (ep.refuge[t][i] !== 1 || !vivaEn(ep, t, i)) continue;
        if (t > 0 && ep.refuge[t - 1][i] === 1) continue;
        out.push({
          t, tipo: "refugio", civs: [i], texto: "Turno " + t + ": " + C(i) + " se refugia.",
        });
      }
    }
  }

  /** Hitos de Kardashev: la primera vez que una civilización cruza el nivel 1, 2 o 3. */
  function hitos(ep, out) {
    if (!ep.res) return;
    for (let i = 0; i < ep.n; i++) {
      let previo = Math.floor(nivelDe(ep.res[0] ? ep.res[0][i] : 0));
      for (let t = 1; t < ep.res.length; t++) {
        if (!vivaEn(ep, t, i)) continue;
        const actual = Math.floor(nivelDe(ep.res[t][i]));
        for (let n = previo + 1; n <= actual; n++) {
          const e = etapaDe(n);
          out.push({
            t, tipo: "hito", civs: [i],
            texto: "Turno " + t + ": " + C(i) + " alcanza la etapa " + e.id + " de Kardashev ("
              + e.nombre + ").",
          });
        }
        previo = Math.max(previo, actual);
      }
    }
  }

  /**
   * Sucesos del episodio ordenados por turno.
   * @param {object} ep episodio
   * @returns {{t:number, tipo:string, civs:number[], texto:string}[]}
   */
  function eventos(ep) {
    const out = [];
    if (!ep) return out;
    estallidos(ep, out);
    ataques(ep, out);
    muertes(ep, out);
    mensajes(ep, out);
    mentiras(ep, out);
    refugios(ep, out);
    hitos(ep, out);
    return out
      .map((e, k) => [e, k])
      .sort((a, b) => a[0].t - b[0].t || ORDEN[a[0].tipo] - ORDEN[b[0].tipo] || a[1] - b[1])
      .map((p) => p[0]);
  }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Iconos de formas simples (cada tipo tiene la suya, no solo color).
  const FORMAS = {
    estallido: '<polygon points="8,1 9.8,6 15,5.5 11,9 13,14.5 8,11.5 3,14.5 5,9 1,5.5 6.2,6"/>',
    muerte: '<circle cx="8" cy="8" r="5.5"/><path d="M4.5 11.5 11.5 4.5"/>',
    ataque: '<path d="M2 8h9"/><path d="M8 4l4 4-4 4"/><path d="M14 3v10"/>',
    mensaje: '<path d="M2 3h12v8H7l-3 3v-3H2z"/>',
    mentira: '<path d="M8 1.5 14.5 8 8 14.5 1.5 8z"/><path d="M8 5v4"/>',
    refugio: '<path d="M2 12a6 6 0 0 1 12 0z"/><path d="M1.5 14h13"/>',
    hito: '<path d="M2 13h4V9h4V5h4"/><path d="M2 15h12"/>',
  };

  function icono(tipo) {
    const forma = FORMAS[tipo] || FORMAS.mensaje;
    return '<svg class="ob-bitacora__icono" viewBox="0 0 16 16" width="16" height="16" role="img" '
      + 'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" '
      + 'stroke-linecap="round"><title>' + esc(ETIQUETAS[tipo] || tipo) + "</title>" + forma
      + "</svg>";
  }

  const memoria = new WeakMap();

  /**
   * Pinta la bitácora hasta el turno `t` (la última entrada resaltada, desplazada al final).
   * @param {object} contenedor elemento (o similar con innerHTML y scrollTop)
   * @param {object[]} lista eventos de `eventos(ep)`
   * @param {number} t turno actual
   * @param {{maximo?: number}} [opciones] maximo: entradas visibles como mucho (por defecto 80)
   */
  function render(contenedor, lista, t, opciones) {
    const maximo = (opciones && opciones.maximo) || 80;
    const visibles = (lista || []).filter((e) => e.t <= t);
    const ver = visibles.slice(-maximo);
    const antes = memoria.get(contenedor);
    // Si nada cambió desde la última pintada, no se toca el DOM (conserva el desplazamiento).
    if (antes && antes.lista === lista && antes.n === visibles.length && antes.maximo === maximo
        && String(contenedor.innerHTML || "").indexOf("ob-bitacora") >= 0) return;
    memoria.set(contenedor, { lista, n: visibles.length, maximo });

    let html = '<div class="ob-bitacora" role="log" aria-live="off" tabindex="0" '
      + 'aria-label="Bitácora del observatorio">';
    if (!ver.length) {
      html += '<p class="ob-bitacora__vacio ob-help">Sin sucesos todavía.</p>';
    } else {
      html += '<ol class="ob-bitacora__lista">';
      ver.forEach((e, k) => {
        const ultimo = k === ver.length - 1;
        html += '<li class="ob-bitacora__item" data-tipo="' + esc(e.tipo) + '"'
          + (ultimo ? ' data-ultimo="true" aria-current="true"' : "") + ">"
          + icono(e.tipo) + '<span class="ob-bitacora__texto">' + esc(e.texto) + "</span></li>";
      });
      html += "</ol>";
    }
    html += "</div>";
    contenedor.innerHTML = html;
    const log = contenedor.firstElementChild || contenedor;
    log.scrollTop = log.scrollHeight || 1e9;
  }

  OB.bitacora = { eventos, render };
})((window.OB = window.OB || {}));

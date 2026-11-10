/* Efectos de la capa narrativa: sondas, flotas, impactos, escudos, supernova y muertes.
 * Todo se deriva de `ep` (proyectiles, refugio, estallidos, muertes); lo decorativo va marcado.
 * Determinista (mulberry32 con semilla) y sin asignaciones pesadas por fotograma. */
(function (OB) {
  "use strict";

  const TAU = Math.PI * 2;
  const NF = 14;          // fragmentos por planeta destruido (decoración)
  const DUR_ONDA = 3.5;   // turnos que dura la onda de choque
  const DUR_FRAG = 5;     // turnos que dura la dispersión de fragmentos
  const DUR_IMPACTO = 6;  // turnos que se mantiene activo un impacto
  const DUR_BURST = 4;    // turno del estallido y los 3 siguientes
  const ASCENSO = 1.5;    // turnos que tarda un planeta en apagarse del todo

  const wrapU = (v) => v - Math.floor(v);
  const easeOut = (u) => 1 - (1 - u) * (1 - u);
  const clamp01 = (u) => (u < 0 ? 0 : u > 1 ? 1 : u);

  /** Generador pseudoaleatorio determinista. */
  function mulberry32(a) {
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hash(a, b) {
    return (Math.imul(a + 1, 73856093) ^ Math.imul(b + 7, 19349663) ^ 0x9e3779b9) >>> 0;
  }

  function distToro(ax, ay, bx, by, wrap) {
    const dx = wrap(bx - ax + 0.5) - 0.5;
    const dy = wrap(by - ay + 0.5) - 0.5;
    return Math.hypot(dx, dy);
  }

  const CACHE = new WeakMap();

  /**
   * Resuelve una vez por episodio lo que depende de los datos: camino corto de cada proyectil,
   * turno de llegada, objetivo, si muere o resiste y el turno de muerte de cada civilización.
   * @param {object} ep episodio (no se muta)
   * @param {function} wrap envoltura al toro [0, 1)
   */
  function preparar(ep, wrap) {
    let c = CACHE.get(ep);
    if (c) return c;
    const pos = ep.pos || [];
    const n = pos.length;
    const v = ep.v > 0 ? ep.v : 0.05;
    const muerte = new Array(n).fill(-1);
    const causa = new Array(n).fill("");
    // Muerte deducida del primer alive == 0; ep.muertes (si existe) la sobrescribe.
    if (Array.isArray(ep.alive)) {
      for (let i = 0; i < n; i++) {
        for (let t = 0; t < ep.alive.length; t++) {
          if (ep.alive[t] && ep.alive[t][i] === 0) { muerte[i] = t; break; }
        }
      }
    }
    if (Array.isArray(ep.muertes)) {
      for (const m of ep.muertes) {
        if (m && m.civ >= 0 && m.civ < n) { muerte[m.civ] = m.t; causa[m.civ] = m.causa || ""; }
      }
    }
    const proy = [];
    for (const p of ep.projectiles || []) {
      const o = pos[p.o];
      if (!o) continue;
      // Camino más corto del toro.
      const dx = wrap(p.tx - o[0] + 0.5) - 0.5;
      const dy = wrap(p.ty - o[1] + 0.5) - 0.5;
      const dist = Math.hypot(dx, dy);
      const dur = Math.max(dist / v, 0.25);
      let tj = -1;
      let mejor = 0.04;
      for (let i = 0; i < n; i++) {
        const d = distToro(pos[i][0], pos[i][1], p.tx, p.ty, wrap);
        if (d < mejor) { mejor = d; tj = i; }
      }
      const ta = p.t + dur;
      const td = tj >= 0 ? muerte[tj] : -1;
      // res: 2 el objetivo muere en ese turno o el siguiente, 1 resiste, 0 sin planeta vivo.
      let res = 0;
      if (tj >= 0) {
        if (td >= Math.floor(ta) && td <= Math.ceil(ta) + 1 && causa[tj] !== "anomalia") res = 2;
        else if (td < 0 || td > Math.ceil(ta) + 1) res = 1;
      }
      const rnd = mulberry32(hash(p.o, p.t));
      // Decoración: tamaño y desfase de la flota, deterministas por origen y turno.
      const nNaves = 3 + Math.floor(rnd() * 3);
      const fase = rnd();
      const frag = new Float32Array(NF * 4);
      for (let k = 0; k < NF * 4; k++) frag[k] = rnd();
      proy.push({
        o: p.o, t: p.t, ox: o[0], oy: o[1], dx, dy, dist, dur, ta, tj, res, nNaves, fase, frag,
        cx: tj >= 0 ? pos[tj][0] : p.tx, cy: tj >= 0 ? pos[tj][1] : p.ty,
        cruza: o[0] + dx < 0 || o[0] + dx >= 1 || o[1] + dy < 0 || o[1] + dy >= 1,
      });
    }
    const bursts = [];
    for (const b of ep.bursts || []) bursts.push(typeof b === "object" && b ? b.t : b);
    c = { proy, muerte, causa, bursts, n };
    CACHE.set(ep, c);
    return c;
  }

  /**
   * Trayectorias resueltas de los proyectiles de un episodio (para pruebas y la bitácora).
   * @returns {Array<{o,t,dx,dy,dist,dur,ta,tj,res,nNaves,cruza}>}
   */
  function trayectorias(ep) {
    return preparar(ep, wrapU).proy;
  }

  function colorCiv(h, i) {
    const c = typeof h.civ === "function" ? h.civ(i) : null;
    if (typeof c === "string") return c;
    return (h.T && h.T["civ-" + ((i % 8) + 1)]) || "#888888";
  }

  const CR = [0, 0, 0, 0, 0, 0, 0, 0];

  /**
   * Fábrica de efectos 2D.
   * @returns {{dibujar: function(CanvasRenderingContext2D, object, number, object)}}
   */
  function crear2d() {
    const G = { P: null, ox: 0, oy: 0, xx: 0, xy: 0, yx: 0, yy: 0, sc: 1, esc: 1, rx: 0, ry: 0 };
    const Q = [0, 0];
    let ink = "#ffffff";

    function pt(x, y) {
      Q[0] = x; Q[1] = y;
      const r = G.P(Q);
      G.rx = r[0]; G.ry = r[1];
    }

    function circulo(ctx, x, y, r) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
    }

    /** Traza el camino a->b (coordenadas sin envolver) partido en cada cruce de borde. */
    function trazo(ctx, ax, ay, bx, by) {
      let n = 0;
      CR[n++] = 0;
      const dx = bx - ax;
      const dy = by - ay;
      if (dx !== 0) {
        const lo = Math.floor(Math.min(ax, bx)) + 1;
        const hi = Math.ceil(Math.max(ax, bx)) - 1;
        for (let k = lo; k <= hi && n < 4; k++) CR[n++] = (k - ax) / dx;
      }
      if (dy !== 0) {
        const lo = Math.floor(Math.min(ay, by)) + 1;
        const hi = Math.ceil(Math.max(ay, by)) - 1;
        for (let k = lo; k <= hi && n < 7; k++) CR[n++] = (k - ay) / dy;
      }
      CR[n++] = 1;
      for (let i = 1; i < n; i++) {
        const v = CR[i];
        let j = i - 1;
        while (j >= 0 && CR[j] > v) { CR[j + 1] = CR[j]; j--; }
        CR[j + 1] = v;
      }
      for (let i = 0; i < n - 1; i++) {
        const u0 = CR[i];
        const u1 = CR[i + 1];
        if (u1 - u0 < 1e-9) continue;
        const um = (u0 + u1) / 2;
        const cx = Math.floor(ax + dx * um);
        const cy = Math.floor(ay + dy * um);
        pt(ax + dx * u0 - cx, ay + dy * u0 - cy);
        ctx.moveTo(G.rx, G.ry);
        pt(ax + dx * u1 - cx, ay + dy * u1 - cy);
        ctx.lineTo(G.rx, G.ry);
      }
    }

    function triangulo(ctx, x, y, c, s, tam) {
      // Punta hacia (c, s); decoración de forma: sonda con muesca trasera.
      const nx = -s;
      const ny = c;
      ctx.beginPath();
      ctx.moveTo(x + c * tam * 1.7, y + s * tam * 1.7);
      ctx.lineTo(x - c * tam + nx * tam * 0.95, y - s * tam + ny * tam * 0.95);
      ctx.lineTo(x - c * tam * 0.35, y - s * tam * 0.35);
      ctx.lineTo(x - c * tam - nx * tam * 0.95, y - s * tam - ny * tam * 0.95);
      ctx.closePath();
    }

    function supernova(ctx, ep, D, t, T) {
      const z = ep.zone;
      if (!z || !D.bursts.length) return false;
      const col = T.burst || "#ff9b3d";
      let hay = false;
      for (let b = 0; b < D.bursts.length; b++) {
        const age = t - D.bursts[b];
        if (age < 0 || age >= DUR_BURST) continue;
        if (!hay) {
          hay = true;
          ctx.save();
          ctx.beginPath();
          ctx.rect(Math.min(G.ox, G.ox + G.xx), Math.min(G.oy, G.oy + G.yy),
            Math.abs(G.xx), Math.abs(G.yy));
          ctx.clip();
        }
        const u = clamp01(age / 3);
        const fade = 1 - age / DUR_BURST;
        const rr = z.r * (0.1 + 0.9 * easeOut(u));
        for (let ix = -1; ix <= 1; ix++) {
          for (let iy = -1; iy <= 1; iy++) {
            const cx = z.x + ix;
            const cy = z.y + iy;
            if (cx + z.r < 0 || cx - z.r > 1 || cy + z.r < 0 || cy - z.r > 1) continue;
            pt(cx, cy);
            const px = G.rx;
            const py = G.ry;
            const R = rr * G.sc;
            ctx.globalAlpha = 0.1 * fade;
            ctx.fillStyle = col;
            circulo(ctx, px, py, R);
            ctx.fill();
            // Onda circular de radio creciente hasta r.
            ctx.globalAlpha = 0.9 * fade;
            ctx.strokeStyle = col;
            ctx.lineWidth = 1.5 + G.esc * 0.012 * fade;
            circulo(ctx, px, py, R);
            ctx.stroke();
            // Rayos de la supernova (decoración de forma).
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let k = 0; k < 12; k++) {
              const a = (k / 12) * TAU + 0.3;
              const r0 = R * 0.25;
              const r1 = R * (0.55 + 0.25 * ((k * 5) % 3) / 2) * Math.min(1, 0.4 + u);
              ctx.moveTo(px + Math.cos(a) * r0, py + Math.sin(a) * r0);
              ctx.lineTo(px + Math.cos(a) * r1, py + Math.sin(a) * r1);
            }
            ctx.stroke();
            // Núcleo brillante que se apaga.
            ctx.globalAlpha = 0.95 * fade;
            ctx.fillStyle = col;
            circulo(ctx, px, py, z.r * G.sc * 0.28 * fade + 1);
            ctx.fill();
            ctx.strokeStyle = ink;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
      if (hay) ctx.restore();
      return hay;
    }

    function refugios(ctx, ep, t, n, pr, h) {
      if (!ep.refuge) return false;
      const ti = Math.max(0, Math.min(ep.refuge.length - 1, Math.floor(t)));
      const fila = ep.refuge[ti];
      if (!fila) return false;
      const T = G.T;
      let hay = false;
      for (let i = 0; i < n; i++) {
        if (fila[i] !== 1) continue;
        hay = true;
        pt(ep.pos[i][0], ep.pos[i][1]);
        const x = G.rx;
        const y = G.ry;
        const R = pr * 2.3 * (1 + 0.03 * Math.sin(t * 3 + i));
        const col = T.refuge || "#4aa3ff";
        // Cúpula translúcida, borde, reflejo y malla punteada.
        ctx.globalAlpha = 0.14;
        ctx.fillStyle = col;
        circulo(ctx, x, y, R);
        ctx.fill();
        ctx.globalAlpha = 0.8;
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.globalAlpha = 0.6;
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(x, y, R * 0.8, Math.PI * 1.1, Math.PI * 1.6);
        ctx.stroke();
        if (ctx.setLineDash) {
          ctx.setLineDash(DASH);
          ctx.globalAlpha = 0.45;
          ctx.strokeStyle = col;
          ctx.beginPath();
          ctx.ellipse ? ctx.ellipse(x, y, R, R * 0.4, 0, 0, TAU) : ctx.arc(x, y, R * 0.7, 0, TAU);
          ctx.stroke();
          ctx.setLineDash(NODASH);
        }
      }
      return hay;
    }

    function muertes(ctx, ep, D, t, pr) {
      const T = G.T;
      let hay = false;
      for (let i = 0; i < D.n; i++) {
        const td = D.muerte[i];
        if (td < 0 || t < td) continue;
        hay = true;
        const age = t - td;
        const r = clamp01(age / ASCENSO);
        pt(ep.pos[i][0], ep.pos[i][1]);
        const x = G.rx;
        const y = G.ry;
        // Planeta apagado: disco oscuro con cruz (la forma no depende del color).
        ctx.globalAlpha = 0.65 * r;
        ctx.fillStyle = T.dead || "#333333";
        circulo(ctx, x, y, pr * 1.05);
        ctx.fill();
        ctx.globalAlpha = 0.8 * r;
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - pr * 0.6, y - pr * 0.6); ctx.lineTo(x + pr * 0.6, y + pr * 0.6);
        ctx.moveTo(x + pr * 0.6, y - pr * 0.6); ctx.lineTo(x - pr * 0.6, y + pr * 0.6);
        ctx.stroke();
        if (age < 2) {
          // Destello final.
          const u = age / 2;
          ctx.globalAlpha = 0.9 * (1 - u);
          ctx.fillStyle = T.sun || "#ffd24a";
          circulo(ctx, x, y, pr * 1.4 * (1 - u) + 1);
          ctx.fill();
          ctx.strokeStyle = T.danger || "#ff4040";
          ctx.lineWidth = 2;
          circulo(ctx, x, y, pr * (1 + 2.5 * u));
          ctx.stroke();
        }
      }
      return hay;
    }

    function proyectiles(ctx, ep, D, t, pr, h) {
      const T = G.T;
      let hay = false;
      for (let k = 0; k < D.proy.length; k++) {
        const p = D.proy[k];
        if (t < p.t || t > p.ta + DUR_IMPACTO) continue;
        hay = true;
        const color = colorCiv(h, p.o);
        const wrap = h.wrap || wrapU;
        // Dirección en píxeles (la base de la proyección es afín).
        let ex = p.dx * G.xx + p.dy * G.yx;
        let ey = p.dx * G.xy + p.dy * G.yy;
        const el = Math.hypot(ex, ey) || 1;
        ex /= el; ey /= el;
        if (t < p.ta) {
          const s = clamp01((t - p.t) / p.dur);
          ctx.lineCap = "round";
          // Guía del camino corto (punteada, tenue).
          if (ctx.setLineDash) {
            ctx.setLineDash(DASH);
            ctx.globalAlpha = 0.2;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            trazo(ctx, p.ox, p.oy, p.ox + p.dx, p.oy + p.dy);
            ctx.stroke();
            ctx.setLineDash(NODASH);
          }
          // Estela que se desvanece (el tramo recorrido más reciente).
          const tl = Math.min(0.15, p.dist * s) / (p.dist || 1);
          const s0 = s - tl;
          const M = 8;
          ctx.strokeStyle = color;
          for (let m = 0; m < M; m++) {
            const a = s0 + (tl * m) / M;
            const b = s0 + (tl * (m + 1)) / M;
            const u = (m + 1) / M;
            ctx.globalAlpha = 0.08 + 0.7 * u * u;
            ctx.lineWidth = 1 + pr * 0.3 * u;
            ctx.beginPath();
            trazo(ctx, p.ox + p.dx * a, p.oy + p.dy * a, p.ox + p.dx * b, p.oy + p.dy * b);
            ctx.stroke();
          }
          const qx = wrap(p.ox + p.dx * s);
          const qy = wrap(p.oy + p.dy * s);
          pt(qx, qy);
          const x = G.rx;
          const y = G.ry;
          // Flota en V detrás de la sonda (decoración).
          const tam = pr * 0.28;
          for (let j = 0; j < p.nNaves; j++) {
            const fila = (j >> 1) + 1;
            const lado = j & 1 ? 1 : -1;
            const atras = pr * (1.5 * fila + p.fase * 0.7);
            const lat = lado * pr * 0.9 * fila;
            const fx = x - ex * atras - ey * lat;
            const fy = y - ey * atras + ex * lat;
            ctx.globalAlpha = 0.9;
            ctx.fillStyle = color;
            ctx.strokeStyle = ink;
            ctx.lineWidth = 0.8;
            triangulo(ctx, fx, fy, ex, ey, tam);
            ctx.fill();
            ctx.stroke();
          }
          // Sonda: punta de flecha grande con núcleo.
          ctx.globalAlpha = 1;
          ctx.fillStyle = color;
          ctx.strokeStyle = ink;
          ctx.lineWidth = 1.4;
          triangulo(ctx, x, y, ex, ey, pr * 0.5);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = ink;
          circulo(ctx, x, y, Math.max(1, pr * 0.14));
          ctx.fill();
        } else {
          impacto(ctx, p, t - p.ta, pr, color, ex, ey, T);
        }
      }
      return hay;
    }

    function impacto(ctx, p, age, pr, color, ex, ey, T) {
      pt(p.cx, p.cy);
      const x = G.rx;
      const y = G.ry;
      if (age < DUR_ONDA) {
        const u = age / DUR_ONDA;
        // Onda de choque (dos anillos desfasados).
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.9 * (1 - u);
        ctx.lineWidth = 0.8 + 2.5 * (1 - u);
        circulo(ctx, x, y, pr * (1.2 + 5 * easeOut(u)));
        ctx.stroke();
        const u2 = clamp01((age - 0.4) / DUR_ONDA);
        if (u2 > 0 && u2 < 1) {
          ctx.strokeStyle = T.danger || "#ff4040";
          ctx.globalAlpha = 0.8 * (1 - u2);
          ctx.lineWidth = 0.8 + 2 * (1 - u2);
          circulo(ctx, x, y, pr * (1.2 + 3.6 * easeOut(u2)));
          ctx.stroke();
        }
      }
      if (age < 1.2) {
        // Destellos: núcleo y chispas radiales.
        const u = age / 1.2;
        ctx.globalAlpha = 0.95 * (1 - u);
        ctx.fillStyle = T.sun || "#ffd24a";
        circulo(ctx, x, y, pr * 1.8 * (1 - u) + 1);
        ctx.fill();
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        for (let k = 0; k < 8; k++) {
          const a = (k / 8) * TAU + p.fase;
          const r0 = pr * 0.8;
          const r1 = pr * (1.3 + 1.7 * p.frag[k]) * (1 - u * 0.5) + r0;
          ctx.moveTo(x + Math.cos(a) * r0, y + Math.sin(a) * r0);
          ctx.lineTo(x + Math.cos(a) * r1, y + Math.sin(a) * r1);
        }
        ctx.stroke();
      }
      if (p.res === 2) {
        if (age < DUR_FRAG) {
          // Fragmentos del planeta (decoración de ángulos y tamaños).
          const u = age / DUR_FRAG;
          const e = easeOut(u);
          ctx.fillStyle = T.dead || "#333333";
          ctx.strokeStyle = color;
          ctx.lineWidth = 0.8;
          ctx.globalAlpha = 1 - u;
          for (let k = 0; k < NF; k++) {
            const a = p.frag[k * 4] * TAU;
            const d = pr * (1 + (1.5 + p.frag[k * 4 + 1] * 3.5) * e);
            const s = pr * (0.12 + p.frag[k * 4 + 2] * 0.22) * (1 - 0.5 * u);
            const fx = x + Math.cos(a) * d;
            const fy = y + Math.sin(a) * d;
            const g = a + age * (1 + p.frag[k * 4 + 3] * 3);
            const c = Math.cos(g);
            const sn = Math.sin(g);
            ctx.beginPath();
            ctx.moveTo(fx + c * s * 1.5, fy + sn * s * 1.5);
            ctx.lineTo(fx - sn * s, fy + c * s);
            ctx.lineTo(fx + sn * s * 0.4, fy - c * s * 0.9);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          }
        }
      } else if (p.res === 1 && age < 4) {
        // Escudo que resiste: anillo brillante y arco reforzado hacia el origen.
        const u = age / 4;
        const col = T.refuge || "#4aa3ff";
        ctx.globalAlpha = 0.95 * (1 - u * 0.7);
        ctx.strokeStyle = col;
        ctx.lineWidth = 3 * (1 - 0.5 * u);
        circulo(ctx, x, y, pr * 2);
        ctx.stroke();
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.2;
        circulo(ctx, x, y, pr * 2.25);
        ctx.stroke();
        const a0 = Math.atan2(-ey, -ex);
        ctx.strokeStyle = T.sun || "#ffd24a";
        ctx.lineWidth = 4 * (1 - u);
        ctx.beginPath();
        ctx.arc(x, y, pr * 2, a0 - 0.9, a0 + 0.9);
        ctx.stroke();
      }
    }

    return {
      /**
       * Dibuja los efectos del turno `t` (entero o fraccionario) sobre el contexto 2D.
       * @param {CanvasRenderingContext2D} ctx
       * @param {object} ep episodio
       * @param {number} t turno
       * @param {{P: function, wrap: function, T: object, civ: function, escala: number}} h
       */
      dibujar(ctx, ep, t, h) {
        if (!ep || !ep.pos || !h || !h.P) return;
        G.P = h.P;
        G.T = h.T || {};
        G.esc = h.escala || 400;
        ink = G.T.ink || "#ffffff";
        pt(0, 0);
        G.ox = G.rx; G.oy = G.ry;
        pt(1, 0);
        G.xx = G.rx - G.ox; G.xy = G.ry - G.oy;
        pt(0, 1);
        G.yx = G.rx - G.ox; G.yy = G.ry - G.oy;
        G.sc = Math.hypot(G.xx, G.xy) || G.esc;
        const D = preparar(ep, h.wrap || wrapU);
        const pr = G.esc * 0.018; // radio del planeta en píxeles (decoración)
        let hay = false;
        // Cada grupo guarda y restaura su propio estado; el orden fija las capas.
        hay = supernova(ctx, ep, D, t, G.T) || hay;
        hay = refugios(ctx, ep, t, D.n, pr, h) || hay;
        hay = muertes(ctx, ep, D, t, pr) || hay;
        hay = proyectiles(ctx, ep, D, t, pr, h) || hay;
        if (hay) { ctx.globalAlpha = 1; ctx.lineCap = "butt"; }
      },
    };
  }

  const DASH = [4, 4];
  const NODASH = [];

  /**
   * Fábrica de efectos 3D. Todo vive en un único THREE.Group con piscinas fijas (sin crecer).
   * @param {object} THREE three.js r128
   * @param {object} escena THREE.Scene (o Object3D) donde se añade el grupo
   * @param {{p3: function, n3: function, T: object, civ: function, radioPlaneta?: number}} h3
   * @returns {{actualizar: function(object, number), limpiar: function()}}
   */
  function crear3d(THREE, escena, h3) {
    const T = h3.T || {};
    const grupo = new THREE.Group();
    grupo.name = "efectos";
    escena.add(grupo);

    const MAXP = 24;  // impactos y sondas simultáneos
    const NN = 5;     // naves máximas por sonda
    const TR = 10;    // puntos de estela por sonda
    const MAXC = 16;  // civilizaciones con efecto propio
    const MAXB = 4;   // estallidos simultáneos

    const V1 = new THREE.Vector3();
    const V2 = new THREE.Vector3();
    const VN = new THREE.Vector3();
    const VT = new THREE.Vector3();
    const VB = new THREE.Vector3();
    const VD = new THREE.Vector3();
    const SC = new THREE.Vector3();
    const MX = new THREE.Matrix4();
    const QA = new THREE.Quaternion();
    const EU = new THREE.Euler();
    const CO = new THREE.Color();
    const UP = new THREE.Vector3(0, 1, 0);
    const ZZ = new THREE.Vector3(0, 0, 1);

    const geos = [];
    const mats = [];
    const geo = (g) => { geos.push(g); return g; };
    const mat = (m) => { mats.push(m); return m; };

    const gSonda = geo(new THREE.OctahedronGeometry(0.6, 0));
    const gNave = geo(new THREE.ConeGeometry(0.4, 1.4, 4));
    const gEstela = geo(new THREE.SphereGeometry(0.3, 6, 4));
    const gFrag = geo(new THREE.TetrahedronGeometry(0.5, 0));
    const gAnillo = geo(new THREE.RingGeometry(0.88, 1, 40));
    const gEsfera = geo(new THREE.SphereGeometry(1, 16, 12));
    const gEsferaB = geo(new THREE.SphereGeometry(1, 10, 8));

    function instancias(g, n) {
      const m = new THREE.InstancedMesh(g, mat(new THREE.MeshBasicMaterial({ color: 0xffffff })), n);
      m.frustumCulled = false;
      m.count = 0;
      CO.set(0xffffff);
      for (let i = 0; i < n; i++) m.setColorAt(i, CO); // crea instanceColor desde el inicio
      grupo.add(m);
      return m;
    }
    const sondas = instancias(gSonda, MAXP);
    const naves = instancias(gNave, MAXP * NN);
    const estela = instancias(gEstela, MAXP * TR);
    const frags = instancias(gFrag, MAXP * 14);

    function malla(g, color, op, extra) {
      const m = new THREE.Mesh(g, mat(new THREE.MeshBasicMaterial(Object.assign({
        color, transparent: true, opacity: op, depthWrite: false, side: THREE.DoubleSide,
      }, extra || {}))));
      m.visible = false;
      m.frustumCulled = false;
      grupo.add(m);
      return m;
    }
    const cRef = T.refuge || "#4aa3ff";
    const cBurst = T.burst || "#ff9b3d";
    const cDanger = T.danger || "#ff4040";
    const cSun = T.sun || "#ffd24a";
    const cDead = T.dead || "#333333";

    const ondas = [];
    const destellos = [];
    const escudos = [];
    for (let i = 0; i < MAXP; i++) {
      ondas.push(malla(gAnillo, cDanger, 0));
      destellos.push(malla(gEsferaB, cSun, 0));
      escudos.push(malla(gAnillo, cRef, 0));
    }
    const nucleos = [];
    const cascaras = [];
    const anillosB = [];
    for (let i = 0; i < MAXB; i++) {
      nucleos.push(malla(gEsfera, cBurst, 0));
      cascaras.push(malla(gEsfera, cBurst, 0));
      anillosB.push(malla(gAnillo, cBurst, 0));
    }
    const domos = [];
    const mallas = [];
    const apagados = [];
    const brillos = [];
    for (let i = 0; i < MAXC; i++) {
      domos.push(malla(gEsfera, cRef, 0.16));
      mallas.push(malla(gEsferaB, cRef, 0.55, { wireframe: true }));
      apagados.push(malla(gEsfera, cDead, 0));
      brillos.push(malla(gAnillo, cSun, 0));
    }

    let S3 = 0;      // unidades del mundo por unidad del toro (promedio)
    let S3min = 0;   // ídem, la dirección más corta (para el radio de la zona)
    function medirEscala() {
      const a = V1.copy(h3.p3(0.25, 0.25, 0));
      const dx = a.distanceTo(h3.p3(0.26, 0.25, 0)) / 0.01;
      const dy = a.distanceTo(h3.p3(0.25, 0.26, 0)) / 0.01;
      S3 = (dx + dy) / 2 || 1;
      S3min = Math.min(dx, dy) || 1;
    }

    // Coloca un anillo plano sobre la superficie (normal VN) con radio y opacidad dados.
    function poner(m, centro, escala, op) {
      m.visible = op > 0.01;
      if (!m.visible) return;
      m.position.copy(centro);
      m.quaternion.setFromUnitVectors(ZZ, VN);
      m.scale.setScalar(Math.max(escala, 1e-4));
      m.material.opacity = op;
    }

    function ocultar(arr, desde) {
      for (let i = desde; i < arr.length; i++) arr[i].visible = false;
    }

    function colorCiv3(i) {
      const c = typeof h3.civ === "function" ? h3.civ(i) : null;
      CO.set(c !== null && c !== undefined ? c : T["civ-" + ((i % 8) + 1)] || "#888888");
      return CO;
    }

    function punto(out, ox, oy, dx, dy, s, alto) {
      return out.copy(h3.p3(wrapU(ox + dx * s), wrapU(oy + dy * s), alto));
    }

    return {
      /**
       * Actualiza los efectos al turno `t`. Reutiliza las piscinas; no crea objetos nuevos.
       * @param {object} ep episodio
       * @param {number} t turno (entero o fraccionario)
       */
      actualizar(ep, t) {
        if (!ep || !ep.pos) { grupo.visible = false; return; }
        grupo.visible = true;
        if (!S3) medirEscala();
        const D = preparar(ep, wrapU);
        const pr = typeof h3.radioPlaneta === "number" ? h3.radioPlaneta : S3 * 0.012; // decoración
        const alto = pr * 1.2;
        let nS = 0;
        let nN = 0;
        let nE = 0;
        let nF = 0;
        let slot = 0;

        for (let k = 0; k < D.proy.length && slot < MAXP; k++) {
          const p = D.proy[k];
          if (t < p.t || t > p.ta + DUR_IMPACTO) continue;
          if (t < p.ta) {
            const s = clamp01((t - p.t) / p.dur);
            const cx = wrapU(p.ox + p.dx * s);
            const cy = wrapU(p.oy + p.dy * s);
            punto(V1, p.ox, p.oy, p.dx, p.dy, s, alto);
            const e = 0.02 / (p.dist || 1);
            punto(V2, p.ox, p.oy, p.dx, p.dy, s + e, alto);
            VD.copy(V2).sub(V1);
            if (VD.lengthSq() < 1e-12) VD.set(1, 0, 0);
            VD.normalize();
            QA.setFromUnitVectors(UP, VD);
            const col = colorCiv3(p.o);
            // Sonda.
            MX.compose(V1, QA, SC.set(pr * 0.5, pr * 0.9, pr * 0.5));
            sondas.setMatrixAt(nS, MX);
            sondas.setColorAt(nS, col);
            nS++;
            // Estela que se desvanece.
            const tl = Math.min(0.15, p.dist * s) / (p.dist || 1);
            if (tl > 1e-5) {
              for (let m = 1; m <= TR; m++) {
                punto(V2, p.ox, p.oy, p.dx, p.dy, s - (tl * m) / TR, alto);
                const f = 1 - m / (TR + 1);
                MX.makeScale(1, 1, 1).setPosition(V2);
                MX.scale(SC.set(pr * 0.3 * f, pr * 0.3 * f, pr * 0.3 * f));
                estela.setMatrixAt(nE, MX);
                estela.setColorAt(nE, col);
                nE++;
              }
            }
            // Flota en V detrás de la sonda (decoración).
            VN.copy(h3.n3(cx, cy)).normalize();
            VT.crossVectors(VN, VD).normalize();
            for (let j = 0; j < p.nNaves; j++) {
              const fila = (j >> 1) + 1;
              const lado = j & 1 ? 1 : -1;
              V2.copy(V1)
                .addScaledVector(VD, -pr * (1.5 * fila + p.fase * 0.7))
                .addScaledVector(VT, lado * pr * 0.9 * fila);
              MX.compose(V2, QA, SC.set(pr * 0.22, pr * 0.22, pr * 0.22));
              naves.setMatrixAt(nN, MX);
              naves.setColorAt(nN, col);
              nN++;
            }
            ondas[slot].visible = false;
            destellos[slot].visible = false;
            escudos[slot].visible = false;
          } else {
            const age = t - p.ta;
            punto(V1, p.cx, p.cy, 0, 0, 0, alto);
            VN.copy(h3.n3(p.cx, p.cy)).normalize();
            const u = clamp01(age / DUR_ONDA);
            poner(ondas[slot], V1, pr * (1.2 + 5 * easeOut(u)), age < DUR_ONDA ? 0.85 * (1 - u) : 0);
            const uf = clamp01(age / 1.2);
            const dm = destellos[slot];
            dm.visible = age < 1.2;
            if (dm.visible) {
              dm.position.copy(V1);
              dm.scale.setScalar(pr * 1.6 * (1 - uf) + 1e-4);
              dm.material.opacity = 0.9 * (1 - uf);
            }
            if (p.res === 1 && age < 4) {
              const us = age / 4;
              poner(escudos[slot], V1, pr * 2, 0.95 * (1 - us * 0.7));
            } else {
              escudos[slot].visible = false;
            }
            if (p.res === 2 && age < DUR_FRAG) {
              // Fragmentos del planeta: velocidades y giros deterministas (decoración).
              const uu = age / DUR_FRAG;
              const ee = easeOut(uu);
              VT.set(0, 1, 0).cross(VN);
              if (VT.lengthSq() < 1e-6) VT.set(1, 0, 0).cross(VN);
              VT.normalize();
              VB.crossVectors(VN, VT);
              for (let j = 0; j < NF; j++) {
                const a = p.frag[j * 4] * TAU;
                const d = pr * (1 + (1.5 + p.frag[j * 4 + 1] * 3.5) * ee);
                V2.copy(V1)
                  .addScaledVector(VT, Math.cos(a) * d)
                  .addScaledVector(VB, Math.sin(a) * d)
                  .addScaledVector(VN, pr * 0.8 * ee * (0.3 + p.frag[j * 4 + 3]));
                const sz = pr * (0.15 + p.frag[j * 4 + 2] * 0.25) * (1 - uu);
                EU.set(a + age * 2, age * (1 + p.frag[j * 4 + 3] * 3), a);
                QA.setFromEuler(EU);
                MX.compose(V2, QA, SC.set(sz, sz, sz));
                frags.setMatrixAt(nF, MX);
                CO.set(j & 1 ? cDead : colorCiv3(p.tj >= 0 ? p.tj : p.o));
                frags.setColorAt(nF, CO);
                nF++;
              }
            }
          }
          slot++;
        }
        ocultar(ondas, slot);
        ocultar(destellos, slot);
        ocultar(escudos, slot);
        sondas.count = nS;
        naves.count = nN;
        estela.count = nE;
        frags.count = nF;
        for (const m of [sondas, naves, estela, frags]) {
          m.instanceMatrix.needsUpdate = true;
          m.instanceColor.needsUpdate = true;
        }

        // Anomalía: supernova en el turno del estallido y los 3 siguientes.
        let nb = 0;
        const z = ep.zone;
        if (z) {
          for (let b = 0; b < D.bursts.length && nb < MAXB; b++) {
            const age = t - D.bursts[b];
            if (age < 0 || age >= DUR_BURST) continue;
            punto(V1, z.x, z.y, 0, 0, 0, 0);
            VN.copy(h3.n3(z.x, z.y)).normalize();
            const fade = 1 - age / DUR_BURST;
            const Rw = z.r * S3min;
            const rr = Rw * (0.1 + 0.55 * easeOut(clamp01(age / 3)));
            const c = cascaras[nb];
            c.visible = true;
            c.position.copy(V1);
            c.scale.setScalar(rr);
            c.material.opacity = 0.03 + 0.06 * fade;
            const n = nucleos[nb];
            n.visible = true;
            n.position.copy(V1);
            n.scale.setScalar(Rw * 0.16 * fade + 1e-4);
            n.material.opacity = 0.7 * fade;
            poner(anillosB[nb], V1, rr, 0.9 * fade);
            nb++;
          }
        }
        ocultar(cascaras, nb);
        ocultar(nucleos, nb);
        ocultar(anillosB, nb);

        // Refugios y muertes por civilización.
        const nc = Math.min(D.n, MAXC);
        const ti = ep.refuge ? Math.max(0, Math.min(ep.refuge.length - 1, Math.floor(t))) : 0;
        const fila = ep.refuge ? ep.refuge[ti] : null;
        for (let i = 0; i < nc; i++) {
          const x = ep.pos[i][0];
          const y = ep.pos[i][1];
          const dm = domos[i];
          const ml = mallas[i];
          if (fila && fila[i] === 1) {
            punto(V1, x, y, 0, 0, 0, alto);
            const R = pr * 2.3 * (1 + 0.03 * Math.sin(t * 3 + i));
            dm.visible = true; ml.visible = true;
            dm.position.copy(V1); ml.position.copy(V1);
            dm.scale.setScalar(R); ml.scale.setScalar(R);
          } else {
            dm.visible = false; ml.visible = false;
          }
          const td = D.muerte[i];
          const ap = apagados[i];
          const br = brillos[i];
          if (td >= 0 && t >= td) {
            const age = t - td;
            punto(V1, x, y, 0, 0, 0, alto);
            ap.visible = true;
            ap.position.copy(V1);
            ap.scale.setScalar(pr * 1.05);
            ap.material.opacity = 0.65 * clamp01(age / ASCENSO);
            VN.copy(h3.n3(x, y)).normalize();
            const u = clamp01(age / 2);
            poner(br, V1, pr * (1 + 2.5 * u), age < 2 ? 0.9 * (1 - u) : 0);
          } else {
            ap.visible = false;
            br.visible = false;
          }
        }
        ocultar(domos, nc); ocultar(mallas, nc); ocultar(apagados, nc); ocultar(brillos, nc);
      },

      /** Quita el grupo de la escena y libera geometrías y materiales. */
      limpiar() {
        escena.remove(grupo);
        for (const g of geos) g.dispose();
        for (const m of mats) m.dispose();
        for (const m of [sondas, naves, estela, frags]) m.dispose();
        grupo.clear();
      },
    };
  }

  OB.efectos = { crear2d, crear3d, trayectorias };
})((window.OB = window.OB || {}));

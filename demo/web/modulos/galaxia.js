/* Fondo de galaxia espiral, procedural y determinista (decorativo).
 * El mapa de la simulación es un cuadrado cuyos bordes se conectan; la galaxia es solo el
 * fondo sobre el que se colocan los sistemas. No codifica ningún dato del episodio. */
(function (OB) {
  "use strict";
  const TAU = Math.PI * 2;
  const cache = {};

  function mulberry32(a) {
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function rgb(hex) {
    const n = parseInt(String(hex).replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgba(hex, a) {
    const c = rgb(hex);
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }
  function mezcla(a, b, f) {
    const x = rgb(a), y = rgb(b);
    return `rgb(${Math.round(x[0] + (y[0] - x[0]) * f)},${Math.round(x[1] + (y[1] - x[1]) * f)},${Math.round(x[2] + (y[2] - x[2]) * f)})`;
  }

  /**
   * Lienzo cuadrado con una galaxia espiral: bulbo, dos brazos, polvo y campo de estrellas.
   * Se guarda en caché por tema y tamaño.
   * @param {object} T tokens (void, ink, sun, share, symbol, line)
   * @param {number} [S=1024] lado en píxeles
   * @returns {HTMLCanvasElement}
   */
  function fondo(T, S = 1024) {
    const clave = [T.void, T.ink, T.sun, S].join("|");
    if (cache[clave]) return cache[clave];
    const c = document.createElement("canvas");
    c.width = c.height = S;
    const x = c.getContext("2d");
    const R = mulberry32(20261109);
    const oscuro = (rgb(T.void)[0] + rgb(T.void)[1] + rgb(T.void)[2]) < 380;
    x.fillStyle = T.void;
    x.fillRect(0, 0, S, S);

    // nebulosas lejanas
    const colores = [T.symbol || T.sun, T.sun, T.share || T.ink];
    for (let k = 0; k < 5; k++) {
      const nx = R() * S, ny = R() * S, nr = S * (0.12 + R() * 0.16);
      const g = x.createRadialGradient(nx, ny, 0, nx, ny, nr);
      g.addColorStop(0, rgba(colores[k % 3], oscuro ? 0.07 : 0.05));
      g.addColorStop(1, rgba(colores[k % 3], 0));
      x.fillStyle = g;
      x.fillRect(nx - nr, ny - nr, nr * 2, nr * 2);
    }
    // disco difuso y bulbo
    const cx = S / 2, cy = S / 2, Rg = S * 0.47;
    let g = x.createRadialGradient(cx, cy, 0, cx, cy, Rg);
    g.addColorStop(0, rgba(T.sun, oscuro ? 0.20 : 0.12));
    g.addColorStop(0.25, rgba(T.sun, oscuro ? 0.07 : 0.05));
    g.addColorStop(1, rgba(T.sun, 0));
    x.fillStyle = g;
    x.fillRect(0, 0, S, S);

    // brazos espirales: densidad de estrellas a lo largo de dos espirales logarítmicas
    x.globalCompositeOperation = oscuro ? "lighter" : "source-over";
    const ladrillo = (n, brazo) => {
      for (let i = 0; i < n; i++) {
        const u = Math.pow(R(), 0.8), r = u * Rg;
        const ang = brazo * Math.PI + (r / Rg) * 5.6 + (R() - 0.5) * (0.5 - 0.3 * u);
        const dr = (R() - 0.5) * S * 0.035 * (0.4 + u);
        const px = cx + Math.cos(ang) * (r + dr), py = cy + Math.sin(ang) * (r + dr) * 0.92;
        const f = 1 - u; // cerca del centro, tono cálido; fuera, azulado
        const col = mezcla(T.symbol || T.share || T.ink, T.sun, Math.min(1, f * 1.4));
        x.fillStyle = col;
        x.globalAlpha = (oscuro ? 0.12 : 0.18) + R() * (oscuro ? 0.55 : 0.4);
        const t = 0.5 + R() * R() * 1.8;
        x.fillRect(px, py, t, t);
      }
    };
    ladrillo(9000, 0); ladrillo(9000, 1);
    // polvo en los bordes de los brazos
    x.globalCompositeOperation = "source-over";
    for (let i = 0; i < 70; i++) {
      const u = 0.12 + R() * 0.85, r = u * Rg, brazo = i % 2;
      const ang = brazo * Math.PI + (r / Rg) * 5.6 + 0.18;
      const px = cx + Math.cos(ang) * r, py = cy + Math.sin(ang) * r * 0.92, pr = S * (0.02 + R() * 0.035);
      const d = x.createRadialGradient(px, py, 0, px, py, pr);
      d.addColorStop(0, rgba(T.void, 0.35)); d.addColorStop(1, rgba(T.void, 0));
      x.fillStyle = d; x.globalAlpha = 1; x.fillRect(px - pr, py - pr, pr * 2, pr * 2);
    }
    // campo de estrellas de fondo
    for (let i = 0; i < 700; i++) {
      x.globalAlpha = 0.15 + R() * 0.6;
      x.fillStyle = R() < 0.15 ? T.sun : T.ink;
      const t = R() < 0.06 ? 2 : 1;
      x.fillRect(R() * S, R() * S, t, t);
    }
    // núcleo brillante
    g = x.createRadialGradient(cx, cy, 0, cx, cy, S * 0.05);
    g.addColorStop(0, rgba(T.sun, oscuro ? 0.55 : 0.35));
    g.addColorStop(1, rgba(T.sun, 0));
    x.globalAlpha = 1; x.fillStyle = g; x.fillRect(cx - S * 0.05, cy - S * 0.05, S * 0.1, S * 0.1);
    x.globalAlpha = 1; x.globalCompositeOperation = "source-over";
    cache[clave] = c;
    return c;
  }

  /**
   * Dibuja la anomalía como una nube suave en cada copia del toro que la muestra.
   * @param {CanvasRenderingContext2D} ctx
   * @param {function} P convierte [x, y] unitario en píxeles
   * @param {object} zona {x, y, r}
   * @param {number} S lado de la arena en píxeles
   * @param {number} intensidad 0 a 1
   * @param {object} T tokens
   */
  function anomalia2d(ctx, P, zona, S, intensidad, T) {
    if (intensidad <= 0.01) return;
    ctx.save();
    for (const ox of [-1, 0, 1]) for (const oy of [-1, 0, 1]) {
      const q = P([zona.x + ox, zona.y + oy]), r = zona.r * S;
      if (q[0] < -r || q[1] < -r || q[0] > ctx.canvas.width + r || q[1] > ctx.canvas.height + r) continue;
      const g = ctx.createRadialGradient(q[0], q[1], r * 0.05, q[0], q[1], r);
      g.addColorStop(0, rgba(T.burst, 0.34 * intensidad));
      g.addColorStop(0.6, rgba(T.burst, 0.14 * intensidad));
      g.addColorStop(1, rgba(T.burst, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, TAU); ctx.fill();
    }
    ctx.restore();
  }

  /** Estrella de fondo para un sistema: halo suave del color de la civilización. */
  function halo(ctx, x, y, r, color, alfa) {
    const g = ctx.createRadialGradient(x, y, r * 0.3, x, y, r);
    g.addColorStop(0, rgba(color, alfa)); g.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
  }

  OB.galaxia = { fondo, anomalia2d, halo };
})((window.OB = window.OB || {}));

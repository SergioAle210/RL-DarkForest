/* Planetas procedurales: texturas equirrectangulares con ruido propio, dibujo 2D y malla 3D.
 * Todo el aspecto es DECORATIVO (no depende de los datos del episodio) y determinista por
 * (mundo, semilla). El ruido se evalua sobre la esfera en 3D, asi la textura cierra sin costura. */
(function (OB) {
  "use strict";

  const TAU = Math.PI * 2;
  const MUNDOS = ["oceanico", "selvatico", "helado", "nebuloso", "desertico", "volcanico",
    "gaseoso", "cristalino"];
  // Tono de la atmosfera por mundo (rgb).
  const ATM = {
    oceanico: [90, 170, 255], selvatico: [130, 230, 150], helado: [190, 225, 255],
    nebuloso: [210, 120, 255], desertico: [255, 190, 110], volcanico: [255, 100, 50],
    gaseoso: [255, 215, 160], cristalino: [120, 255, 230],
  };

  function mulberry32(a) {
    let s = a >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Objeto nulo encadenable: cualquier propiedad o llamada devuelve el mismo objeto.
  const NULO = new Proxy(function () {}, {
    get: (t, p) => (p === Symbol.toPrimitive ? () => 0 : NULO),
    apply: () => NULO,
    set: () => true,
  });

  function canvasFalso(w, h) {
    const base = {
      createImageData: (a, b) => ({ data: new Uint8ClampedArray(a * b * 4), width: a, height: b }),
      putImageData() {},
    };
    const ctx = new Proxy(base, { get: (t, p) => (p in t ? t[p] : NULO), set: () => true });
    return { width: w, height: h, esFalso: true, getContext: () => ctx };
  }

  /** Crea un canvas (DOM si existe; si no, un objeto minimo para Node). */
  function crearCanvas(w, h) {
    if (typeof document !== "undefined" && document.createElement) {
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      return c;
    }
    return canvasFalso(w, h);
  }

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  function smooth(a, b, v) {
    const t = clamp01((v - a) / (b - a));
    return t * t * (3 - 2 * t);
  }

  /** Ruido de valor 3D con semilla: valor, fbm y celdas (Voronoi). */
  function crearRuido(semilla) {
    const base = (mulberry32(semilla >>> 0)() * 4294967296) | 0;
    function hash(x, y, z, k) {
      let h = Math.imul(x, 0x27d4eb2d) ^ Math.imul(y, 0x165667b1) ^ Math.imul(z, 0x9e3779b1)
        ^ base ^ Math.imul(k + 1, 0x85ebca6b);
      h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
      h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
      h ^= h >>> 16;
      return (h >>> 0) / 4294967296;
    }
    function valor(x, y, z, k) {
      const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
      let fx = x - xi, fy = y - yi, fz = z - zi;
      fx = fx * fx * (3 - 2 * fx);
      fy = fy * fy * (3 - 2 * fy);
      fz = fz * fz * (3 - 2 * fz);
      const a = hash(xi, yi, zi, k), b = hash(xi + 1, yi, zi, k);
      const c = hash(xi, yi + 1, zi, k), d = hash(xi + 1, yi + 1, zi, k);
      const e = hash(xi, yi, zi + 1, k), f = hash(xi + 1, yi, zi + 1, k);
      const g = hash(xi, yi + 1, zi + 1, k), h = hash(xi + 1, yi + 1, zi + 1, k);
      const x1 = a + (b - a) * fx, x2 = c + (d - c) * fx;
      const x3 = e + (f - e) * fx, x4 = g + (h - g) * fx;
      const y1 = x1 + (x2 - x1) * fy, y2 = x3 + (x4 - x3) * fy;
      return y1 + (y2 - y1) * fz;
    }
    function fbm(x, y, z, oct, k) {
      let amp = 0.5, f = 1, suma = 0, tot = 0;
      for (let o = 0; o < oct; o++) {
        suma += amp * valor(x * f + 31.7 * o, y * f + 17.3 * o, z * f + 47.1 * o, k);
        tot += amp;
        amp *= 0.5;
        f *= 2;
      }
      return suma / tot;
    }
    const cel = { f1: 0, f2: 0, id: 0 };
    function celdas(x, y, z, k) {
      const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
      let f1 = 9, f2 = 9, id = 0;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const cx = xi + dx, cy = yi + dy, cz = zi + dz;
            const px = cx + hash(cx, cy, cz, k + 11) - x;
            const py = cy + hash(cx, cy, cz, k + 12) - y;
            const pz = cz + hash(cx, cy, cz, k + 13) - z;
            const d = px * px + py * py + pz * pz;
            if (d < f1) { f2 = f1; f1 = d; id = hash(cx, cy, cz, k + 14); }
            else if (d < f2) f2 = d;
          }
        }
      }
      cel.f1 = Math.sqrt(f1);
      cel.f2 = Math.sqrt(f2);
      cel.id = id;
      return cel;
    }
    return { valor, fbm, celdas };
  }

  // Utilidades de color sobre el vector de salida o = [r, g, b].
  function mezcla(o, a, b, t) {
    o[0] = a[0] + (b[0] - a[0]) * t;
    o[1] = a[1] + (b[1] - a[1]) * t;
    o[2] = a[2] + (b[2] - a[2]) * t;
  }
  function sobre(o, c, a) {
    o[0] += (c[0] - o[0]) * a;
    o[1] += (c[1] - o[1]) * a;
    o[2] += (c[2] - o[2]) * a;
  }

  const BLANCO = [250, 252, 255];

  // Una funcion por mundo: (x, y, z) sobre la esfera unitaria -> color en o.
  const PIXEL = {
    oceanico(R, o, x, y, z) {
      const h = R.fbm(x * 1.8, y * 1.8, z * 1.8, 5, 0);
      if (h < 0.53) {
        const d = clamp01((0.53 - h) / 0.28);
        mezcla(o, [44, 150, 205], [8, 38, 112], Math.pow(d, 0.7));
      } else {
        const e = clamp01((h - 0.53) / 0.22);
        if (e < 0.05) mezcla(o, [214, 200, 140], [90, 140, 70], e / 0.05);
        else mezcla(o, [70, 135, 62], [125, 108, 82], clamp01((e - 0.05) / 0.9));
      }
      const hielo = smooth(0.84, 0.92, Math.abs(y));
      if (hielo > 0) sobre(o, BLANCO, hielo);
      const nube = smooth(0.54, 0.7, R.fbm(x * 2.6 + 5, y * 4.5, z * 2.6, 5, 1));
      if (nube > 0) sobre(o, BLANCO, nube * 0.85);
    },
    selvatico(R, o, x, y, z) {
      const h = R.fbm(x * 1.6, y * 1.6, z * 1.6, 5, 0);
      mezcla(o, [22, 84, 38], [66, 136, 48], h);
      if (h > 0.66) sobre(o, [112, 98, 62], smooth(0.66, 0.78, h) * 0.8);
      const mancha = smooth(0.52, 0.62, R.fbm(x * 4, y * 4, z * 4, 4, 1));
      sobre(o, [6, 46, 20], mancha * 0.75);
      const r = Math.abs(R.fbm(x * 2.2 + 9, y * 2.2, z * 2.2, 4, 2) - 0.5);
      if (r < 0.013) sobre(o, [70, 150, 190], 1 - r / 0.013);
      const nube = smooth(0.6, 0.74, R.fbm(x * 3 + 2, y * 5, z * 3, 4, 3));
      sobre(o, BLANCO, nube * 0.6);
    },
    helado(R, o, x, y, z) {
      const a = R.fbm(x * 2, y * 2, z * 2, 4, 0);
      mezcla(o, [198, 224, 246], [242, 249, 255], a);
      sobre(o, [120, 170, 222], smooth(0.5, 0.35, R.fbm(x * 1.4, y * 1.4, z * 1.4, 3, 1)) * 0.6);
      sobre(o, BLANCO, smooth(0.66, 0.8, Math.abs(y)));
      const r = Math.abs(R.fbm(x * 3, y * 3, z * 3, 4, 2) - 0.5);
      if (r < 0.011) sobre(o, [38, 78, 150], (1 - r / 0.011) * 0.9);
      const r2 = Math.abs(R.fbm(x * 5 + 4, y * 5, z * 5, 3, 3) - 0.5);
      if (r2 < 0.006) sobre(o, [70, 110, 180], (1 - r2 / 0.006) * 0.6);
    },
    nebuloso(R, o, x, y, z) {
      const q = R.fbm(x * 1.2, y * 1.2, z * 1.2, 3, 0);
      const w = R.fbm(x * 2 + q * 3, y * 2 + q * 3, z * 2 + q * 3, 4, 1);
      const rem = 0.5 + 0.5 * Math.sin(w * 14 + q * 6);
      mezcla(o, [26, 10, 66], [150, 70, 204], rem);
      sobre(o, [242, 150, 232], smooth(0.58, 0.72, w) * 0.7);
      sobre(o, [255, 226, 255], smooth(0.76, 0.84, R.fbm(x * 5, y * 5, z * 5, 3, 2)) * 0.8);
    },
    desertico(R, o, x, y, z) {
      const w = R.fbm(x * 2, y * 2, z * 2, 3, 0);
      const d = Math.pow(0.5 + 0.5 * Math.sin(x * 9 + z * 6 + y * 3 + w * 7), 1.4);
      mezcla(o, [168, 108, 50], [238, 198, 122], d);
      const f = R.fbm(x * 12, y * 12, z * 12, 2, 1);
      o[0] += (f - 0.5) * 30; o[1] += (f - 0.5) * 24; o[2] += (f - 0.5) * 16;
      sobre(o, [92, 62, 42], smooth(0.6, 0.7, R.fbm(x * 1.5, y * 1.5, z * 1.5, 4, 2)) * 0.7);
    },
    volcanico(R, o, x, y, z) {
      const v = R.fbm(x * 2.2, y * 2.2, z * 2.2, 5, 0);
      mezcla(o, [24, 19, 22], [64, 48, 42], v);
      const r = Math.abs(R.fbm(x * 2.4 + 3, y * 2.4, z * 2.4, 5, 1) - 0.5);
      const r2 = Math.abs(R.fbm(x * 3.6, y * 3.6 + 5, z * 3.6, 4, 2) - 0.5);
      const m = Math.min(r, r2 * 1.3);
      sobre(o, [120, 22, 6], (1 - smooth(0.0, 0.1, m)) * 0.7);
      const lava = 1 - smooth(0.0, 0.04, m);
      if (lava > 0) {
        const c = [0, 0, 0];
        mezcla(c, [205, 42, 10], [255, 214, 84], lava * lava);
        sobre(o, c, Math.min(1, lava * 1.4));
      }
    },
    gaseoso(R, o, x, y, z) {
      const wy = y + 0.14 * (R.fbm(x * 3, y * 1.5, z * 3, 4, 0) - 0.5);
      const t = R.fbm(x * 2, y * 6, z * 2, 3, 1);
      const b = 0.5 + 0.5 * Math.sin(wy * 16 + 2 * t);
      mezcla(o, [204, 152, 96], [247, 228, 194], b);
      sobre(o, [140, 90, 60], smooth(0.4, 0.9, Math.sin(wy * 5 + 1)) * 0.5);
      const d2 = (x - 0.62) * (x - 0.62) + (y + 0.3) * (y + 0.3) * 4 + (z - 0.72) * (z - 0.72);
      sobre(o, [192, 82, 52], (1 - smooth(0.02, 0.06, d2)) * 0.9);
    },
    cristalino(R, o, x, y, z) {
      const c = R.celdas(x * 3.2, y * 3.2, z * 3.2, 0);
      const i = c.id;
      if (i < 0.34) mezcla(o, [34, 160, 156], [90, 210, 220], i / 0.34);
      else if (i < 0.67) mezcla(o, [90, 210, 220], [150, 140, 232], (i - 0.34) / 0.33);
      else mezcla(o, [150, 140, 232], [40, 170, 170], (i - 0.67) / 0.33);
      const br = 0.62 + 0.5 * R.valor(x * 6, y * 6, z * 6, 5) * (1 - c.f1 * 0.5);
      o[0] *= br; o[1] *= br; o[2] *= br;
      const e = c.f2 - c.f1;
      if (e < 0.07) sobre(o, [224, 255, 250], (1 - e / 0.07) * 0.9);
    },
  };

  // Textura de grietas incandescentes (emisiva del planeta apagado), sobre negro.
  function pixelGrietas(R, o, x, y, z) {
    const r = Math.abs(R.fbm(x * 3, y * 3, z * 3, 4, 7) - 0.5);
    const r2 = Math.abs(R.fbm(x * 5 + 2, y * 5, z * 5, 3, 8) - 0.5);
    const a = Math.max(1 - r / 0.02, (1 - r2 / 0.012) * 0.8, 0);
    o[0] = 255 * a; o[1] = 70 * a * a; o[2] = 12 * a * a * a;
  }

  // Cuerpo comun: recorre la textura equirrectangular y evalua la funcion sobre la esfera.
  function generar(mundo, semilla, ancho, alto, fn) {
    const R = crearRuido(semilla);
    const cv = crearCanvas(ancho, alto);
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(ancho, alto);
    const d = img.data;
    const cosL = new Float32Array(ancho), sinL = new Float32Array(ancho);
    for (let x = 0; x < ancho; x++) {
      cosL[x] = Math.cos((x / ancho) * TAU);
      sinL[x] = Math.sin((x / ancho) * TAU);
    }
    const o = [0, 0, 0];
    let k = 0;
    for (let j = 0; j < alto; j++) {
      const lat = ((j + 0.5) / alto - 0.5) * Math.PI;
      const cl = Math.cos(lat), py = Math.sin(lat);
      for (let i = 0; i < ancho; i++) {
        fn(R, o, cl * cosL[i], py, cl * sinL[i]);
        d[k++] = o[0] < 0 ? 0 : o[0] > 255 ? 255 : o[0];
        d[k++] = o[1] < 0 ? 0 : o[1] > 255 ? 255 : o[1];
        d[k++] = o[2] < 0 ? 0 : o[2] > 255 ? 255 : o[2];
        d[k++] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    cv.datos = d; // pixeles RGBA, tambien disponibles sin DOM
    return cv;
  }

  // Cache acotada por (tipo, mundo, semilla, tamano).
  const CACHE = new Map();
  const LIMITE = 48;
  function cachear(clave, crear) {
    let v = CACHE.get(clave);
    if (!v) {
      v = crear();
      if (CACHE.size >= LIMITE) CACHE.delete(CACHE.keys().next().value);
      CACHE.set(clave, v);
    }
    return v;
  }
  const normMundo = (m) => (PIXEL[m] ? m : "oceanico");
  function obtener(mundo, semilla, w, h) {
    const m = normMundo(mundo), s = semilla | 0;
    return cachear("t|" + m + "|" + s + "|" + w + "x" + h, () => generar(m, s, w, h, PIXEL[m]));
  }
  function obtenerGrietas(semilla, w, h) {
    const s = semilla | 0;
    return cachear("g|" + s + "|" + w + "x" + h, () => generar("g", s, w, h, pixelGrietas));
  }

  /**
   * Textura equirrectangular del mundo (cierra sin costura en el borde horizontal).
   * Se cachea: devuelve el mismo canvas para los mismos argumentos; tratalo como solo lectura.
   * El canvas lleva `datos` con los pixeles RGBA.
   */
  function textura(mundo, semilla, ancho = 512, alto = 256) {
    return obtener(mundo, semilla, ancho, alto);
  }

  /** Vaciado de la cache (para pruebas o liberar memoria). */
  function limpiarCache() {
    CACHE.clear();
    GRIETAS2D.clear();
  }

  /** Disco sombreado con atmosfera, `tam` pixeles de lado. Cada llamada da un canvas nuevo. */
  function miniatura(mundo, semilla, tam = 96) {
    const m = normMundo(mundo);
    const buf = cachear("m|" + m + "|" + (semilla | 0) + "|" + tam, () => {
      const tx = obtener(m, semilla, 256, 128).datos;
      const out = new Uint8ClampedArray(tam * tam * 4);
      const c = tam / 2, Rr = tam * 0.41;
      const at = ATM[m];
      const L = [-0.5, -0.5, 0.7], ln = Math.hypot(L[0], L[1], L[2]);
      for (let j = 0; j < tam; j++) {
        for (let i = 0; i < tam; i++) {
          const nx = (i + 0.5 - c) / Rr, ny = (j + 0.5 - c) / Rr;
          const r2 = nx * nx + ny * ny, p = (j * tam + i) * 4;
          if (r2 <= 1) {
            const nz = Math.sqrt(1 - r2);
            const lon = Math.atan2(nx, nz) + 0.6, lat = Math.asin(ny);
            let u = lon / TAU; u -= Math.floor(u);
            const tx0 = Math.min(255, (u * 256) | 0);
            const ty0 = Math.min(127, ((lat / Math.PI + 0.5) * 128) | 0);
            const q = (ty0 * 256 + tx0) * 4;
            const lam = Math.max(0, (nx * L[0] + ny * L[1] + nz * L[2]) / ln);
            const sh = 0.22 + 0.9 * lam;
            const rim = Math.pow(1 - nz, 2.6) * 0.9;
            for (let k = 0; k < 3; k++) out[p + k] = tx[q + k] * sh + at[k] * rim * (0.35 + lam);
            out[p + 3] = 255;
          } else {
            const r = Math.sqrt(r2), a = clamp01(1 - (r - 1) / 0.22);
            out[p] = at[0]; out[p + 1] = at[1]; out[p + 2] = at[2];
            out[p + 3] = a * a * 120;
          }
        }
      }
      return out;
    });
    const cv = crearCanvas(tam, tam);
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(tam, tam);
    img.data.set(buf);
    ctx.putImageData(img, 0, 0);
    cv.datos = img.data;
    return cv;
  }

  // Grietas 2D del planeta apagado: poligonales en coordenadas de disco unitario.
  const GRIETAS2D = new Map();
  function grietas2d(semilla) {
    const s = semilla | 0;
    let g = GRIETAS2D.get(s);
    if (g) return g;
    const rnd = mulberry32(s * 7919 + 13);
    g = [];
    for (let n = 0; n < 8; n++) {
      let a = rnd() * TAU, r = 0.15 + rnd() * 0.6;
      const pts = [[Math.cos(a) * r, Math.sin(a) * r]];
      let dir = rnd() * TAU;
      for (let k = 0; k < 5; k++) {
        dir += (rnd() - 0.5) * 1.6;
        const px = pts[k][0] + Math.cos(dir) * 0.2, py = pts[k][1] + Math.sin(dir) * 0.2;
        pts.push([px, py]);
      }
      g.push(pts);
    }
    GRIETAS2D.set(s, g);
    return g;
  }

  const rgba = (c, a) => "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + a + ")";

  /**
   * Dibuja el planeta en 2D centrado en (x, y). `t` en segundos rota el mapa.
   * Con `estado.viva === false` se oscurece, muestra grietas rojas y no tiene atmosfera.
   */
  function canvas(ctx, mundo, semilla, x, y, radio, t, estado) {
    const m = normMundo(mundo);
    const viva = !estado || estado.viva !== false;
    const tex = obtener(m, semilla, 256, 128);
    const W = radio * 4, H = radio * 2;
    const off = ((((t || 0) * 0.05 + (semilla | 0) * 0.137) % 1) + 1) % 1 * W;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radio, 0, TAU);
    ctx.clip();
    const x0 = x - radio - off;
    ctx.drawImage(tex, x0, y - radio, W, H);
    ctx.drawImage(tex, x0 + W, y - radio, W, H);
    // Sombreado de esfera: luz arriba a la izquierda y limbo oscuro.
    const lx = x - radio * 0.35, ly = y - radio * 0.35;
    const g = ctx.createRadialGradient(lx, ly, radio * 0.1, x, y, radio);
    g.addColorStop(0, "rgba(255,255,255,0.12)");
    g.addColorStop(0.55, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,12,0.75)");
    ctx.fillStyle = g;
    ctx.fillRect(x - radio, y - radio, radio * 2, radio * 2);
    if (!viva) {
      ctx.fillStyle = "rgba(0,0,0,0.62)";
      ctx.fillRect(x - radio, y - radio, radio * 2, radio * 2);
      const pulso = 0.65 + 0.3 * Math.sin((t || 0) * 2 + (semilla | 0));
      ctx.strokeStyle = "rgba(255," + ((70 + 60 * pulso) | 0) + ",30," + pulso.toFixed(2) + ")";
      ctx.lineWidth = Math.max(1, radio * 0.035);
      ctx.lineCap = "round";
      for (const pts of grietas2d(semilla)) {
        ctx.beginPath();
        ctx.moveTo(x + pts[0][0] * radio, y + pts[0][1] * radio);
        for (let k = 1; k < pts.length; k++) {
          ctx.lineTo(x + pts[k][0] * radio, y + pts[k][1] * radio);
        }
        ctx.stroke();
      }
    }
    ctx.restore();
    if (viva) {
      const at = ATM[m];
      const h = ctx.createRadialGradient(x, y, radio * 0.92, x, y, radio * 1.22);
      h.addColorStop(0, rgba(at, 0.38));
      h.addColorStop(0.4, rgba(at, 0.15));
      h.addColorStop(1, rgba(at, 0));
      ctx.fillStyle = h;
      ctx.beginPath();
      ctx.arc(x, y, radio * 1.22, 0, TAU);
      ctx.fill();
    }
  }

  /**
   * Grupo three.js: esfera con la textura, atmosfera aditiva (BackSide) y
   * `userData.actualizar(t, estado)` (rota; si no `viva` apaga y muestra grietas).
   */
  function three(THREE, mundo, semilla, radio = 1) {
    const m = normMundo(mundo);
    const grupo = new THREE.Group();
    const tex = new THREE.CanvasTexture(obtener(m, semilla, 512, 256));
    const emi = new THREE.CanvasTexture(obtenerGrietas(semilla, 256, 128));
    tex.wrapS = emi.wrapS = THREE.RepeatWrapping;
    if (THREE.sRGBEncoding) tex.encoding = THREE.sRGBEncoding;
    const mat = new THREE.MeshStandardMaterial({
      map: tex, roughness: m === "cristalino" ? 0.35 : m === "oceanico" ? 0.6 : 0.9,
      metalness: m === "cristalino" ? 0.3 : 0, emissive: new THREE.Color(0xff5a1a),
      emissiveMap: emi, emissiveIntensity: 0,
    });
    const esfera = new THREE.Mesh(new THREE.SphereGeometry(radio, 48, 32), mat);
    const at = ATM[m];
    const matAtm = new THREE.MeshBasicMaterial({
      color: new THREE.Color(at[0] / 255, at[1] / 255, at[2] / 255), transparent: true,
      opacity: 0.3, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false,
    });
    const atm = new THREE.Mesh(new THREE.SphereGeometry(radio * 1.07, 40, 28), matAtm);
    grupo.add(esfera, atm);
    let viva = true;
    grupo.userData.esfera = esfera;
    grupo.userData.atmosfera = atm;
    grupo.userData.actualizar = function (t, estado) {
      const v = !estado || estado.viva !== false;
      esfera.rotation.y = (t || 0) * 0.1 + (semilla | 0) * 0.7;
      if (v !== viva) { // cambios de estado: solo al conmutar
        viva = v;
        mat.color.setHex(v ? 0xffffff : 0x3a3a3a);
        atm.visible = v;
        if (v) mat.emissiveIntensity = 0;
      }
      if (!v) mat.emissiveIntensity = 0.7 + 0.35 * Math.sin((t || 0) * 2 + (semilla | 0));
    };
    return grupo;
  }

  OB.planeta = { textura, miniatura, canvas, three, mundos: MUNDOS, limpiarCache, crearCanvas };
})((window.OB = window.OB || {}));

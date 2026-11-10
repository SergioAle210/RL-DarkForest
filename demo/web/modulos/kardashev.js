/* Escala de Kardashev ILUSTRATIVA: los recursos del episodio son adimensionales, de modo que el
 * nivel es una metafora visual del crecimiento (no una medida de energia). El nivel es un dato
 * derivado de los recursos; los adornos (luces, paneles, puntos) son decorativos y deterministas.
 * Etapas: 0 luces dispersas, I ciudades y anillo orbital, II enjambre de Dyson, III resplandor. */
(function (OB) {
  "use strict";

  const TAU = Math.PI * 2;

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
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  function sm(a, b, v) {
    const t = clamp((v - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
  }

  const ETAPAS = [
    { id: "0", nombre: "Planetaria incipiente",
      descripcion: "Unas pocas luces: la civilización apenas aprovecha su mundo." },
    { id: "I", nombre: "Planetaria",
      descripcion: "Ciudades iluminadas y un anillo orbital con satélites." },
    { id: "II", nombre: "Estelar",
      descripcion: "Un enjambre de paneles orbita la estrella y capta su energía." },
    { id: "III", nombre: "Galáctica",
      descripcion: "Un resplandor se extiende y puntos de luz se expanden por la galaxia." },
  ];

  /** Nivel ilustrativo de 0 a 3 a partir de recursos adimensionales (log en la referencia). */
  function nivel(recursos, referencia = 12) {
    const r = Math.max(Number(recursos) || 0, 1);
    return clamp((3 * Math.log(r)) / Math.log(referencia), 0, 3);
  }

  /** Etapa alcanzada (piso del nivel): {id, nombre, descripcion}. */
  function etapa(n) {
    const v = Number(n) || 0;
    return ETAPAS[clamp(Math.floor(v + 1e-9), 0, 3)];
  }

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  /** Barra SVG de 0 a III con marcas I, II, III, valor numerico y titulo accesible. */
  function medidorSvg(n, ancho = 160) {
    const v = clamp(Number(n) || 0, 0, 3);
    const w = Math.max(80, Math.round(ancho)), h = 42, pad = 10, bw = w - 2 * pad;
    const et = etapa(v);
    const titulo = "Escala de Kardashev (ilustrativa): nivel " + v.toFixed(1) + ", etapa "
      + et.id + ", " + et.nombre;
    const marcas = ["0", "I", "II", "III"].map((rot, i) => {
      const x = (pad + (bw * i) / 3).toFixed(1);
      return '<line x1="' + x + '" y1="16" x2="' + x + '" y2="30" stroke="currentColor"'
        + ' stroke-width="1.5"/><text x="' + x + '" y="11" text-anchor="middle" font-size="10"'
        + ' fill="currentColor">' + rot + "</text>";
    }).join("");
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + w + " " + h + '" width="'
      + w + '" height="' + h + '" role="img" aria-label="' + esc(titulo) + '"><title>'
      + esc(titulo) + '</title><rect x="' + pad + '" y="19" width="' + bw + '" height="8" rx="2"'
      + ' fill="none" stroke="currentColor" stroke-opacity="0.6"/><rect x="' + pad + '" y="19"'
      + ' width="' + ((bw * v) / 3).toFixed(1) + '" height="8" rx="2" fill="currentColor"'
      + ' fill-opacity="0.8"/>' + marcas + '<text x="' + pad + '" y="40" font-size="10"'
      + ' fill="currentColor">Nivel ' + v.toFixed(1) + " (" + esc(et.id) + ")</text></svg>";
  }

  // Precalculo decorativo (semilla fija): luces nocturnas, paneles del enjambre y abanico.
  const rnd = mulberry32(20250);
  const N_LUCES = 32, N_PANELES = 220, N_ANILLOS = 5, N_PUNTOS = 36, N_SATS = 4;
  const LUCES = [];
  while (LUCES.length < N_LUCES) { // lado nocturno: mitad izquierda del disco
    const u = rnd() * 2 - 1, v = rnd() * 2 - 1;
    if (u < -0.05 && u * u + v * v < 0.85) LUCES.push({ u, v, ph: rnd() * TAU });
  }
  const TILT = [-0.5, 0.2, 0.7, -0.2, 0.45];
  const PANELES = [];
  for (let i = 0; i < N_PANELES; i++) {
    const k = i % N_ANILLOS;
    PANELES.push({
      k, ph: rnd() * TAU, sp: 0.35 / (1 + 0.35 * k), r: 1.35 + 0.2 * k,
      c: Math.cos(TILT[k]), s: Math.sin(TILT[k]),
    });
  }
  const PUNTOS = [];
  for (let i = 0; i < N_PUNTOS; i++) {
    PUNTOS.push({ ph: rnd(), ang: (rnd() * 2 - 1) * 0.7, pit: (rnd() * 2 - 1) * 0.35 });
  }

  function hexRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ""));
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgba = (c, a) => "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";

  // Contadores segun el nivel (los usan 2D y 3D por igual).
  function rampas(nv) {
    return {
      luces: Math.round(2 + 22 * Math.min(nv, 2) / 2 + 8 * clamp(nv - 2, 0, 1)),
      luzA: 0.35 + 0.45 * sm(0, 1, nv),
      anillo: sm(0.7, 1.3, nv),
      enjFrac: sm(1.8, 2.1, nv),
      enjN: Math.round(sm(1.8, 2.1, nv) * (50 + 170 * clamp(nv - 2, 0, 1))),
      enjR: 1 + 0.15 * clamp(nv - 2, 0, 1),
      glow: sm(2.4, 3, nv),
    };
  }

  /**
   * Dibuja la etapa alrededor del planeta en (x, y). `T` aporta colores (sun, ink, share);
   * si falta usa valores por defecto. Interpola de forma continua con `nivel`.
   */
  function canvas(ctx, x, y, radioPlaneta, nv, t, T) {
    const rp = radioPlaneta, n = clamp(Number(nv) || 0, 0, 3), tt = t || 0;
    const sol = (T && T.sun) || "#ffd36b", tinta = (T && (T.share || T.ink)) || "#9fdcff";
    const solC = hexRgb(sol) || [255, 211, 107];
    const q = rampas(n);
    ctx.save();
    // Luces de ciudad en el lado nocturno (izquierda).
    ctx.fillStyle = sol;
    const tam = Math.max(1, rp * 0.03);
    for (let i = 0; i < q.luces; i++) {
      const l = LUCES[i];
      ctx.globalAlpha = q.luzA * (0.7 + 0.3 * Math.sin(tt * 2 + l.ph));
      ctx.fillRect(x + l.u * rp * 0.9 - tam / 2, y + l.v * rp * 0.9 - tam / 2, tam, tam);
    }
    // Anillo orbital fino con satelites (etapa I).
    if (q.anillo > 0.01) {
      ctx.globalAlpha = 0.7 * q.anillo;
      ctx.strokeStyle = tinta;
      ctx.lineWidth = Math.max(1, rp * 0.02);
      ctx.beginPath();
      ctx.ellipse(x, y, rp * 1.55, rp * 0.45, -0.3, 0, TAU);
      ctx.stroke();
      ctx.fillStyle = tinta;
      for (let i = 0; i < N_SATS; i++) {
        const a = tt * 0.5 + (i * TAU) / N_SATS;
        const ex = Math.cos(a) * rp * 1.55, ey = Math.sin(a) * rp * 0.45;
        const c = Math.cos(-0.3), s = Math.sin(-0.3);
        ctx.globalAlpha = (Math.sin(a) < 0 ? 0.5 : 1) * q.anillo;
        ctx.fillRect(x + ex * c - ey * s - 1.5, y + ex * s + ey * c - 1.5, 3, 3);
      }
    }
    // Enjambre de Dyson: muchos paneles pequenos en varios anillos inclinados (etapa II).
    if (q.enjN > 0) {
      ctx.fillStyle = sol;
      const p = Math.max(1.2, rp * 0.035);
      for (let i = 0; i < q.enjN; i++) {
        const d = PANELES[i], a = d.ph + tt * d.sp, R = rp * d.r * q.enjR;
        const ex = Math.cos(a) * R, ey = Math.sin(a) * R * 0.38;
        ctx.globalAlpha = Math.sin(a) < 0 ? 0.45 : 0.95;
        ctx.fillRect(x + ex * d.c - ey * d.s - p / 2, y + ex * d.s + ey * d.c - p / 2, p, p);
      }
    }
    // Resplandor en abanico y puntos de luz que se expanden (etapa III).
    if (q.glow > 0.01) {
      ctx.globalCompositeOperation = "lighter";
      const Rg = rp * (2 + 1.4 * q.glow);
      const g = ctx.createRadialGradient(x, y, rp, x, y, Rg);
      g.addColorStop(0, rgba(solC, 0.22));
      g.addColorStop(0.45, rgba(solC, 0.06));
      g.addColorStop(1, rgba(solC, 0));
      ctx.fillStyle = g;
      ctx.globalAlpha = q.glow;
      ctx.beginPath();
      ctx.arc(x, y, Rg, 0, TAU);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, Rg * 0.55, 0, TAU);
      ctx.globalAlpha = q.glow * 0.4;
      ctx.fill();
      ctx.fillStyle = sol;
      for (let i = 0; i < N_PUNTOS; i++) {
        const d = PUNTOS[i], f = (d.ph + tt * 0.12) % 1;
        const r = rp * (1.4 + f * 5 * (0.5 + 0.5 * q.glow));
        const a = d.ang + d.pit * 0.5;
        ctx.globalAlpha = (1 - f) * q.glow;
        ctx.fillRect(x + Math.cos(a) * r - 1, y + Math.sin(a) * r - 1, 2.2, 2.2);
      }
    }
    ctx.restore();
  }

  // Textura de resplandor (disco con degradado) para el halo 3D.
  let halo = null;
  function texturaHalo() {
    if (halo) return halo;
    const cv = OB.planeta && OB.planeta.crearCanvas ? OB.planeta.crearCanvas(64, 64)
      : (typeof document !== "undefined" ? Object.assign(document.createElement("canvas"),
        { width: 64, height: 64 }) : { width: 64, height: 64, getContext: () => null });
    const c = cv.getContext("2d");
    if (c) {
      const g = c.createRadialGradient(32, 32, 0, 32, 32, 32);
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(0.4, "rgba(255,255,255,0.35)");
      g.addColorStop(1, "rgba(255,255,255,0)");
      c.fillStyle = g;
      c.fillRect(0, 0, 64, 64);
    }
    halo = cv;
    return cv;
  }

  /**
   * Grupo three.js con luces nocturnas, anillo con satelites, enjambre (una InstancedMesh) y
   * halo con puntos que se expanden. `userData.actualizar(nivel, t)` lo anima. Las luces
   * estan fijas en el lado -x del planeta; el integrador orienta la luz hacia +x.
   */
  function three(THREE, radio = 1) {
    const R = radio, grupo = new THREE.Group();
    const aditivo = THREE.AdditiveBlending;

    // Luces de ciudad: puntos sobre el hemisferio nocturno.
    const pl = new Float32Array(N_LUCES * 3);
    for (let i = 0; i < N_LUCES; i++) {
      const l = LUCES[i], z = Math.sqrt(Math.max(0, 1 - l.u * l.u - l.v * l.v));
      const hx = l.u, hy = l.v, hz = i % 2 ? z : -z; // delante y detras del limbo
      const m = Math.hypot(hx, hy, hz) || 1;
      pl.set([(hx / m) * R * 1.012, (hy / m) * R * 1.012, (hz / m) * R * 1.012], i * 3);
    }
    const gl = new THREE.BufferGeometry();
    gl.setAttribute("position", new THREE.BufferAttribute(pl, 3));
    const ml = new THREE.PointsMaterial({
      color: 0xffd36b, size: R * 0.05, transparent: true, opacity: 0.8, blending: aditivo,
      depthWrite: false,
    });
    const luces = new THREE.Points(gl, ml);
    luces.frustumCulled = false;
    grupo.add(luces);

    // Anillo orbital delgado y satelites (una InstancedMesh).
    const aro = new THREE.Group();
    aro.rotation.set(Math.PI / 2 - 0.3, 0.2, 0);
    const mAnillo = new THREE.MeshBasicMaterial({ color: 0x9fdcff, transparent: true, opacity: 0 });
    const anillo = new THREE.Mesh(new THREE.TorusGeometry(R * 1.55, R * 0.008, 6, 128), mAnillo);
    const mSat = new THREE.MeshBasicMaterial({ color: 0xcfeaff, transparent: true, opacity: 0 });
    const sats = new THREE.InstancedMesh(new THREE.BoxGeometry(R * 0.06, R * 0.06, R * 0.06),
      mSat, N_SATS);
    sats.frustumCulled = false;
    aro.add(anillo, sats);
    grupo.add(aro);

    // Enjambre de Dyson: una sola llamada de dibujo.
    const mEnj = new THREE.MeshBasicMaterial({ color: 0xffd36b, side: THREE.DoubleSide });
    const enjambre = new THREE.InstancedMesh(
      new THREE.BoxGeometry(R * 0.07, R * 0.006, R * 0.05), mEnj, N_PANELES);
    if (THREE.DynamicDrawUsage) enjambre.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    enjambre.frustumCulled = false;
    enjambre.count = 0;
    enjambre.visible = false;
    grupo.add(enjambre);

    // Halo y puntos de luz que se expanden (etapa III).
    const mHalo = new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(texturaHalo()), color: 0xffcf70, transparent: true,
      opacity: 0, blending: aditivo, depthWrite: false,
    });
    const sprite = new THREE.Sprite(mHalo);
    sprite.visible = false;
    grupo.add(sprite);
    const pp = new Float32Array(N_PUNTOS * 3);
    const gp = new THREE.BufferGeometry();
    gp.setAttribute("position", new THREE.BufferAttribute(pp, 3));
    const mp = new THREE.PointsMaterial({
      color: 0xffe9a8, size: R * 0.06, transparent: true, opacity: 0, blending: aditivo,
      depthWrite: false,
    });
    const puntos = new THREE.Points(gp, mp);
    puntos.frustumCulled = false;
    puntos.visible = false;
    grupo.add(puntos);

    const obj = new THREE.Object3D();

    grupo.userData.enjambre = enjambre;
    grupo.userData.actualizar = function (nv, t) {
      const n = clamp(Number(nv) || 0, 0, 3), tt = t || 0, q = rampas(n);
      // Luces
      luces.geometry.setDrawRange(0, q.luces);
      ml.opacity = q.luzA;
      // Anillo y satelites
      aro.visible = q.anillo > 0.01;
      if (aro.visible) {
        mAnillo.opacity = 0.65 * q.anillo;
        mSat.opacity = q.anillo;
        for (let i = 0; i < N_SATS; i++) {
          const a = tt * 0.5 + (i * TAU) / N_SATS;
          obj.position.set(Math.cos(a) * R * 1.55, Math.sin(a) * R * 1.55, 0);
          obj.rotation.set(0, 0, a);
          obj.updateMatrix();
          sats.setMatrixAt(i, obj.matrix);
        }
        sats.instanceMatrix.needsUpdate = true;
      }
      // Enjambre
      enjambre.count = q.enjN;
      enjambre.visible = q.enjN > 0;
      if (q.enjN > 0) {
        for (let i = 0; i < q.enjN; i++) {
          const d = PANELES[i], a = d.ph + tt * d.sp, rr = R * d.r * q.enjR;
          const x = Math.cos(a) * rr, z = Math.sin(a) * rr;
          // Inclinacion del anillo alrededor del eje X.
          obj.position.set(x, -z * d.s, z * d.c);
          obj.rotation.set(0, -a, 0);
          obj.updateMatrix();
          enjambre.setMatrixAt(i, obj.matrix);
        }
        enjambre.instanceMatrix.needsUpdate = true;
      }
      // Halo y puntos
      sprite.visible = puntos.visible = q.glow > 0.01;
      if (sprite.visible) {
        const s = R * (5 + 4 * q.glow);
        sprite.scale.set(s, s, 1);
        mHalo.opacity = 0.55 * q.glow;
        mp.opacity = q.glow;
        for (let i = 0; i < N_PUNTOS; i++) {
          const d = PUNTOS[i], f = (d.ph + tt * 0.12) % 1;
          const r = R * (1.4 + f * 5 * (0.5 + 0.5 * q.glow));
          const cp = Math.cos(d.pit);
          pp[i * 3] = Math.cos(d.ang) * cp * r;
          pp[i * 3 + 1] = Math.sin(d.pit) * r;
          pp[i * 3 + 2] = Math.sin(d.ang) * cp * r;
        }
        gp.attributes.position.needsUpdate = true;
      }
    };
    grupo.userData.actualizar(0, 0);
    return grupo;
  }

  OB.kardashev = { nivel, etapa, medidorSvg, canvas, three, etapas: ETAPAS };
})((window.OB = window.OB || {}));

// Pruebas del módulo astronauta. Uso: node demo/web/modulos/pruebas/astronauta.test.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const aqui = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(aqui, "..", "..");
globalThis.window = globalThis;
for (const f of ["modulos/estilos.js", "modulos/astronauta.js"]) {
  vm.runInThisContext(fs.readFileSync(path.join(web, f), "utf8"), { filename: f });
}
vm.runInThisContext(fs.readFileSync(path.join(web, "vendor/three.min.js"), "utf8"));
const OB = globalThis.OB;
const THREE = globalThis.THREE;
const A = OB.astronauta;
const estilos = OB.estilos.ids.map((id) => OB.estilos.tabla[id]);

const BASE = { viva: true, emite: 0, refugio: false, accion: "" };
const ESTADOS = {
  viva: BASE,
  comparte: { ...BASE, emite: 2, accion: "compartir" },
  miente: { ...BASE, accion: "mentir" },
  ataca: { ...BASE, accion: "atacar" },
  refugio: { ...BASE, refugio: true, accion: "refugiarse" },
  muerta: { ...BASE, viva: false },
};

/** Validador mínimo de XML: etiquetas balanceadas y atributos entre comillas. */
function bienFormado(s) {
  const pila = [];
  const re = /<(\/?)([a-zA-Z][\w:-]*)((?:\s+[\w:-]+="[^"]*")*)\s*(\/?)>/g;
  let fin = 0, m;
  while ((m = re.exec(s))) {
    assert.equal(s.slice(fin, m.index).includes("<"), false, "'<' suelto fuera de etiqueta");
    fin = re.lastIndex;
    if (m[1]) assert.equal(pila.pop(), m[2], "cierre inesperado " + m[2]);
    else if (!m[4]) pila.push(m[2]);
  }
  assert.equal(s.slice(fin).trim(), "");
  assert.equal(pila.length, 0, "etiquetas sin cerrar");
}

const pruebas = [];
const prueba = (nombre, fn) => pruebas.push([nombre, fn]);

prueba("svg: los ocho estilos y seis estados son XML bien formado con title", () => {
  assert.equal(estilos.length, 8);
  for (const e of estilos) {
    for (const [nombre, est] of Object.entries(ESTADOS)) {
      const s = A.svg(e, est, 96);
      assert.ok(s.startsWith("<svg"), nombre);
      bienFormado(s);
      assert.match(s, /<title>[^<]+<\/title>/);
      assert.ok(s.includes(e.nombre), "el título nombra el estilo " + e.nombre);
      assert.ok(/viva|muerta/.test(s.match(/<title>([^<]+)/)[1]), "el título dice el estado");
      assert.ok(!/NaN|undefined/.test(s), "sin NaN ni undefined: " + e.id + " " + nombre);
      assert.ok(s.includes('height="96"'));
    }
  }
});

prueba("svg: cambia entre estados y es determinista", () => {
  for (const e of estilos) {
    const v = ["viva", "muerta", "refugio", "miente", "ataca", "comparte"].map((k) => A.svg(e, ESTADOS[k]));
    assert.equal(new Set(v).size, v.length, "estados distintos en " + e.id);
    assert.equal(A.svg(e, ESTADOS.viva), v[0]);
  }
  const e = estilos[0];
  assert.ok(A.svg(e, ESTADOS.refugio).includes("<ellipse"));
  assert.ok(A.svg(e, ESTADOS.miente).includes("#a67cff"));
  assert.ok(A.svg(e, { ...BASE, emite: 2 }).length > A.svg(e, { ...BASE, emite: 1 }).length);
  assert.ok(A.svg(e, { ...BASE, emite: 1 }).length > A.svg(e, BASE).length);
});

prueba("svg: los estilos se distinguen por la silueta y no solo por el color", () => {
  const sinColor = (s) => s.replace(/#[0-9a-fA-F]{3,6}/g, "#");
  const formas = estilos.map((e) => sinColor(A.svg(e, BASE)));
  assert.equal(new Set(formas).size, 8);
});

prueba("svg: tolera estilo y estado ausentes", () => {
  bienFormado(A.svg(undefined, undefined));
  bienFormado(A.svg("mentiroso", { viva: false }, 48));
});

prueba("insignia: las ocho existen y son SVG válidos", () => {
  const vistas = new Set();
  for (const e of estilos) {
    const s = A.insignia(e.insignia, 24);
    bienFormado(s);
    assert.match(s, /<title>Insignia de /);
    assert.ok(s.includes("currentColor"));
    vistas.add(s);
  }
  assert.equal(vistas.size, 8);
  bienFormado(A.insignia("desconocida"));
});

/** Contexto 2D falso que registra las operaciones. */
function ctxFalso() {
  const llamadas = [];
  const ctx = new Proxy({}, {
    get: (o, k) => (k in o ? o[k] : (...a) => { llamadas.push(k); }),
    set: (o, k, v) => { o[k] = v; llamadas.push("set:" + k); return true; },
  });
  return { ctx, llamadas };
}

prueba("canvas: no lanza y llama a operaciones de trazado", () => {
  for (const e of estilos) {
    for (const est of Object.values(ESTADOS)) {
      const { ctx, llamadas } = ctxFalso();
      A.canvas(ctx, e, est, 100, 200, 120, 1.37);
      for (const op of ["save", "restore", "beginPath", "fill", "stroke", "arc", "lineTo", "translate", "scale"]) {
        assert.ok(llamadas.includes(op), op + " en " + e.id);
      }
      assert.equal(llamadas.filter((x) => x === "save").length, llamadas.filter((x) => x === "restore").length);
    }
  }
  const { ctx } = ctxFalso();
  A.canvas(ctx, undefined, undefined, 0, 0, 48);
});

prueba("canvas: t anima (las posiciones cambian con el tiempo)", () => {
  const rot = (t) => {
    const r = [];
    const ctx = new Proxy({}, { get: (o, k) => (k === "rotate" ? (a) => r.push(a) : () => {}), set: () => true });
    A.canvas(ctx, estilos[0], BASE, 0, 0, 100, t);
    return r.join(",");
  };
  assert.notEqual(rot(0), rot(0.7));
});

prueba("three: devuelve un Group con actualizar (three.min.js real)", () => {
  assert.ok(THREE && THREE.REVISION, "three.min.js cargado");
  for (const e of estilos) {
    const g = A.three(THREE, e);
    assert.ok(g instanceof THREE.Group);
    assert.equal(typeof g.userData.actualizar, "function");
    const caja = new THREE.Box3().setFromObject(g.userData.partes.cuerpo);
    const alto = caja.max.y - caja.min.y;
    assert.ok(alto > 0.45 && alto < 0.6, "alto " + alto + " en " + e.id);
    assert.ok(Math.abs(caja.min.y) < 0.01, "origen en los pies");
    let t = 0;
    for (const est of Object.values(ESTADOS)) g.userData.actualizar(est, (t += 0.5));
    g.userData.actualizar(undefined, undefined);
  }
});

prueba("three: actualizar cambia visor, escudo, halo, lanzador y postura", () => {
  const g = A.three(THREE, estilos[5]);
  const p = g.userData.partes;
  g.userData.actualizar(BASE, 0.3);
  const brilloVivo = p.visor.material.emissiveIntensity;
  assert.ok(brilloVivo > 0.4);
  assert.equal(p.escudo.visible, false);
  assert.equal(p.halo.visible, false);
  assert.equal(p.lanzador.visible, false);
  g.userData.actualizar({ ...BASE, refugio: true, emite: 2, accion: "atacar" }, 0.3);
  assert.equal(p.escudo.visible, true);
  assert.equal(p.halo.visible, true);
  assert.equal(p.lanzador.visible, true);
  g.userData.actualizar({ ...BASE, accion: "compartir" }, 0.3);
  assert.ok(p.pulsos.every((o) => o.visible));
  g.userData.actualizar({ ...BASE, viva: false, emite: 2 }, 0.3);
  assert.equal(p.visor.material.emissiveIntensity, 0);
  assert.equal(p.halo.visible, false);
  assert.ok(Math.abs(p.cuerpo.rotation.z) > 0.3, "inclinada si muerta");
  g.userData.actualizar(BASE, 0.3);
  const z1 = p.cuerpo.rotation.z;
  g.userData.actualizar(BASE, 1.2);
  assert.notEqual(z1, p.cuerpo.rotation.z, "balanceo con t");
  assert.ok(Math.abs(z1) < 0.1);
});

let fallos = 0;
for (const [nombre, fn] of pruebas) {
  try { fn(); console.log("ok   " + nombre); } catch (err) { fallos++; console.log("FALLO " + nombre + "\n" + err.stack); }
}
if (fallos) { console.log(fallos + " pruebas fallaron"); process.exit(1); }
console.log(pruebas.length + " pruebas pasaron");

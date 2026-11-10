// Pruebas de kardashev.js en Node (sin DOM): node kardashev.test.mjs
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import path from "node:path";

const aqui = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(aqui, "..", "..");
globalThis.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(web, "modulos", "planetas.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(web, "modulos", "kardashev.js"), "utf8"));
const K = globalThis.OB.kardashev;

// nivel: bordes, referencia y monotonia.
assert.strictEqual(K.nivel(0), 0);
assert.strictEqual(K.nivel(1), 0);
assert.strictEqual(K.nivel(-5), 0);
assert.ok(Math.abs(K.nivel(12) - 3) < 1e-9);
assert.strictEqual(K.nivel(1e9), 3);
assert.ok(Math.abs(K.nivel(Math.sqrt(12)) - 1.5) < 1e-9);
assert.ok(Math.abs(K.nivel(12 ** (1 / 3)) - 1) < 1e-9);
assert.ok(Math.abs(K.nivel(100, 100) - 3) < 1e-9);
let prev = -1;
for (let r = 0; r <= 40; r += 0.5) {
  const v = K.nivel(r);
  assert.ok(v >= prev && v >= 0 && v <= 3, "monotonia en " + r);
  prev = v;
}

// etapa: piso del nivel.
const id = (n) => K.etapa(n).id;
assert.strictEqual(id(0), "0");
assert.strictEqual(id(0.99), "0");
assert.strictEqual(id(1), "I");
assert.strictEqual(id(1.99), "I");
assert.strictEqual(id(2), "II");
assert.strictEqual(id(2.99), "II");
assert.strictEqual(id(3), "III");
assert.strictEqual(K.etapa(0).nombre, "Planetaria incipiente");
assert.strictEqual(K.etapa(1.5).nombre, "Planetaria");
assert.strictEqual(K.etapa(2.5).nombre, "Estelar");
assert.strictEqual(K.etapa(3).nombre, "Galáctica");
assert.ok(K.etapa(3).descripcion.length > 10);

// medidorSvg
const svg = K.medidorSvg(1.46, 200);
assert.ok(svg.startsWith("<svg") && svg.endsWith("</svg>"));
assert.ok(svg.includes("<title>") && svg.includes("role=\"img\""));
for (const m of [">0<", ">I<", ">II<", ">III<", "1.5"]) assert.ok(svg.includes(m), m);
const abre = (svg.match(/<(?!\/)[a-z]+/g) || []).length;
const cierra = (svg.match(/<\/[a-z]+>|\/>/g) || []).length;
assert.strictEqual(abre, cierra, "etiquetas balanceadas");
assert.ok(K.medidorSvg(-4).includes("0.0") && K.medidorSvg(9).includes("3.0"));
assert.ok(K.medidorSvg(2, 10).includes("viewBox"));

// canvas con contexto falso
function grabador() {
  const llamadas = {};
  return new Proxy({}, {
    get(t, p) {
      if (p === "llamadas") return llamadas;
      if (p in t) return t[p];
      return () => { llamadas[p] = (llamadas[p] || 0) + 1; return { addColorStop() {} }; };
    },
    set(t, p, v) { t[p] = v; return true; },
  });
}
const T = { sun: "#ffd36b", ink: "#eee", share: "#8fdcff" };
const rects = [];
for (const n of [0, 0.5, 1, 1.5, 2, 2.5, 3]) {
  const g = grabador();
  K.canvas(g, 100, 100, 30, n, 1.2, T);
  rects.push(g.llamadas.fillRect || 0);
  assert.ok(g.llamadas.save === g.llamadas.restore, "save/restore equilibrado");
  if (n === 0) assert.ok(!g.llamadas.ellipse, "sin anillo en etapa 0");
  if (n >= 1.5) assert.ok(g.llamadas.ellipse, "anillo orbital");
  if (n >= 3) assert.ok(g.llamadas.createRadialGradient, "resplandor");
}
assert.ok(rects[0] > 0 && rects[0] < 5, "etapa 0: apenas unas luces");
assert.ok(rects[4] > rects[2] + 40, "II dibuja un enjambre");
assert.ok(rects[6] > rects[4], "el enjambre crece con el nivel");
K.canvas(grabador(), 0, 0, 10, 2, 0); // sin T

// three real
vm.runInThisContext(fs.readFileSync(path.join(web, "vendor", "three.min.js"), "utf8"));
const THREE = globalThis.THREE;
const g = K.three(THREE, 1.5);
assert.ok(g instanceof THREE.Group);
assert.strictEqual(typeof g.userData.actualizar, "function");
const instancias = [];
g.traverse((o) => { if (o.isInstancedMesh) instancias.push(o); });
assert.strictEqual(instancias.length, 3 - 1, "enjambre y satelites son InstancedMesh");
const enj = g.userData.enjambre;
g.userData.actualizar(0, 0);
assert.ok(!enj.visible && enj.count === 0, "sin enjambre en etapa 0");
g.userData.actualizar(1.5, 1);
assert.ok(!enj.visible);
g.userData.actualizar(2.2, 1);
assert.ok(enj.visible && enj.count > 0);
const c22 = enj.count;
g.userData.actualizar(3, 2);
assert.ok(enj.count > c22 && enj.count <= 220);
// Determinismo: misma entrada, mismas matrices.
const m1 = Array.from(enj.instanceMatrix.array.slice(0, 32));
g.userData.actualizar(3, 2);
assert.deepStrictEqual(Array.from(enj.instanceMatrix.array.slice(0, 32)), m1);
for (let n = 0; n <= 3; n += 0.25) g.userData.actualizar(n, n * 3);
console.log("kardashev: todo correcto");

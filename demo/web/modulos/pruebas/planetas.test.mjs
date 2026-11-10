// Pruebas de planetas.js en Node (sin DOM): node planetas.test.mjs
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import path from "node:path";

const aqui = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(aqui, "..", "..");
globalThis.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(web, "modulos", "planetas.js"), "utf8"));
const P = globalThis.OB.planeta;
const MUNDOS = ["oceanico", "selvatico", "helado", "nebuloso", "desertico", "volcanico",
  "gaseoso", "cristalino"];
const W = 128, H = 64;

// Contexto 2D que registra las llamadas.
function grabador() {
  const llamadas = {};
  const base = {};
  return new Proxy(base, {
    get(t, p) {
      if (p === "llamadas") return llamadas;
      if (p in t) return t[p];
      return () => {
        llamadas[p] = (llamadas[p] || 0) + 1;
        return { addColorStop() {} };
      };
    },
    set(t, p, v) { t[p] = v; return true; },
  });
}

assert.deepStrictEqual(P.mundos, MUNDOS);

for (const m of MUNDOS) {
  P.limpiarCache();
  const a = Uint8ClampedArray.from(P.textura(m, 7, W, H).datos);
  assert.strictEqual(a.length, W * H * 4, m + ": tamano");
  P.limpiarCache();
  const b = P.textura(m, 7, W, H).datos;
  assert.deepStrictEqual(Array.from(a), Array.from(b), m + ": determinista");
  const c = P.textura(m, 8, W, H).datos;
  let dif = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== c[i]) dif++;
  assert.ok(dif > a.length * 0.05, m + ": otra semilla distinta (" + dif + ")");

  // Variedad: no es un color plano.
  let suma = 0, suma2 = 0;
  for (let i = 0; i < a.length; i += 4) { suma += a[i + 1]; suma2 += a[i + 1] * a[i + 1]; }
  const n = W * H, sd = Math.sqrt(suma2 / n - (suma / n) ** 2);
  assert.ok(sd > 4, m + ": textura con detalle (sd " + sd.toFixed(1) + ")");

  // Sin costura: la diferencia ultima/primera columna es del orden de la de dos vecinas.
  const col = (x, y, k) => a[(y * W + x) * 4 + k];
  let borde = 0, inter = 0;
  for (let y = 0; y < H; y++) {
    for (let k = 0; k < 3; k++) {
      borde += Math.abs(col(0, y, k) - col(W - 1, y, k));
      for (let x = 0; x < W - 1; x++) inter += Math.abs(col(x, y, k) - col(x + 1, y, k)) / (W - 1);
    }
  }
  borde /= H * 3; inter /= H * 3;
  assert.ok(borde <= inter * 3 + 5, m + ": costura (borde " + borde.toFixed(1)
    + ", interior " + inter.toFixed(1) + ")");
  assert.ok(borde < 30, m + ": borde pequeno");

  // Cache: mismos argumentos, mismo objeto.
  assert.strictEqual(P.textura(m, 7, W, H), P.textura(m, 7, W, H));

  const mini = P.miniatura(m, 3, 32);
  assert.strictEqual(mini.datos.length, 32 * 32 * 4);
  assert.strictEqual(mini.datos[(16 * 32 + 16) * 4 + 3], 255, m + ": centro opaco");

  // Dibujo 2D, viva y apagado.
  const g1 = grabador();
  P.canvas(g1, m, 3, 100, 100, 40, 1.5, { viva: true });
  assert.ok(g1.llamadas.arc && g1.llamadas.drawImage, m + ": dibuja disco y textura");
  assert.ok(g1.llamadas.createRadialGradient >= 2, m + ": atmosfera");
  const g2 = grabador();
  P.canvas(g2, m, 3, 100, 100, 40, 1.5, { viva: false });
  assert.ok(g2.llamadas.stroke >= 8, m + ": grietas al apagarse");
  assert.strictEqual(g2.llamadas.createRadialGradient, 1, m + ": sin atmosfera si apagado");
  P.canvas(grabador(), m, 3, 10, 10, 5, 0); // sin estado
}

// Mundo desconocido cae a oceanico.
assert.strictEqual(P.textura("nada", 1, 16, 8).datos.length, 16 * 8 * 4);

// three.js real (UMD).
vm.runInThisContext(fs.readFileSync(path.join(web, "vendor", "three.min.js"), "utf8"));
const THREE = globalThis.THREE;
assert.ok(THREE, "THREE cargado");
for (const m of MUNDOS) {
  const g = P.three(THREE, m, 5, 2);
  assert.ok(g instanceof THREE.Group);
  assert.strictEqual(typeof g.userData.actualizar, "function");
  assert.strictEqual(g.children.length, 2);
  const [esfera, atm] = g.children;
  assert.ok(esfera.material instanceof THREE.MeshStandardMaterial);
  assert.ok(esfera.material.map);
  assert.strictEqual(atm.material.side, THREE.BackSide);
  assert.strictEqual(atm.material.blending, THREE.AdditiveBlending);
  assert.ok(atm.geometry.parameters.radius > esfera.geometry.parameters.radius);
  g.userData.actualizar(2, { viva: true });
  assert.ok(atm.visible);
  g.userData.actualizar(3, { viva: false });
  assert.ok(!atm.visible);
  assert.ok(esfera.material.emissiveIntensity > 0);
  g.userData.actualizar(4, { viva: true });
  assert.ok(atm.visible && esfera.material.emissiveIntensity === 0);
  // Textura cacheada: la misma imagen para el mismo (mundo, semilla, tamano).
  const otra = P.three(THREE, m, 5, 1).children[0].material.map.image;
  assert.strictEqual(otra, esfera.material.map.image);
}
console.log("planetas: todo correcto");

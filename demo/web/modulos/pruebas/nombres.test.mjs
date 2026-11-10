// Pruebas de OB.nombres: determinismo, variedad y formato.
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const aqui = path.dirname(fileURLToPath(import.meta.url));
globalThis.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(aqui, "..", "nombres.js"), "utf8"));
const { generar } = globalThis.OB.nombres;

// Determinista
for (const s of [0, 1, 42, 123456, 4294967295]) assert.strictEqual(generar(s), generar(s));

// Variado: unas 4000 semillas consecutivas dan casi todos distintos
const vistos = new Set();
for (let s = 0; s < 4000; s++) vistos.add(generar(s));
assert.ok(vistos.size >= 3950, "nombres distintos: " + vistos.size);

// Formato: pronunciable, con inicial mayúscula, sin vocales ni consonantes en racha larga
const patron = /^[A-Z][a-z]+( (Prime|Minor|Major|Alfa|Nova|II|III|IV|V|VI|VII)|-\d{1,2})?$/;
for (let s = 0; s < 4000; s++) {
  const n = generar(s);
  assert.match(n, patron, n);
  assert.ok(n.length >= 3 && n.length <= 24, n);
  assert.ok(!/[^aeiou\s\d-]{5,}/i.test(n), "consonantes seguidas: " + n);
}

// Semillas no enteras o ausentes no fallan
assert.strictEqual(typeof generar(undefined), "string");
assert.strictEqual(typeof generar("7"), "string");
assert.strictEqual(generar("7"), generar(7));

const muestra = [0, 1, 2, 3, 4, 5].map(generar).join(", ");
console.log("nombres.test.mjs ok (" + vistos.size + " distintos; ej.: " + muestra + ")");

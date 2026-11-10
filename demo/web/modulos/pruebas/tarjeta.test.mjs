// Pruebas de OB.tarjeta: sin módulos hermanos (reemplazo de texto) y con stubs.
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const aqui = path.dirname(fileURLToPath(import.meta.url));
globalThis.window = globalThis;
const carga = (f) => vm.runInThisContext(fs.readFileSync(path.join(aqui, "..", f), "utf8"));
carga("estilos.js");
carga("nombres.js");
carga("tarjeta.js");
const OB = globalThis.OB;

const STEPS = 20;
const N_SYM = 3;
function episodio() {
  const alive = [], res = [], refuge = [], emision = [], kind = [], act = [];
  for (let t = 0; t <= STEPS; t++) {
    alive.push([1, 1, t >= 10 ? 0 : 1, 1]);
    res.push([1 + t, 2, 3, 4]);
    refuge.push([0, t >= 5 ? 1 : 0, 0, 0]);
  }
  for (let t = 0; t < STEPS; t++) {
    emision.push([1, 0, 2, 0]);
    kind.push([t < 3 ? 0 : 3, 6, t >= 10 ? -1 : 5, 4]);
    act.push([t < 3 ? 0 : 2 + N_SYM, 0, 0, 0]);
  }
  return {
    n: 4, steps: STEPS, seed: 7, deception: true, n_symbols: N_SYM,
    politicas: ["fijo:coop_real", "fijo:prudente", "fijo:ataque_solo", "red ft_x (seed 0)"],
    alive, res, refuge, emision, kind, act,
    mensajes: [[1, 1, 0], [2, 0, 1], [3, 1, 0], [30, 1, 0]],
    signals: [{ o: 0, t: 4, msg: 3 }, { o: 0, t: 6, msg: 3 }, { o: 1, t: 5, msg: 3 }],
  };
}
const ep = episodio();
const copia = JSON.stringify(ep);
const falso = () => ({ innerHTML: "" });

// Estado para los módulos hermanos
{
  const e0 = OB.tarjeta.estado(ep, 0, 2);
  assert.strictEqual(e0.viva, true);
  assert.strictEqual(e0.accion, "crecer");
  assert.strictEqual(e0.emite, 1);
  assert.strictEqual(e0.recursos, 3);
  assert.ok(Math.abs(e0.nivel - (3 * Math.log(3)) / Math.log(12)) < 1e-9);
  // transmitir con act = 2 + n_symbols
  assert.strictEqual(OB.tarjeta.estado(ep, 0, 5).accion, "mentir");
  assert.strictEqual(OB.tarjeta.estado(ep, 1, 6).refugio, true);
  assert.strictEqual(OB.tarjeta.estado(ep, 2, 12).viva, false);
  assert.strictEqual(OB.tarjeta.estado(ep, 2, 12).accion, "");
  assert.strictEqual(OB.tarjeta.estado(ep, 3, STEPS).accion, "compartir"); // t = steps usa steps-1
  const sinEmision = { ...ep, emision: undefined };
  assert.strictEqual(OB.tarjeta.estado(sinEmision, 0, 3).emite, 0);
}

// Sin módulos hermanos: reemplazos de texto, sin lanzar
{
  assert.ok(!OB.astronauta && !OB.planeta && !OB.kardashev);
  const c = falso();
  OB.tarjeta.render(c, ep, 0, 8, {});
  const h = c.innerHTML;
  assert.match(h, /class="ob-ficha /);
  for (const cl of ["ob-ficha__nombre", "ob-ficha__chips", "ob-ficha__stat", "ob-ficha__escaner",
    "ob-ficha__accion", "ob-ficha__astronauta", "ob-ficha__planeta", "ob-chip"]) {
    assert.ok(h.includes(cl), "falta " + cl);
  }
  assert.match(h, /Astronauta no disponible \(Cooperadora\)/);
  assert.match(h, /Planeta oceánico/);
  assert.match(h, /Nivel \d,\d\d de 3/);
  assert.match(h, /Civilización 1/);
  assert.match(h, /Mundo:<\/span> <span class="ob-chip__valor">oceánico/);
  assert.match(h, /Política:<\/span> <span class="ob-chip__valor">Cooperadora/);
  assert.match(h, /Estado:<\/span> <span class="ob-chip__valor">viva/);
  assert.match(h, /<polyline/);
  assert.match(h, /2 enviados, 1 recibidos/);
  assert.match(h, /Mentiras<\/span><span class="ob-ficha__valor">2 emitidas/);
  assert.match(h, /Miente \(turno 8\)/);
  assert.match(h, /Etapa II: estelar/);

  // Muerta, y la acción más reciente anterior a la muerte
  OB.tarjeta.render(c, ep, 2, 14, {});
  assert.match(c.innerHTML, /data-viva="false"/);
  assert.match(c.innerHTML, /chip__valor">muerta/);
  assert.match(c.innerHTML, /Ataca \(turno 9\)/);
  assert.match(c.innerHTML, /Política:<\/span> <span class="ob-chip__valor">Belicosa/);

  // Refugio, y reemplaza el contenido en cada llamada
  OB.tarjeta.render(c, ep, 1, 8, {});
  assert.match(c.innerHTML, /Refugio:/);
  assert.strictEqual((c.innerHTML.match(/<article/g) || []).length, 1);

  // Sin deception no hay mentiras; sin emisión no hay chip de emisión
  const simple = { ...ep, deception: false, emision: undefined, politicas: undefined,
    policy: "fijo:solo" };
  OB.tarjeta.render(c, simple, 0, 8, {});
  assert.ok(!/Mentiras/.test(c.innerHTML));
  assert.ok(!/Emisión:/.test(c.innerHTML));
  assert.match(c.innerHTML, /Solitaria/);

  // t fuera de rango y t = 0 no fallan
  OB.tarjeta.render(c, ep, 3, 999, {});
  OB.tarjeta.render(c, ep, 3, -4, {});
  assert.match(c.innerHTML, /Comparte [(]turno 0[)]/);
}

// Nombre y semilla estables
{
  const a = falso(), b = falso();
  OB.tarjeta.render(a, ep, 1, 3, {});
  OB.tarjeta.render(b, ep, 1, 15, {});
  const nombre = (h) => h.match(/ob-ficha__nombre">([^<]+)</)[1];
  assert.strictEqual(nombre(a.innerHTML), nombre(b.innerHTML));
  const c = falso();
  OB.tarjeta.render(c, ep, 0, 3, {});
  assert.notStrictEqual(nombre(a.innerHTML), nombre(c.innerHTML));
  assert.strictEqual(JSON.stringify(ep), copia, "no muta ep");
}

// Con stubs de los módulos hermanos
{
  const llamadas = { astro: [], planeta: 0, medidor: [] };
  OB.astronauta = {
    svg: (estilo, estado, tam) => {
      llamadas.astro.push([estilo.id, estado, tam]);
      return '<svg class="stub-astro"><title>astro</title></svg>';
    },
  };
  OB.planeta = {
    miniatura: (mundo, semilla, tam) => {
      llamadas.planeta++;
      return { mundo, semilla, tam, attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } };
    },
  };
  OB.kardashev = {
    nivel: (r) => Math.min(3, r / 4),
    etapa: (k) => ({ id: ["0", "I", "II", "III"][Math.floor(k)], nombre: "de prueba" }),
    medidorSvg: (k, ancho) => {
      llamadas.medidor.push([k, ancho]);
      return '<svg class="stub-med"></svg>';
    },
  };
  const hueco = { hijos: [], appendChild(n) { this.hijos.push(n); } };
  const c = { innerHTML: "", querySelector: (s) => (s === "[data-planeta]" ? hueco : null) };

  OB.tarjeta.render(c, ep, 0, 8, {});
  assert.match(c.innerHTML, /stub-astro/);
  assert.match(c.innerHTML, /stub-med/);
  assert.ok(!/no disponible/.test(c.innerHTML));
  const [id, estado, tam] = llamadas.astro[0];
  assert.strictEqual(id, "cooperador");
  assert.strictEqual(tam, 120);
  assert.strictEqual(estado.accion, "mentir");
  assert.strictEqual(estado.nivel, 2.25);
  assert.deepStrictEqual(llamadas.medidor[0], [2.25, 180]);
  assert.match(c.innerHTML, /Etapa II: de prueba/);
  assert.strictEqual(llamadas.planeta, 1);
  assert.strictEqual(hueco.hijos.length, 1);
  assert.strictEqual(hueco.hijos[0].mundo, "oceanico");
  assert.strictEqual(hueco.hijos[0].attrs.role, "img");

  // Misma civilización en otro turno: no se regenera la miniatura, se reinserta
  OB.tarjeta.render(c, ep, 0, 12, {});
  assert.strictEqual(llamadas.planeta, 1);
  assert.strictEqual(hueco.hijos.length, 2);
  assert.strictEqual(hueco.hijos[0], hueco.hijos[1]);
  // Otra civilización (otro mundo y semilla): sí
  OB.tarjeta.render(c, ep, 1, 12, {});
  assert.strictEqual(llamadas.planeta, 2);
  assert.strictEqual(hueco.hijos[2].mundo, "helado");

  // Un módulo que lanza: reemplazo de texto
  OB.astronauta.svg = () => { throw new Error("falla"); };
  OB.planeta.miniatura = () => { throw new Error("falla"); };
  const d = { innerHTML: "" };
  OB.tarjeta.render(d, ep, 3, 4, {});
  assert.match(d.innerHTML, /Astronauta no disponible/);
  assert.match(d.innerHTML, /Planeta cristalino/);
  assert.ok(!/[\u{1F300}-\u{1FAFF}]/u.test(d.innerHTML), "sin emojis");
}

console.log("tarjeta.test.mjs ok");

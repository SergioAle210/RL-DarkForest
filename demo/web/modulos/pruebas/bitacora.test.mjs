// Pruebas de OB.bitacora con un episodio sintético de 3 civilizaciones.
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const aqui = path.dirname(fileURLToPath(import.meta.url));
globalThis.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(aqui, "..", "bitacora.js"), "utf8"));
const B = globalThis.OB.bitacora;

const STEPS = 60;
// C1 (0) crece y se refugia en 38; C2 (1) muere por el estallido de 40; C3 (2) cae por un ataque.
function episodio(conMuertes) {
  const alive = [], res = [], refuge = [];
  for (let t = 0; t <= STEPS; t++) {
    alive.push([1, t >= 41 ? 0 : 1, t >= 9 ? 0 : 1]);
    res.push([1 + 0.3 * t, 2, 1]);
    refuge.push([t >= 38 ? 1 : 0, 0, 0]);
  }
  const mensajes = [];
  for (let t = 3; t < STEPS; t++) mensajes.push([t, 1, 0], [t, 0, 1]);
  const signals = [{ o: 1, t: 1, msg: 0 }];
  for (let t = 2; t < STEPS; t += 3) signals.push({ o: 1, t, msg: 3 });
  const ep = {
    n: 3, steps: STEPS, v: 0.1, c: 1, pos: [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
    alive, res, refuge, deception: true, n_symbols: 2, mensajes, signals,
    projectiles: [{ o: 0, t: 5, tx: 0.9, ty: 0.9 }],
    bursts: [40],
  };
  if (conMuertes) {
    ep.muertes = [
      { t: 9, civ: 2, causa: "ataque", atacante: 0 },
      { t: 41, civ: 1, causa: "anomalia", atacante: -1 },
    ];
  }
  return ep;
}

const por = (ev, tipo) => ev.filter((e) => e.tipo === tipo);

// Sin ep.muertes: se deducen del primer alive = 0
{
  const ep = episodio(false);
  const copia = JSON.stringify(ep);
  const ev = B.eventos(ep);
  assert.strictEqual(JSON.stringify(ep), copia, "no muta ep");
  for (let k = 1; k < ev.length; k++) assert.ok(ev[k - 1].t <= ev[k].t, "ordenado");
  for (const e of ev) {
    assert.ok(typeof e.texto === "string" && e.texto.startsWith("Turno " + e.t + ":"), e.texto);
    assert.ok(Array.isArray(e.civs));
  }
  const m = por(ev, "muerte");
  assert.strictEqual(m.length, 2);
  const m3 = m.find((e) => e.civs.includes(2));
  assert.strictEqual(m3.t, 9);
  assert.match(m3.texto, /C3 es destruida por un ataque de C1/);
  const m2 = m.find((e) => e.civs[0] === 1);
  assert.match(m2.texto, /C2 se apaga por el estallido de la anomal/);

  const at = por(ev, "ataque");
  assert.strictEqual(at.length, 1);
  assert.match(at[0].texto, /C1 lanza una sonda de ataque hacia C3/);

  const est = por(ev, "estallido");
  assert.strictEqual(est.length, 1);
  assert.strictEqual(est[0].t, 40);
  assert.match(est[0].texto, /C1 estaba refugiada/);
  assert.match(est[0].texto, /C2 no estaba refugiada y se apaga/);
  assert.ok(!/C3/.test(est[0].texto), "la civilización muerta no figura");

  // Mensajes repetidos: el primero y un resumen por tramo de 25 turnos
  const ms = por(ev, "mensaje");
  assert.ok(ms.length >= 3 && ms.length <= 6, "mensajes colapsados: " + ms.length);
  assert.strictEqual(ms[0].t, 3);
  assert.match(ms[0].texto, /primer mensaje/);

  // Mentiras: la primera y resúmenes
  const li = por(ev, "mentira");
  assert.ok(li.length >= 2 && li.length <= 4, "mentiras colapsadas: " + li.length);
  assert.strictEqual(li[0].t, 2);
  assert.match(li[0].texto, /C2 emite una medición falsa/);
  assert.ok(!ev.some((e) => e.tipo === "mentira" && e.t === 1), "msg 0 no es mentira");

  // Refugio: solo cuando empieza
  const rf = por(ev, "refugio");
  assert.strictEqual(rf.length, 1);
  assert.strictEqual(rf[0].t, 38);

  // Hitos con la fórmula local: niveles 1, 2 y 3 de C1 (res = 1 + 0.3 t)
  const h = por(ev, "hito").filter((e) => e.civs[0] === 0);
  assert.deepStrictEqual(h.map((e) => e.t), [5, 15, 37]);
  assert.match(h[0].texto, /etapa I de Kardashev/);
  assert.match(h[2].texto, /etapa III/);
  // C2 parte de recursos 2 (nivel 0,8) y no cruza nada
  assert.strictEqual(por(ev, "hito").filter((e) => e.civs[0] === 1).length, 0);
}

// Con ep.muertes se usa tal cual
{
  const ev = B.eventos(episodio(true));
  const m = por(ev, "muerte");
  assert.strictEqual(m.length, 2);
  assert.ok(m.some((e) => /C3 es destruida por un ataque de C1/.test(e.texto)));
}

// Episodio mínimo, sin campos opcionales
{
  assert.deepStrictEqual(B.eventos(null), []);
  const ev = B.eventos({ n: 2, steps: 3, alive: [[1, 1], [1, 1], [1, 1], [1, 1]] });
  assert.deepStrictEqual(ev, []);
}

// Muerte sin causa conocida
{
  const ev = B.eventos({ n: 1, steps: 3, alive: [[1], [1], [0], [0]] });
  assert.strictEqual(ev.length, 1);
  assert.match(ev[0].texto, /Turno 2: C1 se apaga\./);
}

// OB.kardashev presente: se usa su nivel y su etapa
{
  globalThis.OB.kardashev = {
    nivel: (r) => (r >= 3 ? 2.5 : 0),
    etapa: (k) => ({ id: ["0", "I", "II", "III"][Math.floor(k)], nombre: "de prueba" }),
  };
  const ev = B.eventos(episodio(true));
  const h = por(ev, "hito").filter((e) => e.civs[0] === 0);
  assert.strictEqual(h.length, 2); // cruza de 0 a 2: etapas I y II en el mismo turno
  assert.match(h[1].texto, /etapa II de Kardashev \(de prueba\)/);
  delete globalThis.OB.kardashev;
}

// render sobre un contenedor falso
{
  const ev = B.eventos(episodio(true));
  const cont = { innerHTML: "", scrollTop: 0, scrollHeight: 500 };
  B.render(cont, ev, 0, {});
  assert.match(cont.innerHTML, /role="log"/);
  assert.match(cont.innerHTML, /aria-live="off"/);
  assert.match(cont.innerHTML, /Sin sucesos/);

  B.render(cont, ev, 45);
  const visibles = ev.filter((e) => e.t <= 45).length;
  assert.strictEqual((cont.innerHTML.match(/<li /g) || []).length, visibles);
  assert.strictEqual((cont.innerHTML.match(/data-ultimo="true"/g) || []).length, 1);
  assert.match(cont.innerHTML, /data-tipo="estallido"/);
  assert.match(cont.innerHTML, /<svg[^>]*><title>Estallido<\/title>/);
  assert.ok(cont.scrollTop > 0, "desplazada al final");
  assert.ok(!/[\u{1F300}-\u{1FAFF}]/u.test(cont.innerHTML), "sin emojis");

  // Sin cambios no se repinta; con cambios, sí
  cont.scrollTop = 0;
  B.render(cont, ev, 45);
  assert.strictEqual(cont.scrollTop, 0);
  B.render(cont, ev, 60);
  assert.ok(cont.scrollTop > 0);

  // Límite de entradas y escape de HTML
  B.render(cont, ev, 60, { maximo: 3 });
  assert.strictEqual((cont.innerHTML.match(/<li /g) || []).length, 3);
  const c2 = { innerHTML: "", scrollTop: 0 };
  B.render(c2, [{ t: 0, tipo: "mensaje", civs: [0], texto: "<b>&" }], 0);
  assert.ok(c2.innerHTML.includes("&lt;b&gt;&amp;"));
}

console.log("bitacora.test.mjs ok");

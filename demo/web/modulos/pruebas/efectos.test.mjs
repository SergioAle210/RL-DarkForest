// Pruebas de efectos.js sin navegador: contexto 2D falso y three.js real (UMD) en Node.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const aqui = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(aqui, "..", "..");
globalThis.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(web, "modulos", "efectos.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(web, "vendor", "three.min.js"), "utf8"));
const OB = globalThis.OB;
const THREE = globalThis.THREE;

// Contexto 2D falso: registra cada llamada con sus argumentos.
function ctxFalso() {
  const llamadas = [];
  const ctx = new Proxy({}, {
    get(_, nombre) {
      if (nombre === "llamadas") return llamadas;
      return (...args) => { llamadas.push({ nombre, args }); };
    },
    set() { return true; },
  });
  return ctx;
}
const TRAZADO = new Set(["stroke", "fill", "arc", "lineTo", "moveTo", "ellipse"]);
const nTrazado = (ctx) => ctx.llamadas.filter((c) => TRAZADO.has(c.nombre)).length;

const T = { burst: "#ff9b3d", refuge: "#4aa3ff", danger: "#ff4040", sun: "#ffd24a", dead: "#444444",
  ink: "#ffffff", "civ-1": "#e15759", "civ-2": "#4e79a7" };
const TAM = 100;
const h = {
  P: (q) => [q[0] * TAM, q[1] * TAM], wrap: (v) => v - Math.floor(v), T,
  civ: (i) => T["civ-" + ((i % 8) + 1)], escala: TAM,
};

// Episodio sintético: la civ 0 (x=0.9) ataca a la civ 1 (x=0.12) cruzando el borde.
function crearEp(conMuertes) {
  const steps = 40;
  const alive = [];
  const refuge = [];
  for (let t = 0; t <= steps; t++) {
    alive.push([1, t >= 10 ? 0 : 1]);
    refuge.push([t >= 25 && t <= 30 ? 1 : 0, 0]);
  }
  const ep = {
    n: 2, steps, v: 0.05, pos: [[0.9, 0.5], [0.12, 0.5]], alive, refuge,
    projectiles: [{ o: 0, t: 5, tx: 0.12, ty: 0.5 }],
    bursts: [20], zone: { x: 0.95, y: 0.5, r: 0.2 },
  };
  if (conMuertes) ep.muertes = [{ t: 10, civ: 1, causa: "ataque", atacante: 0 }];
  return ep;
}

// Camino corto: cruza el borde cuando corresponde.
{
  const tr = OB.efectos.trayectorias(crearEp(true))[0];
  assert.ok(Math.abs(tr.dx - 0.22) < 1e-9, "delta x por el borde");
  assert.ok(tr.cruza, "debe cruzar el borde");
  assert.equal(tr.tj, 1);
  assert.equal(tr.res, 2, "el objetivo muere en el turno siguiente");
  assert.ok(tr.nNaves >= 3 && tr.nNaves <= 5);
  const sin = OB.efectos.trayectorias({ ...crearEp(true), pos: [[0.1, 0.5], [0.8, 0.5]],
    projectiles: [{ o: 0, t: 1, tx: 0.8, ty: 0.5 }], muertes: [], alive: [[1, 1]] })[0];
  assert.ok(Math.abs(sin.dx + 0.3) < 1e-9, "hacia atrás por el borde");
  const dentro = OB.efectos.trayectorias({ ...crearEp(true), pos: [[0.3, 0.5], [0.5, 0.5]],
    projectiles: [{ o: 0, t: 1, tx: 0.5, ty: 0.5 }], muertes: [], alive: [[1, 1]] })[0];
  assert.ok(!dentro.cruza, "no cruza dentro de la arena");
  assert.equal(dentro.res, 1, "sobrevive: el escudo resiste");
}

// Determinismo de la flota.
{
  const a = OB.efectos.trayectorias(crearEp(true))[0];
  const b = OB.efectos.trayectorias(crearEp(true))[0];
  assert.equal(a.nNaves, b.nNaves);
  assert.equal(a.fase, b.fase);
}

// Dibujo 2D: no lanza y traza donde hay eventos.
for (const conMuertes of [true, false]) {
  const ep = crearEp(conMuertes);
  const fx = OB.efectos.crear2d();
  for (const t of [0, 1, 5, 6, 7, 8, 9, 9.5, 10, 11, 14, 20, 22, 26, 39, 40]) {
    const ctx = ctxFalso();
    fx.dibujar(ctx, ep, t, h);
  }
  let ctx = ctxFalso();
  fx.dibujar(ctx, ep, 2, h);
  assert.equal(nTrazado(ctx), 0, "turno sin eventos no dibuja");
  ctx = ctxFalso();
  fx.dibujar(ctx, ep, 7, h);
  assert.ok(nTrazado(ctx) > 10, "turno de vuelo dibuja");
  ctx = ctxFalso();
  fx.dibujar(ctx, ep, 30, h);
  assert.ok(ctx.llamadas.some((c) => c.nombre === "arc"), "refugio dibuja cúpula");
  ctx = ctxFalso();
  fx.dibujar(ctx, ep, 35, h);
  assert.ok(nTrazado(ctx) > 0, "planeta muerto se muestra apagado (" + conMuertes + ")");
}

// Sin alive ni muertes ni refuge ni proyectiles: tolera campos ausentes.
{
  const fx = OB.efectos.crear2d();
  const ctx = ctxFalso();
  fx.dibujar(ctx, { n: 1, pos: [[0.5, 0.5]], v: 0.05 }, 3, h);
  fx.dibujar(ctx, { n: 1, pos: [[0.5, 0.5]], v: 0.05, bursts: [3] }, 3, h);
  assert.equal(nTrazado(ctx), 0);
}

// El camino partido: ningún segmento moveTo -> lineTo cruza la arena entera.
{
  const ep = crearEp(true);
  const fx = OB.efectos.crear2d();
  const ctx = ctxFalso();
  fx.dibujar(ctx, ep, 8, h);
  const ll = ctx.llamadas;
  let cruces = 0;
  for (let i = 0; i < ll.length - 1; i++) {
    if (ll[i].nombre === "moveTo" && ll[i + 1].nombre === "lineTo") {
      assert.ok(Math.abs(ll[i + 1].args[0] - ll[i].args[0]) < 60, "segmento cruza la arena");
    }
  }
  for (const c of ll) if (c.nombre === "lineTo" && Math.abs(c.args[0] - TAM) < 1e-6) cruces++;
  assert.ok(cruces > 0, "debe haber un segmento que sale por el borde derecho");
  assert.ok(ll.some((c) => c.nombre === "moveTo" && Math.abs(c.args[0]) < 1e-6),
    "y otro que entra por el izquierdo");
}

// Anomalía: círculo con desplazamiento de -1 recortado al lienzo.
{
  const fx = OB.efectos.crear2d();
  const ctx = ctxFalso();
  fx.dibujar(ctx, crearEp(true), 21, h);
  assert.ok(ctx.llamadas.some((c) => c.nombre === "clip"), "recorta al lienzo");
  assert.ok(ctx.llamadas.some((c) => c.nombre === "arc" && c.args[0] < 0), "copia envuelta");
  assert.equal(ctx.llamadas.filter((c) => c.nombre === "save").length,
    ctx.llamadas.filter((c) => c.nombre === "restore").length);
  const ctx2 = ctxFalso();
  fx.dibujar(ctx2, crearEp(true), 24, h);
  assert.ok(!ctx2.llamadas.some((c) => c.nombre === "clip"), "tras 4 turnos no hay supernova");
}

// 3D: toro simple, el grupo se acota y no crece.
{
  const R = 3;
  const r = 1;
  const h3 = {
    T, civ: h.civ,
    p3(x, y, alto) {
      const u = x * Math.PI * 2;
      const v = y * Math.PI * 2;
      const rr = R + (r + alto) * Math.cos(v);
      return new THREE.Vector3(rr * Math.cos(u), (r + alto) * Math.sin(v), rr * Math.sin(u));
    },
    n3(x, y) {
      const u = x * Math.PI * 2;
      const v = y * Math.PI * 2;
      return new THREE.Vector3(Math.cos(v) * Math.cos(u), Math.sin(v), Math.cos(v) * Math.sin(u));
    },
  };
  const escena = new THREE.Scene();
  const fx = OB.efectos.crear3d(THREE, escena, h3);
  assert.equal(escena.children.length, 1);
  const grupo = escena.children[0];
  const base = grupo.children.length;
  assert.ok(base > 10, "crea objetos");
  assert.ok(base < 400, "número acotado");
  const ep = crearEp(true);
  for (let t = 0; t <= 40; t += 0.5) {
    fx.actualizar(ep, t);
    assert.equal(grupo.children.length, base, "sin acumular objetos en t=" + t);
    assert.equal(escena.children.length, 1);
  }
  for (let rep = 0; rep < 3; rep++) for (let t = 0; t <= 40; t++) fx.actualizar(ep, t);
  assert.equal(grupo.children.length, base);
  fx.actualizar(ep, 7);
  const sondas = grupo.children.find((o) => o.isInstancedMesh);
  assert.ok(sondas.count > 0, "hay sonda en vuelo");
  fx.actualizar(ep, 2);
  assert.equal(sondas.count, 0, "sin eventos no hay sondas");
  fx.actualizar(null, 0);
  fx.actualizar({ n: 1, pos: [[0.5, 0.5]] }, 3);
  fx.limpiar();
  assert.equal(escena.children.length, 0);
}

console.log("efectos.test.mjs: ok");

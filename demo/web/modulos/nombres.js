/* Nombres de sistema procedurales. Decorativo: el nombre no es un dato del episodio. */
(function (OB) {
  "use strict";

  const INICIOS = ["b", "c", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
    "th", "kr", "tr", "sh", "gl", "br", "st", "qu"];
  const SIMPLES = INICIOS.filter((c) => c.length === 1);
  const VOCALES = ["a", "e", "i", "o", "u", "a", "e", "o", "a", "e", "ia", "ai"];
  const CODAS_FINALES = ["", "", "", "n", "r", "s", "l", "x", "th", "m"];
  const CODAS_MEDIAS = ["", "", "", "", "n", "r", "s", "l"];
  const ROMANOS = ["II", "III", "IV", "V", "VI", "VII"];
  const PALABRAS = ["Prime", "Minor", "Major", "Alfa", "Nova"];

  /** Generador pseudoaleatorio con semilla (mulberry32). */
  function mulberry32(a) {
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function elegir(rnd, lista) {
    return lista[Math.floor(rnd() * lista.length)];
  }

  /** Una sílaba: la primera siempre lleva consonante inicial y la última puede cerrar. */
  function silaba(rnd, primera, ultima) {
    const inicio = elegir(rnd, primera ? INICIOS : SIMPLES);
    const coda = elegir(rnd, ultima ? CODAS_FINALES : CODAS_MEDIAS);
    return inicio + elegir(rnd, VOCALES) + coda;
  }

  /**
   * Nombre de sistema pronunciable y determinista a partir de una semilla entera.
   * Ejemplos: «Thalesor Prime», «Ketu-47».
   */
  function generar(semilla) {
    // Se descartan unos valores para que semillas consecutivas no se parezcan.
    const rnd = mulberry32((Number(semilla) | 0) ^ 0x9e3779b9);
    rnd(); rnd();
    const r = rnd();
    const n = r < 0.35 ? 2 : r < 0.9 ? 3 : 4;
    let base = "";
    for (let k = 0; k < n; k++) base += silaba(rnd, k === 0, k === n - 1);
    base = base.charAt(0).toUpperCase() + base.slice(1);
    const s = rnd();
    if (s < 0.45) return base;
    if (s < 0.7) return base + " " + elegir(rnd, PALABRAS);
    if (s < 0.9) return base + "-" + (2 + Math.floor(rnd() * 98));
    return base + " " + elegir(rnd, ROMANOS);
  }

  OB.nombres = { generar };
})((window.OB = window.OB || {}));

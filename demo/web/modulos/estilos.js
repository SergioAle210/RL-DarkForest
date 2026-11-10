/* Estilos de civilización: astronauta y mundo según la estrategia que juega.
 * Es la única fuente de la identidad visual: el resto de los módulos la consultan.
 * Los colores son arte de las pieles (no tokens de interfaz). */
(function (OB) {
  "use strict";

  // Cada estilo: id, nombre y rol en español, tipo de mundo, paleta del traje e insignia.
  const TABLA = {
    cooperador: { id: "cooperador", nombre: "Cooperadora", rol: "Comparte lo que mide y usa lo que recibe", mundo: "oceanico", traje: "#eef3f8", acento: "#2f9bd6", visor: "#8fdcff", mochila: "#a9bdd0", insignia: "ola" },
    reciproco: { id: "reciproco", nombre: "Recíproca", rol: "Deja de compartir con quien no responde", mundo: "selvatico", traje: "#e6efe2", acento: "#3d9a5b", visor: "#a6e8b4", mochila: "#a4b99d", insignia: "hoja" },
    prudente: { id: "prudente", nombre: "Prudente", rol: "Solo acepta mensajes verificados", mundo: "helado", traje: "#f2f6fb", acento: "#5a7fc4", visor: "#cfe3ff", mochila: "#b9c6da", insignia: "cristal" },
    mentiroso: { id: "mentiroso", nombre: "Mentirosa", rol: "Comparte mediciones falsas", mundo: "nebuloso", traje: "#3b2a4d", acento: "#d98bb8", visor: "#f0b6dd", mochila: "#5a4470", insignia: "espejo" },
    solo: { id: "solo", nombre: "Solitaria", rol: "Se protege con sus propias mediciones", mundo: "desertico", traje: "#e7d7b8", acento: "#c2813a", visor: "#ffd9a0", mochila: "#b59a6c", insignia: "duna" },
    atacante: { id: "atacante", nombre: "Belicosa", rol: "Dispara al primer contacto", mundo: "volcanico", traje: "#4a2a26", acento: "#ff7a45", visor: "#ffb48a", mochila: "#6b3b33", insignia: "llama" },
    ignorar: { id: "ignorar", nombre: "Callada", rol: "Solo crece y no se protege", mundo: "gaseoso", traje: "#c9ccd6", acento: "#8a90a8", visor: "#dfe3ee", mochila: "#9a9fb2", insignia: "nube" },
    red: { id: "red", nombre: "Neuronal", rol: "Política aprendida con aprendizaje por refuerzo", mundo: "cristalino", traje: "#dff6f0", acento: "#00a58e", visor: "#9ff4e2", mochila: "#9fcfc4", insignia: "nodos" },
  };

  // Equivalencia entre la política del servidor y el estilo.
  const POR_ESTRATEGIA = {
    coop_real: "cooperador", recip: "reciproco", prudente: "prudente", mentiroso: "mentiroso",
    solo: "solo", ataque_solo: "atacante", ignorar: "ignorar",
  };

  /** Estilo de una política: "fijo:coop_real" o una red ("red:...", "red ft_... (seed 0)"). */
  function deEstrategia(politica) {
    const p = String(politica || "");
    if (p.startsWith("fijo:")) return TABLA[POR_ESTRATEGIA[p.slice(5)]] || TABLA.red;
    if (POR_ESTRATEGIA[p]) return TABLA[POR_ESTRATEGIA[p]];
    return TABLA.red;
  }

  OB.estilos = { tabla: TABLA, ids: Object.keys(TABLA), deEstrategia };
})((window.OB = window.OB || {}));

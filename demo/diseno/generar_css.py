"""Compila los tokens del sistema de diseño a CSS para la demo web.

Lee demo/diseno/project/tokens.json y components/bundle.css y escribe demo/web/tokens.css y
demo/web/componentes.css. Así los tokens son la única fuente de los valores y la demo usa los
mismos componentes que el sistema de diseño publicado.

Uso: python -m demo.diseno.generar_css
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ.parent / "web"


def valor(token: dict, tema: str, primero: str) -> str:
    """Valor de un token en un tema; si falta, hereda el del primer tema."""
    v = token["value"]
    return v if isinstance(v, str) else v.get(tema, v[primero])


def compilar(tokens: dict) -> str:
    """CSS de variables por tema, escalas y una clase por estilo tipográfico."""
    temas = [t["id"] for t in tokens["color"]["themes"]]
    primero = temas[0]
    lineas = [f"/* {tokens['name']} — generated from tokens.json */"]
    for tema in temas:
        selector = (
            f':root, [data-theme="{primero}"]' if tema == primero else f'[data-theme="{tema}"]'
        )
        cuerpo = [f"  --{t['name']}: {valor(t, tema, primero)};" for t in tokens["color"]["tokens"]]
        cuerpo += [
            f"  --{t['name']}: {valor(t, tema, primero)};" for t in tokens["shadow"]["tokens"]
        ]
        lineas += [selector + " {", *cuerpo, "}"]
    escalas = [t for familia in ("spacing", "radius") for t in tokens[familia]["tokens"]]
    familias = [f"  --font-{k}: {v};" for k, v in tokens["type"]["families"].items()]
    lineas += [":root {", *[f"  --{t['name']}: {t['value']};" for t in escalas], *familias, "}"]
    for grupo in tokens["type"]["groups"]:
        for estilo in grupo["styles"]:
            familia = estilo.get("family", grupo["family"])
            reglas = [
                f"font-family: var(--font-{familia})",
                f"font-size: {estilo['fontSize']}",
                f"line-height: {estilo['lineHeight']}",
                f"font-weight: {estilo['fontWeight']}",
            ]
            if "letterSpacing" in estilo:
                reglas.append(f"letter-spacing: {estilo['letterSpacing']}")
            lineas.append(f".{estilo['name']} {{ " + "; ".join(reglas) + "; }")
    return "\n".join(lineas) + "\n"


def main() -> None:
    """Escribe los CSS de la demo."""
    tokens = json.loads((RAIZ / "project" / "tokens.json").read_text(encoding="utf-8"))
    (SALIDA / "tokens.css").write_text(compilar(tokens), encoding="utf-8")
    shutil.copy(RAIZ / "project" / "components" / "bundle.css", SALIDA / "componentes.css")
    print(f"{SALIDA / 'tokens.css'} y {SALIDA / 'componentes.css'}")


if __name__ == "__main__":
    main()

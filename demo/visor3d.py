"""Visor 3D de un episodio sobre un toro, con PyVista.

La arena es un toro (lo que significa que los bordes se unen), así que aquí se dibuja como un
toro de verdad: las civilizaciones son esferas cuyo tamaño crece con el logaritmo de
sus recursos, las señales son anillos que se expanden por la superficie a la velocidad c, los
proyectiles viajan por el camino más corto y el estallido ilumina la zona de la anomalía. Los
colores salen de demo/diseno/project/tokens.json (temas noche y carta). Lee el JSON que escribe
viz.exportar_json o la demo, y no ejecuta el simulador.

Uso: python -m demo.visor3d --episodio demo/web/catalogo/escenario_01.json              # ventana
     python -m demo.visor3d --episodio <json> --turno 60 --out foto.png                 # imagen
     python -m demo.visor3d --episodio <json> --out video.gif --cada 2 --giro           # animación
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv

RAIZ = Path(__file__).resolve().parent
GRANDE, PEQUENO = 3.0, 1.1  # radio mayor y radio del tubo del toro
ACCIONES = ["crecer", "ocultarse", "observar", "transmitir", "compartir", "atacar", "refugiarse"]


def colores(tema: str) -> dict[str, str]:
    """Colores del sistema de diseño en un tema; los que faltan heredan del primero."""
    tokens = json.loads((RAIZ / "diseno" / "project" / "tokens.json").read_text(encoding="utf-8"))
    primero = tokens["color"]["themes"][0]["id"]
    return {t["name"]: t["value"].get(tema, t["value"][primero]) for t in tokens["color"]["tokens"]}


def rgb(hex_color: str) -> np.ndarray:
    """Color hexadecimal a un arreglo de tres componentes en 0 a 1."""
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def mezcla(color: str, fondo: str, alfa: float) -> np.ndarray:
    """Color mezclado con el fondo: una forma de opacidad que sirve en líneas."""
    return rgb(color) * alfa + rgb(fondo) * (1 - alfa)


def toro(x, y, altura: float = 0.0) -> np.ndarray:
    """Punto de la superficie del toro para (x, y) en [0, 1)^2, elevado `altura` hacia afuera."""
    u, v = 2 * np.pi * np.asarray(x), 2 * np.pi * np.asarray(y)
    rho = GRANDE + (PEQUENO + altura) * np.cos(v)
    return np.stack([rho * np.cos(u), rho * np.sin(u), (PEQUENO + altura) * np.sin(v)], axis=-1)


def normal(x: float, y: float) -> np.ndarray:
    """Normal exterior del toro en (x, y)."""
    u, v = 2 * np.pi * x, 2 * np.pi * y
    return np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])


def anillo(centro, radio: float, color: np.ndarray, altura: float, puntos: int = 160):
    """Polilínea de una circunferencia del dominio (x, y) sobre el toro, sin cruzar el borde."""
    a = np.linspace(0, 2 * np.pi, puntos + 1)
    x = (centro[0] + radio * np.cos(a)) % 1.0
    y = (centro[1] + radio * np.sin(a)) % 1.0
    pts = toro(x, y, altura)
    unen = (np.abs(np.diff(x)) < 0.5) & (np.abs(np.diff(y)) < 0.5)
    celdas = np.array([[2, i, i + 1] for i in np.flatnonzero(unen)], dtype=np.int64)
    return (
        pts,
        celdas.ravel() if len(celdas) else np.array([], dtype=np.int64),
        np.tile(color, (len(pts), 1)),
    )


def lineas(partes) -> pv.PolyData | None:
    """Une varias polilíneas con su color por punto en una sola malla."""
    todos, celdas, colores_ = [], [], []
    base = 0
    for pts, cel, col in partes:
        if len(cel) == 0:
            continue
        c = cel.reshape(-1, 3).copy()
        c[:, 1:] += base
        celdas.append(c.ravel())
        todos.append(pts)
        colores_.append(col)
        base += len(pts)
    if not todos:
        return None
    malla = pv.PolyData(np.vstack(todos))
    malla.lines = np.concatenate(celdas)
    malla.point_data["color"] = np.vstack(colores_)
    return malla


class Escena:
    """Toro, civilizaciones y señales de un episodio; se redibuja turno a turno."""

    def __init__(self, ep: dict, tema: str, zona: bool, tam=(1280, 900), fuera=True):
        self.ep, self.col, self.zona = ep, colores(tema), zona
        self.pl = pv.Plotter(off_screen=fuera, window_size=list(tam))
        self.pl.set_background(self.col["void"])
        self.pl.enable_anti_aliasing("ssaa")
        self._estatico()

    def _estatico(self) -> None:
        """Superficie del toro y su retícula de meridianos y paralelos."""
        c = self.col
        superficie = pv.ParametricTorus(ringradius=GRANDE, crosssectionradius=PEQUENO * 0.975)
        self.pl.add_mesh(
            superficie,
            color=c["raised"],
            smooth_shading=True,
            specular=0.35,
            specular_power=18,
            name="toro",
        )
        guia = mezcla(c["line-strong"], c["surface"], 0.55)
        partes = []
        for k in range(20):
            t = k / 20
            for fija in ("x", "y"):
                s = np.linspace(0, 1, 121)
                x, y = (np.full_like(s, t), s) if fija == "x" else (s, np.full_like(s, t))
                pts = toro(x, y, -0.004)
                cel = np.array([[2, i, i + 1] for i in range(len(pts) - 1)]).ravel()
                partes.append((pts, cel, np.tile(guia, (len(pts), 1))))
        reticula = lineas(partes)
        self.pl.add_mesh(
            reticula,
            scalars="color",
            rgb=True,
            line_width=2,
            render_lines_as_tubes=True,
            name="reticula",
        )

    def _quitar(self, nombre: str) -> None:
        """Retira un actor dinámico si existe."""
        if nombre in self.pl.renderer.actors:
            self.pl.remove_actor(nombre)

    def dibujar(self, t: int, giro: float | None = None) -> None:
        """Dibuja el estado del turno t."""
        ep, c, pl = self.ep, self.col, self.pl
        t = max(0, min(int(t), ep["steps"]))
        fondo = c["surface"]
        # estallido y zona
        flash = max((1 - (t - b) / 4 for b in ep["bursts"] if b <= t and t - b < 4), default=0.0)
        zona = [ep["zone"]["x"], ep["zone"]["y"]]
        self._quitar("estallido")
        if flash > 0:
            rho = np.linspace(0, ep["zone"]["r"], 10)
            tha = np.linspace(0, 2 * np.pi, 72)
            gx = (zona[0] + rho[:, None] * np.cos(tha)[None, :]) % 1.0
            gy = (zona[1] + rho[:, None] * np.sin(tha)[None, :]) % 1.0
            p = toro(gx, gy, 0.012)
            cuadricula = pv.StructuredGrid(p[..., 0], p[..., 1], p[..., 2])
            pl.add_mesh(cuadricula, color=c["burst"], opacity=0.5 * flash, name="estallido")
        partes = []
        if self.zona:
            partes.append(anillo(zona, ep["zone"]["r"], mezcla(c["burst"], fondo, 0.55), 0.015))
        # señales
        colores_msg = [c["muted"], c["symbol"], c["share"], c["lie"]]
        for s in ep["signals"]:
            if s["t"] > t:
                continue
            radio = ep["c"] * (t - s["t"])
            if not 0 < radio <= 1.0:
                continue
            alfa = (0.18 if s["msg"] == 0 else 1.0) * (1 - radio)
            partes.append(
                anillo(ep["pos"][s["o"]], radio, mezcla(colores_msg[s["msg"]], fondo, alfa), 0.02)
            )
        # proyectiles
        for q in ep["projectiles"]:
            if q["t"] > t:
                continue
            o = np.array(ep["pos"][q["o"]])
            d = ((np.array([q["tx"], q["ty"]]) - o + 0.5) % 1.0) - 0.5
            largo = max(float(np.hypot(*d)), 1e-6)
            f = min(ep["v"] * (t - q["t"]) / largo, 1.0)
            if f < 1.0:
                punta = (o + d * f) % 1.0
                esfera = pv.Sphere(radius=0.07, center=toro(punta[0], punta[1], 0.1))
                pl.add_mesh(
                    esfera, color=c[f"civ-{q['o'] % 8 + 1}"], name=f"proyectil{q['t']}_{q['o']}"
                )
        self._quitar("senales")
        malla = lineas(partes)
        if malla is not None:
            pl.add_mesh(
                malla,
                scalars="color",
                rgb=True,
                line_width=5,
                render_lines_as_tubes=True,
                name="senales",
            )
        # civilizaciones
        k0 = min(t, ep["steps"] - 1)
        for i in range(ep["n"]):
            x, y = ep["pos"][i]
            vivo = ep["alive"][t][i] == 1
            rec = max(ep["res"][t][i], 0.0)
            radio = (0.10 + 0.07 * np.log1p(rec)) * (1.0 if vivo else 0.6)
            centro = toro(x, y, radio)
            color = c[f"civ-{i % 8 + 1}"] if vivo else c["dead"]
            pl.add_point_labels(
                [toro(x, y, 2 * radio + 0.3)],
                [f"C{i + 1}  {rec:.1f}"],
                font_size=14,
                text_color=c["ink"],
                shape=None,
                show_points=False,
                always_visible=True,
                name=f"rotulo{i}",
            )
            pl.add_mesh(
                pv.Sphere(radius=radio, center=centro, theta_resolution=40, phi_resolution=40),
                color=color,
                smooth_shading=True,
                name=f"civ{i}",
            )
            self._quitar(f"escudo{i}")
            if vivo:
                if ep["refuge"][t][i] == 1:
                    pl.add_mesh(
                        pv.Sphere(radius=radio * 2.1, center=centro),
                        color=c["refuge"],
                        opacity=0.28,
                        name=f"escudo{i}",
                    )
        # texto
        vivos = sum(ep["alive"][t])
        lineas_txt = [f"turno {t}/{ep['steps']}  ·  vivos {vivos}/{ep['n']}"]
        for i in range(ep["n"]):
            kind = ep["kind"][k0][i]
            nombre = ACCIONES[kind] if kind >= 0 else "—"
            if ep.get("deception") and kind == 3 and ep["act"][k0][i] == 2 + ep["n_symbols"]:
                nombre = "mentir"
            estado = "vivo" if ep["alive"][t][i] else "muerto"
            lineas_txt.append(f"C{i + 1}  {estado:<6}  r={ep['res'][t][i]:.2f}  {nombre}")
        pl.add_text(
            "\n".join(lineas_txt), position="upper_left", font_size=11, color=c["ink"], name="hud"
        )
        titulo = re.sub(
            r"red:.*?runs[\\/]([^\\/]+?)(?:-[0-9a-f]{10})?[\\/](\d+)[\\/]checkpoints[\\/]final\.msgpack",
            r"red \1 (seed \2)",
            ep.get("policy", "").replace("fijo:", ""),
        )
        pl.add_text(
            titulo,
            position="lower_left",
            font_size=9,
            color=c["muted"],
            name="titulo",
        )
        self._camara(t if giro is None else giro)

    def _camara(self, t: float) -> None:
        """Cámara en órbita lenta alrededor del toro."""
        az = np.deg2rad(35 + 0.5 * t)
        self.pl.camera_position = [
            (12.5 * np.cos(az), 12.5 * np.sin(az), 8.5),
            (0, 0, -0.3),
            (0, 0, 1),
        ]


def cargar(ruta: Path) -> dict:
    """Episodio en el JSON que escribe viz.exportar_json o la demo."""
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def main() -> None:
    """Imagen, animación o ventana interactiva de un episodio."""
    p = argparse.ArgumentParser()
    p.add_argument("--episodio", required=True, type=Path)
    p.add_argument("--out", type=Path, help="imagen .png o animación .gif/.mp4; sin esto, ventana")
    p.add_argument("--turno", type=int, default=0, help="turno de la imagen")
    p.add_argument("--tema", choices=("noche", "carta"), default="noche")
    p.add_argument("--cada", type=int, default=2, help="una imagen cada N turnos en la animación")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--giro", action="store_true", help="gira la cámara; si no, queda fija")
    p.add_argument("--zona", action="store_true", help="muestra la zona oculta de la anomalía")
    args = p.parse_args()
    ep = cargar(args.episodio)
    esc = Escena(ep, args.tema, args.zona, fuera=args.out is not None)
    if args.out is None:
        estado = {"t": 0, "juega": True}
        esc.dibujar(0)
        esc.pl.add_slider_widget(
            lambda v: (estado.update(t=int(v)), esc.dibujar(int(v))),
            [0, ep["steps"]],
            value=0,
            title="turno",
            color=esc.col["sun"],
        )
        esc.pl.add_key_event("space", lambda: estado.update(juega=not estado["juega"]))

        def avanzar(_=None) -> None:
            """Avanza un turno si la reproducción está activa."""
            if estado["juega"] and estado["t"] < ep["steps"]:
                estado["t"] += 1
                esc.dibujar(estado["t"])

        esc.pl.add_callback(avanzar, interval=80)
        esc.pl.show()
    elif args.out.suffix.lower() == ".png":
        esc.dibujar(args.turno, None if args.giro else 0)
        esc.pl.screenshot(str(args.out))
        print(f"imagen: {args.out}")
    else:
        if args.out.suffix.lower() == ".gif":
            esc.pl.open_gif(str(args.out), fps=args.fps)
        else:
            esc.pl.open_movie(str(args.out), framerate=args.fps)
        for t in range(0, ep["steps"] + 1, args.cada):
            esc.dibujar(t, t if args.giro else 0)
            esc.pl.write_frame()
        esc.pl.close()
        print(f"animación: {args.out}")


if __name__ == "__main__":
    main()

"""Servidor local de la demo: corre el simulador y las políticas y devuelve episodios en JSON.

Sirve la página de demo/web y dos rutas: GET /api/opciones (redes, estrategias y valores del
mundo) y POST /api/simular (mundo, una política por civilización, semilla). Las redes se leen de
los checkpoints que encuentre en las carpetas indicadas. Cada mundo nuevo obliga a recompilar el
simulador (30 a 60 s en CPU); los siguientes episodios del mismo mundo son rápidos.

Uso: python -m demo.servidor [--puerto 8765] [--redes ../kout8 ../kout9]
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from darkforest.config import load_config
from darkforest.env import DarkForestEnv
from train import socios
from train.cruce import load_player, play
from viz.exportar_json import export_episode

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ.parent / "configs" / "pob_socios.yaml"
MUNDO = {  # nombre en la interfaz: (clave de configuración, tipo)
    "lethality": ("env.anomaly.lethality", float),
    "signal_speed": ("env.signal_speed", float),
    "noise_std": ("env.anomaly.noise_std", float),
    "cost_ref": ("env.cost_ref", float),
    "p_kill": ("env.p_kill", float),
    "deception": ("env.deception", bool),
    "costly_signals": ("env.costly_signals", bool),
    "n_civs": ("env.n_civs", int),
}
POR_DEFECTO = {
    "lethality": 0.5,
    "signal_speed": 0.22,
    "noise_std": 0.3,
    "cost_ref": 0.25,
    "p_kill": 0.75,
    "deception": False,
    "costly_signals": False,
    "n_civs": 2,
}
FIJAS = {
    "coop_real": "Coopera: comparte y usa lo que recibe",
    "recip": "Recíproca: deja de compartir con quien no responde",
    "prudente": "Prudente: solo acepta mensajes verificados",
    "mentiroso": "Mentiroso: comparte mediciones falsas",
    "solo": "Solo: se protege con sus propias mediciones",
    "ataque_solo": "Ataque: como solo, pero dispara al primer contacto",
    "ignorar": "Ignorar: solo crece",
}
LOCK = threading.Lock()
MOTORES: dict[str, tuple] = {}
POLITICAS: dict[str, tuple] = {}


def descubrir_redes(carpetas: list[Path], todas: bool = False) -> list[dict]:
    """Checkpoints finales de redes (corridas e imitaciones) con un nombre legible.

    Por defecto solo la seed 0 de cada corrida, para que la lista sea manejable.
    """
    redes = []
    for carpeta in carpetas:
        for ruta in sorted(carpeta.rglob("checkpoints/final.msgpack")):
            partes = ruta.parts
            if "runs" in partes and len(partes) >= 4:
                corrida = re.sub(r"-[0-9a-f]{10}$", "", partes[-4])
                if partes[-3] != "0" and not todas:
                    continue
                nombre = f"{corrida} (seed {partes[-3]})"
            else:
                nombre = f"imitación {partes[-3]}"
            redes.append({"id": f"red:{ruta}", "nombre": nombre})
    return redes


def etiqueta(spec: str) -> str:
    """Nombre legible de una política: la estrategia, o la corrida y la seed de una red."""
    if spec.startswith("fijo:"):
        return spec.split(":", 1)[1]
    partes = Path(spec.removeprefix("red:")).parts
    corrida = re.sub(r"-[0-9a-f]{10}$", "", partes[-4]) if len(partes) >= 4 else partes[-1]
    return f"red {corrida} (seed {partes[-3]})" if len(partes) >= 4 else corrida


def politica_para_cliente(spec: str) -> str:
    """Estrategias fijas con su prefijo (el cliente asigna el estilo); redes con su etiqueta."""
    return spec if spec.startswith("fijo:") else etiqueta(spec)


def motor(overrides: dict):
    """Entorno y funciones compiladas de un mundo; se reutilizan entre episodios."""
    clave = json.dumps(overrides, sort_keys=True)
    if clave not in MOTORES:
        cfg = load_config(
            BASE, {"env.anomaly.radius": 0.45, "env.detection_growth": 0.05, **overrides}
        )
        env = DarkForestEnv(cfg.env)
        reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
        MOTORES[clave] = (cfg, env, reset, step)
    return MOTORES[clave]


def jugador(spec: str, cfg, clave_mundo: str, voraz: bool):
    """Política de una civilización, con caché; las redes no dependen del mundo."""
    es_red = spec.startswith("red:")
    if es_red and voraz:
        spec += ":greedy"
    clave = spec if es_red else f"{spec}@{clave_mundo}"
    if clave not in POLITICAS:
        POLITICAS[clave] = load_player(spec, cfg)
    return POLITICAS[clave]


def simular(mundo: dict, politicas: list[str], seed: int, voraz: bool, promediar: int = 0) -> dict:
    """Un episodio completo con las políticas dadas, exportado para la interfaz."""
    overrides = {
        MUNDO[k][0]: MUNDO[k][1](v) for k, v in {**POR_DEFECTO, **mundo}.items() if k in MUNDO
    }
    if len(politicas) == 1:  # una sola política: la usan todas las civilizaciones
        politicas = [politicas[0], politicas[0]]
    with LOCK:
        cfg, env, reset, step = motor(overrides)
        clave_mundo = json.dumps(overrides, sort_keys=True)
        n = env.n_agents
        # la política A juega la civilización 1 y la B todas las demás
        a, b = (jugador(p, cfg, clave_mundo, voraz) for p in politicas[:2])
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = reset(jax.random.split(reset_key, 1))
        aux = socios.initial_auxiliar(1, n)
        carry_a, carry_b = a[2](1), b[2](1)
        estados, observaciones, infos = [state], [obs], []
        acciones, recompensas, hechos = [], [], []
        for _ in range(cfg.env.max_steps):
            key, key_a, key_b, step_key = jax.random.split(key, 4)
            carry_a, act_a = a[1](key_a, carry_a, state, obs, aux)
            carry_b, act_b = b[1](key_b, carry_b, state, obs, aux)
            actions = act_b.astype(jnp.int32).at[:, 0].set(act_a[:, 0].astype(jnp.int32))
            out = step(jax.random.split(step_key, 1), state, actions)
            aux = socios.update_auxiliar(cfg, aux, state, out.state, actions)
            acciones.append(actions)
            recompensas.append(out.reward)
            hechos.append(out.done)
            infos.append(out.info)
            state, obs = out.state, out.obs
            estados.append(state)
            observaciones.append(obs)
            if bool(out.done.all()):
                break
    traj = {}
    for valores, prefijo in ((estados, ""), (observaciones, "obs_"), (infos, "info_")):
        for campo in fields(valores[0]):
            if getattr(valores[0], campo.name) is not None:
                traj[prefijo + campo.name] = np.stack(
                    [np.asarray(getattr(v, campo.name))[0] for v in valores]
                )
    traj["actions"] = np.stack([np.asarray(x)[0] for x in acciones])
    traj["rewards"] = np.stack([np.asarray(x)[0] for x in recompensas])
    traj["done"] = np.stack([np.asarray(x)[0] for x in hechos])
    metadata = {
        "n_symbols": cfg.env.n_symbols,
        "signal_speed": cfg.env.signal_speed,
        "proj_speed": cfg.env.proj_speed,
        "policy": " contra ".join(etiqueta(p) for p in politicas[:2])
        if n == 2
        else f"{etiqueta(politicas[0])} (civilización 1) y {etiqueta(politicas[1])} (el resto)",
        "seed": seed,
    }
    nombres = [politica_para_cliente(politicas[0])] + [politica_para_cliente(politicas[1])] * (
        n - 1
    )
    datos = export_episode(traj, metadata, deception=cfg.env.deception, politicas=nombres)
    tipos = traj["info_executed_kind"]
    datos["resumen"] = {
        "retorno": np.round(traj["rewards"].sum(axis=0), 3).tolist(),
        "supervivencia": traj["alive"][-1].astype(int).tolist(),
        "mensajes": int((tipos == 3).sum() + (tipos == 4).sum()),
        "ataques": int((tipos == 5).sum()),
        "refugios": int((tipos == 6).sum()),
        "mentiras": sum(1 for s in datos["signals"] if s["msg"] == 3),
        "muertes_por_ataque": int((traj["info_death_cause"] == 2).sum()),
    }
    datos["mundo"] = {**POR_DEFECTO, **mundo}
    if promediar and n == 2:
        with LOCK:
            a, b = (jugador(p, cfg, clave_mundo, voraz) for p in politicas[:2])
            media = play(cfg, a, b, promediar, seed + 1000)
        datos["promedio"] = {
            "episodios": promediar,
            "retorno": [round(x, 2) for x in media["retorno"]],
            "supervivencia": [round(x, 2) for x in media["supervivencia"]],
        }
    return datos


class Manejador(BaseHTTPRequestHandler):
    """Rutas de la demo."""

    redes: list[dict] = []

    def _json(self, datos, codigo: int = 200) -> None:
        cuerpo = json.dumps(datos, separators=(",", ":")).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self) -> None:  # noqa: N802
        """Opciones de la demo y archivos estáticos."""
        if self.path == "/api/opciones":
            self._json(
                {
                    "fijas": [{"id": f"fijo:{k}", "nombre": v} for k, v in FIJAS.items()],
                    "redes": self.redes,
                    "mundo": POR_DEFECTO,
                }
            )
            return
        nombre = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        ruta = (RAIZ / "web" / nombre).resolve()
        if not str(ruta).startswith(str(RAIZ / "web")) or not ruta.is_file():
            self.send_error(404)
            return
        tipos = {
            ".html": "text/html",
            ".js": "text/javascript",
            ".css": "text/css",
            ".json": "application/json",
        }
        cuerpo = ruta.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{tipos.get(ruta.suffix, 'text/plain')}; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_POST(self) -> None:  # noqa: N802
        """Simula un episodio."""
        if self.path != "/api/simular":
            self.send_error(404)
            return
        try:
            largo = int(self.headers.get("Content-Length", 0))
            pedido = json.loads(self.rfile.read(largo))
            self._json(
                simular(
                    pedido.get("mundo", {}),
                    pedido["politicas"],
                    int(pedido.get("seed", 0)),
                    bool(pedido.get("voraz", True)),
                    int(pedido.get("promediar", 0)),
                )
            )
        except Exception as error:  # la interfaz muestra el motivo
            self._json({"error": f"{type(error).__name__}: {error}"}, 400)

    def log_message(self, formato, *args) -> None:
        """Silencia el registro de cada petición."""


def main() -> None:
    """Arranca el servidor local."""
    p = argparse.ArgumentParser()
    p.add_argument("--puerto", type=int, default=8765)
    p.add_argument("--redes", nargs="*", type=Path, default=[])
    p.add_argument("--todas-las-seeds", action="store_true")
    args = p.parse_args()
    carpetas = args.redes or sorted(RAIZ.parent.parent.glob("kout*"))
    Manejador.redes = descubrir_redes(carpetas, args.todas_las_seeds)
    print(f"{len(Manejador.redes)} redes encontradas. http://localhost:{args.puerto}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador).serve_forever()


if __name__ == "__main__":
    main()

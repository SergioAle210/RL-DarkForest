"""Demo: exportación de episodios, sistema de diseño y visor 3D."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from darkforest.config import EnvConfig
from darkforest.env import DarkForestEnv
from darkforest.policies import RandomPolicy
from demo.diseno import generar_css
from demo.servidor import descubrir_redes, etiqueta, politica_para_cliente
from train.episodes import collect_episode
from viz.exportar_json import export_episode, extraer_muertes

TOKENS = Path(__file__).resolve().parent.parent / "demo" / "diseno" / "project" / "tokens.json"


def _episodio(deception: bool = False) -> dict:
    cfg = EnvConfig(n_civs=2, max_steps=8, deception=deception)
    traj = collect_episode(DarkForestEnv(cfg), RandomPolicy(cfg.n_discrete), jax.random.PRNGKey(0))
    meta = {
        "n_symbols": cfg.n_symbols,
        "signal_speed": cfg.signal_speed,
        "proj_speed": cfg.proj_speed,
    }
    return export_episode(traj, {**meta, "policy": "prueba", "seed": 0}, deception=deception)


def test_exported_episode_has_the_fields_the_viewers_read():
    """El JSON trae posiciones, estados por turno y eventos con las formas de los visores."""
    ep = _episodio()
    assert ep["n"] == 2 and len(ep["pos"]) == 2
    assert len(ep["alive"]) == ep["steps"] + 1 and len(ep["kind"]) == ep["steps"]
    assert len(ep["act"]) == ep["steps"] and "n_symbols" in ep
    assert set(ep["zone"]) == {"x", "y", "r"}
    json.dumps(ep)  # serializable


def test_policy_labels_hide_checkpoint_paths():
    """Una estrategia se nombra por su clave y una red por su corrida y seed."""
    assert etiqueta("fijo:coop_real") == "coop_real"
    ruta = "red:/k/runs/ft_l0p5_c0p22-301475f86c/2/checkpoints/final.msgpack"
    assert etiqueta(ruta) == "red ft_l0p5_c0p22 (seed 2)"


def test_networks_are_found_by_checkpoint(tmp_path):
    """Solo se listan las seeds 0 salvo que se pidan todas."""
    for seed in (0, 1):
        carpeta = tmp_path / "runs" / "ft_a-0123456789" / str(seed) / "checkpoints"
        carpeta.mkdir(parents=True)
        (carpeta / "final.msgpack").write_bytes(b"x")
    assert [r["nombre"] for r in descubrir_redes([tmp_path])] == ["ft_a (seed 0)"]
    assert len(descubrir_redes([tmp_path], todas=True)) == 2


def _luminancia(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    canales = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lineal = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in canales]
    return 0.2126 * lineal[0] + 0.7152 * lineal[1] + 0.0722 * lineal[2]


def _contraste(a: str, b: str) -> float:
    claro, oscuro = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (claro + 0.05) / (oscuro + 0.05)


@pytest.mark.parametrize("tema", ["noche", "carta"])
def test_design_tokens_meet_the_contrast_floor_in_both_themes(tema):
    """Texto 4.5:1 y marcas 3:1 sobre los fondos que declaran sus notas, en cada tema."""
    tokens = json.loads(TOKENS.read_text(encoding="utf-8"))
    c = {t["name"]: t["value"][tema] for t in tokens["color"]["tokens"]}
    for texto in ("ink", "muted", "sun"):
        for fondo in ("void", "surface", "raised"):
            assert _contraste(c[texto], c[fondo]) >= 4.5, (texto, fondo)
    assert _contraste(c["on-sun"], c["sun"]) >= 4.5
    for marca in ("line-strong", "focus", "share", "symbol", "lie", "burst", "refuge"):
        for fondo in ("surface", "raised"):
            assert _contraste(c[marca], c[fondo]) >= 3.0, (marca, fondo)


def test_generated_css_declares_every_token_in_every_theme():
    """El CSS de la demo trae las variables de color de ambos temas y una clase por estilo."""
    tokens = json.loads(TOKENS.read_text(encoding="utf-8"))
    css = generar_css.compilar(tokens)
    assert ':root, [data-theme="noche"]' in css and '[data-theme="carta"]' in css
    assert css.count("--civ-8:") == 2 and ".title-xl {" in css and "--font-mono:" in css


def test_3d_viewer_renders_an_image(tmp_path):
    """La escena 3D dibuja un turno y guarda una imagen no vacía."""
    pytest.importorskip("pyvista")
    from demo.visor3d import Escena

    ep = _episodio()
    escena = Escena(ep, "noche", True, tam=(320, 240))
    escena.dibujar(3, 0)
    salida = tmp_path / "toro.png"
    escena.pl.screenshot(str(salida))
    escena.pl.close()
    assert salida.stat().st_size > 1000


def test_demo_simulates_a_liar_among_cooperators_with_four_civilizations():
    """Con 4 civilizaciones la política A juega la civilización 1 y la B todas las demás."""
    from demo.servidor import simular

    mundo = {"n_civs": 4, "deception": True}
    datos = simular(mundo, ["fijo:mentiroso", "fijo:coop_real"], 0, True)
    assert datos["n"] == 4 and len(datos["resumen"]["retorno"]) == 4
    assert datos["resumen"]["mentiras"] > 0
    assert "civilización 1" in datos["policy"]


def test_episode_exports_emision_muertes_y_politicas():
    """El episodio trae la emisión por turno, las muertes con su forma y las políticas dadas."""
    ep = _episodio()
    assert len(ep["emision"]) == ep["steps"] + 1
    assert all(len(f) == ep["n"] and set(f) <= {0, 1, 2} for f in ep["emision"])
    assert isinstance(ep["muertes"], list)
    for m in ep["muertes"]:
        assert set(m) == {"t", "civ", "causa", "atacante"}
        assert m["causa"] in {"anomalia", "ataque"}
    assert ep["politicas"] == []
    cfg = EnvConfig(n_civs=2, max_steps=4)
    traj = collect_episode(DarkForestEnv(cfg), RandomPolicy(cfg.n_discrete), jax.random.PRNGKey(0))
    meta = {"n_symbols": cfg.n_symbols, "signal_speed": 1.0, "proj_speed": 1.0}
    nombres = ["fijo:coop_real", "red x (seed 0)"]
    assert export_episode(traj, meta, politicas=nombres)["politicas"] == nombres


def test_muertes_primera_vez_por_civilizacion():
    """Anomalía y ataque se registran una sola vez, en el turno siguiente a la transición."""
    causa = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
    atacante = np.array([[-1, -1, -1], [-1, -1, -1], [-1, 0, -1], [-1, 0, -1]])
    assert extraer_muertes(causa, atacante) == [
        {"t": 2, "civ": 0, "causa": "anomalia", "atacante": -1},
        {"t": 3, "civ": 1, "causa": "ataque", "atacante": 0},
    ]
    assert extraer_muertes(np.zeros((3, 2), dtype=int), -np.ones((3, 2), dtype=int)) == []


def test_politica_para_cliente_conserva_prefijo_fijo_y_etiqueta_redes():
    """Las fijas llevan «fijo:»; las redes, su etiqueta legible."""
    assert politica_para_cliente("fijo:coop_real") == "fijo:coop_real"
    ruta = "red:/k/runs/ft_l0p5_c0p22-301475f86c/0/checkpoints/final.msgpack"
    assert politica_para_cliente(ruta) == "red ft_l0p5_c0p22 (seed 0)"

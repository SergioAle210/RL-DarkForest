from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from darkforest.config import (
    AnomalyConfig,
    Config,
    EnvConfig,
    LoggingConfig,
    SelfPlayConfig,
    TrainConfig,
    apply_overrides,
    load_config,
    to_dict,
)
from train.sweep import expand_grid

CONFIGS = Path(__file__).parents[1] / "configs"
COUNTS = (
    "env.max_steps",
    "env.max_detections",
    "env.n_symbols",
    "env.anomaly.period",
    "train.n_envs",
    "train.total_steps",
    "train.rollout_len",
    "train.gru_hidden",
    "train.self_play.snapshot_every",
)
PROBABILITIES = (
    "env.p_kill",
    "env.anomaly.lethality",
    "train.gamma",
    "train.gae_lambda",
    "train.self_play.rho",
)
NONNEGATIVE = (
    "env.growth",
    "env.cost_obs",
    "env.cost_tx",
    "env.cost_atk",
    "env.cost_ref",
    "env.anomaly.noise_std",
    "train.ent_coef",
    "train.vf_coef",
)
POSITIVE = ("env.signal_speed", "env.proj_speed", "train.lr", "train.clip_eps")
BOOLEANS = (
    "env.costly_signals",
    "env.channel_enabled",
    "env.shuffle_messages",
    "env.anomaly.interfere",
)


def _replace_field(cfg, path: str, value):
    """Construye una configuración modificando una dataclass anidada."""
    name, *rest = path.split(".", 1)
    updated = _replace_field(getattr(cfg, name), rest[0], value) if rest else value
    return replace(cfg, **{name: updated})


@pytest.fixture(params=("constructor", "yaml"))
def build_config(request, tmp_path):
    """Comprueba la misma validación en constructores y carga YAML."""
    path = tmp_path / "config.yaml"
    path.write_text("{}", encoding="utf-8")
    if request.param == "constructor":
        return lambda field, value: _replace_field(Config(), field, value)
    return lambda field, value: load_config(path, {field: value})


def test_base_loads():
    """La configuración base conserva los valores acordados."""
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.env.n_civs == 4
    assert cfg.env.proj_speed <= cfg.env.signal_speed


def test_extends_and_overrides():
    """La herencia y los overrides conservan los demás campos."""
    cfg = load_config(CONFIGS / "h1_sanity.yaml", {"env.signal_speed": 0.5})
    assert cfg.env.anomaly.lethality == 0.0
    assert cfg.env.n_civs == 2
    assert cfg.env.signal_speed == 0.5
    assert cfg.env.anomaly.period == 20
    assert cfg.train.total_steps == 5_000_000


@pytest.mark.parametrize(
    "field", ("nope", "env.nope", "env.anomaly.nope", "train.self_play.nope", "logging.nope")
)
def test_unknown_key_rejected(field):
    """Las claves desconocidas fallan también en secciones anidadas."""
    with pytest.raises(KeyError):
        load_config(CONFIGS / "base.yaml", {field: 1})


@pytest.mark.parametrize("field", COUNTS)
@pytest.mark.parametrize("value", (0, -1))
def test_counts_must_be_positive(build_config, field, value):
    """Los tamaños estáticos y las frecuencias deben ser positivos."""
    with pytest.raises(ValueError):
        build_config(field, value)


@pytest.mark.parametrize("field", (*COUNTS, "env.n_civs", "train.seed"))
@pytest.mark.parametrize("value", (True, 1.5, "2"))
def test_counts_require_integers(build_config, field, value):
    """Un booleano o un número decimal no es un conteo válido."""
    with pytest.raises(TypeError):
        build_config(field, value)


@pytest.mark.parametrize("value", (1, 3, 5, 7, 10))
def test_civilization_count_is_supported(build_config, value):
    """El entorno permite los tamaños definidos en la propuesta."""
    with pytest.raises(ValueError):
        build_config("env.n_civs", value)


@pytest.mark.parametrize("value", (2, 4, 6, 8))
def test_supported_civilization_counts(build_config, value):
    """Todos los tamaños previstos se pueden construir."""
    assert build_config("env.n_civs", value).env.n_civs == value


@pytest.mark.parametrize("value", (-1, 2**32))
def test_seed_cannot_wrap_around(build_config, value):
    """Una seed fuera de 32 bits no debe aliasar otra corrida."""
    with pytest.raises(ValueError):
        build_config("train.seed", value)


@pytest.mark.parametrize("value", (0, 2**32 - 1))
def test_seed_boundaries(build_config, value):
    """Los extremos del dominio de seeds son válidos."""
    assert build_config("train.seed", value).train.seed == value


@pytest.mark.parametrize("field", PROBABILITIES)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_probabilities_stay_in_unit_interval(build_config, field, value):
    """Las probabilidades y descuentos permanecen entre cero y uno."""
    with pytest.raises(ValueError):
        build_config(field, value)


@pytest.mark.parametrize("field", PROBABILITIES)
@pytest.mark.parametrize("value", (0, 1))
def test_probability_boundaries(build_config, field, value):
    """Se permiten probabilidades deterministas en los extremos."""
    build_config(field, value)


@pytest.mark.parametrize("field", NONNEGATIVE)
def test_nonnegative_parameters(build_config, field):
    """Costes, ruido y coeficientes admiten cero, pero no negativos."""
    build_config(field, 0)
    with pytest.raises(ValueError):
        build_config(field, -0.01)


@pytest.mark.parametrize("field", POSITIVE)
@pytest.mark.parametrize("value", (0, -0.01))
def test_strictly_positive_parameters(build_config, field, value):
    """Velocidades y parámetros de actualización deben superar cero."""
    with pytest.raises(ValueError):
        build_config(field, value)


@pytest.mark.parametrize("field", (*PROBABILITIES, *NONNEGATIVE, *POSITIVE))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_parameters_are_rejected(build_config, field, value):
    """Ningún parámetro numérico puede propagar NaN o infinito."""
    with pytest.raises(ValueError):
        build_config(field, value)


@pytest.mark.parametrize("field", (*PROBABILITIES, *NONNEGATIVE, *POSITIVE))
@pytest.mark.parametrize("value", (True, "0.1", None))
def test_numeric_parameters_require_numbers(build_config, field, value):
    """No se convierten silenciosamente cadenas ni booleanos."""
    with pytest.raises(TypeError):
        build_config(field, value)


@pytest.mark.parametrize("field", BOOLEANS)
@pytest.mark.parametrize("value", (0, 1, "false"))
def test_switches_require_booleans(build_config, field, value):
    """Las opciones de activación exigen un booleano real."""
    with pytest.raises(TypeError):
        build_config(field, value)


def test_projectiles_do_not_overtake_signals(build_config):
    """La validación física rechaza ambas maneras de invertir velocidades."""
    with pytest.raises(ValueError, match="proj_speed"):
        build_config("env.signal_speed", 0.01)
    with pytest.raises(ValueError, match="proj_speed"):
        build_config("env.proj_speed", 0.2)
    assert build_config("env.proj_speed", 0.1).env.proj_speed == 0.1


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("train.algo", "ppo", ValueError),
        ("train.algo", 1, TypeError),
        ("logging.backend", "csv", ValueError),
        ("logging.backend", False, TypeError),
        ("logging.out_dir", "", ValueError),
        ("logging.out_dir", "  ", ValueError),
        ("logging.out_dir", "runs\0invalid", ValueError),
        ("logging.out_dir", 1, TypeError),
    ),
)
def test_text_options(build_config, field, value, error):
    """Las opciones textuales no admiten valores ambiguos."""
    with pytest.raises(error):
        build_config(field, value)


@pytest.mark.parametrize("backend", ("tensorboard", "wandb"))
def test_supported_logging_backends(build_config, backend):
    """Ambos backends previstos siguen siendo configurables."""
    assert build_config("logging.backend", backend).logging.backend == backend


def test_logging_output_cannot_be_file(build_config, tmp_path):
    """Una salida existente debe ser un directorio, no un archivo."""
    path = tmp_path / "output.txt"
    path.write_text("contenido", encoding="utf-8")
    with pytest.raises(ValueError, match="archivo"):
        build_config("logging.out_dir", str(path))


def test_logging_output_does_not_create_directories(build_config, tmp_path):
    """La validación permite destinos nuevos sin crearlos."""
    build_config("logging.out_dir", str(tmp_path))
    path = tmp_path / "future_runs"
    assert build_config("logging.out_dir", str(path)).logging.out_dir == str(path)
    assert not path.exists()


@pytest.mark.parametrize(
    ("cls", "field"),
    (
        (Config, "env"),
        (Config, "train"),
        (Config, "logging"),
        (EnvConfig, "anomaly"),
        (TrainConfig, "self_play"),
    ),
)
@pytest.mark.parametrize("value", ({}, None, "config"))
def test_constructors_require_typed_nested_sections(cls, field, value):
    """Los constructores reciben dataclasses, no diccionarios sin validar."""
    with pytest.raises(TypeError):
        cls(**{field: value})


def test_typed_nested_sections_are_preserved():
    """Cada sección acepta su dataclass y conserva sus valores."""
    anomaly = AnomalyConfig(lethality=0)
    self_play = SelfPlayConfig(rho=1)
    cfg = Config(
        env=EnvConfig(anomaly=anomaly),
        train=TrainConfig(self_play=self_play),
        logging=LoggingConfig(backend="wandb"),
    )
    assert cfg.env.anomaly is anomaly
    assert cfg.train.self_play is self_play
    assert cfg.logging.backend == "wandb"


@pytest.mark.parametrize("field", ("env", "train", "logging", "env.anomaly", "train.self_play"))
@pytest.mark.parametrize("value", (None, [], "config"))
def test_yaml_sections_must_be_mappings(tmp_path, field, value):
    """Las secciones YAML requieren mapas para construir dataclasses."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(apply_overrides({}, {field: value})), encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)


@pytest.mark.parametrize("content", ("[]", "false", "0", "texto"))
def test_yaml_root_must_be_mapping(tmp_path, content):
    """Un escalar o una lista no representa una configuración."""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)


@pytest.mark.parametrize("content", ("", "{}"))
def test_empty_yaml_uses_defaults(tmp_path, content):
    """Un archivo vacío o un mapa vacío usa los valores por defecto."""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    assert load_config(path) == Config()


def test_override_inputs_are_not_mutated():
    """Los cambios resueltos no alteran la base ni los overrides originales."""
    raw = {"env": {"anomaly": {"lethality": 0.5}}}
    overrides = {"env.anomaly.lethality": 0, "logging": {"out_dir": "salidas"}}
    result = apply_overrides(raw, overrides)
    result["logging"]["out_dir"] = "otro"
    assert raw == {"env": {"anomaly": {"lethality": 0.5}}}
    assert overrides["logging"]["out_dir"] == "salidas"
    assert result["env"]["anomaly"]["lethality"] == 0


@pytest.mark.parametrize("field", ("", ".env", "env.", "env..growth"))
def test_override_paths_cannot_have_empty_segments(field):
    """Un segmento vacío no se interpreta como nombre de campo."""
    with pytest.raises(KeyError):
        apply_overrides({}, {field: 1})


def test_override_cannot_descend_into_scalar():
    """Un override anidado no puede atravesar un escalar."""
    with pytest.raises(TypeError):
        apply_overrides({"env": {"growth": 0.05}}, {"env.growth.value": 1})


@pytest.mark.parametrize(("value", "error"), ((True, TypeError), ([], TypeError), ("", ValueError)))
def test_extends_requires_nonempty_path(tmp_path, value, error):
    """La referencia a la configuración base debe ser una ruta textual."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"extends": value}), encoding="utf-8")
    with pytest.raises(error):
        load_config(path)


def test_resolved_config_round_trip(tmp_path):
    """La configuración resuelta se puede guardar y recargar sin pérdidas."""
    cfg = load_config(CONFIGS / "h1_sanity.yaml", {"train.seed": 12})
    path = tmp_path / "resolved.yaml"
    resolved = to_dict(cfg)
    path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    assert load_config(path) == cfg
    resolved["env"]["n_civs"] = 8
    assert cfg.env.n_civs == 2


@pytest.mark.parametrize(("symbols", "detections"), ((1, 1), (4, 4), (7, 3)))
def test_discrete_actions_include_parameterized_options(symbols, detections):
    """El conteo distingue familias de acción de sus opciones discretas."""
    cfg = EnvConfig(n_symbols=symbols, max_detections=detections)
    assert cfg.n_discrete == 5 + symbols + detections


def test_grid_expansion():
    """El barrido conserva el producto cartesiano de parámetros."""
    cells = expand_grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(cells) == 6

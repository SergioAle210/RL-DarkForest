"""Contratos del catálogo de acciones, sin ejecutar el simulador."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.config import EnvConfig
from darkforest.env import DarkForestEnv

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64


@pytest.mark.parametrize(("symbols", "detections"), [(1, 1), (2, 3), (4, 4), (7, 2)])
def test_every_action_has_the_expected_family_and_argument(symbols: int, detections: int) -> None:
    """Comprueba cada identificador contra el catálogo de sus familias."""
    cfg = EnvConfig(n_symbols=symbols, max_detections=detections)
    env = DarkForestEnv(cfg)
    expected = (
        [(0, -1), (1, -1), (2, -1)]
        + [(3, symbol) for symbol in range(symbols)]
        + [(4, -1)]
        + [(5, target) for target in range(detections)]
        + [(6, -1)]
    )
    kinds, arguments = env.decode_action(jnp.arange(len(expected)))

    assert env.n_actions == 7
    assert env.n_discrete == cfg.n_discrete == len(expected)
    np.testing.assert_array_equal(kinds, [kind for kind, _ in expected])
    np.testing.assert_array_equal(arguments, [argument for _, argument in expected])


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (-2, (-1, -1)),
        (-1, (-1, -1)),
        (0, (0, -1)),
        (1, (1, -1)),
        (2, (2, -1)),
        (3, (3, 0)),
        (6, (3, 3)),
        (7, (4, -1)),
        (8, (5, 0)),
        (11, (5, 3)),
        (12, (6, -1)),
        (13, (-1, -1)),
    ],
)
def test_scalar_boundaries(action: int, expected: tuple[int, int]) -> None:
    """Comprueba extremos de familias y los identificadores inválidos adyacentes."""
    env = DarkForestEnv(EnvConfig())
    result = env.decode_action(action)

    assert all(value.shape == () for value in result)
    assert tuple(int(value) for value in result) == expected


@pytest.mark.parametrize("action", [-(2**80), -(2**40), 2**40, 2**80])
def test_large_python_integers_are_invalid(action: int) -> None:
    """Valida los enteros Python antes de convertirlos al rango de JAX."""
    env = DarkForestEnv(EnvConfig())
    result = env.decode_action(action)

    assert tuple(int(value) for value in result) == (-1, -1)


def test_batches_preserve_the_input_shape() -> None:
    """Mantiene las dimensiones de un lote de acciones y sus centinelas."""
    env = DarkForestEnv(EnvConfig())
    actions = jnp.array([[0, 3, 7], [8, 12, 13]])
    kinds, arguments = jax.jit(env.decode_action)(actions)

    assert kinds.shape == arguments.shape == actions.shape
    np.testing.assert_array_equal(kinds, [[0, 3, 4], [5, 6, -1]])
    np.testing.assert_array_equal(arguments, [[-1, 0, -1], [0, -1, -1]])


def test_jit_and_vmap_match_eager_decoding() -> None:
    """Decodifica el catálogo completo y sus límites con las tres formas de ejecución."""
    env = DarkForestEnv(EnvConfig(n_symbols=2, max_detections=3))
    actions = jnp.arange(-1, env.n_discrete + 1)
    expected = env.decode_action(actions)

    for decoded in (jax.jit(env.decode_action)(actions), jax.vmap(env.decode_action)(actions)):
        for actual, reference in zip(decoded, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    "dtype", [jnp.int8, jnp.int16, jnp.int32, jnp.uint8, jnp.uint16, jnp.uint32]
)
def test_large_integer_values_remain_invalid(dtype: type) -> None:
    """Evita que una conversión convierta un identificador grande en una acción válida."""
    env = DarkForestEnv(EnvConfig())
    actions = jnp.array([0, 12, 13, jnp.iinfo(dtype).max], dtype=dtype)

    for decode in (env.decode_action, jax.jit(env.decode_action)):
        kinds, arguments = decode(actions)
        assert jnp.issubdtype(kinds.dtype, jnp.signedinteger)
        assert jnp.issubdtype(arguments.dtype, jnp.signedinteger)
        np.testing.assert_array_equal(kinds, [0, 6, -1, -1])
        np.testing.assert_array_equal(arguments, [-1, -1, -1, -1])


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.int16, jnp.int32])
def test_minimum_signed_integer_is_invalid(dtype: type) -> None:
    """Rechaza el mínimo del tipo sin desbordar al calcular argumentos."""
    env = DarkForestEnv(EnvConfig())
    action = jnp.array(jnp.iinfo(dtype).min, dtype=dtype)
    kinds, arguments = jax.jit(env.decode_action)(action)

    assert int(kinds) == int(arguments) == -1


def test_uint64_values_do_not_wrap_into_valid_actions() -> None:
    """Conserva como inválidos valores sin signo que perderían bits al reducirlos."""
    with enable_x64():
        env = DarkForestEnv(EnvConfig())
        actions = jnp.array([0, 2**32, 2**32 + 3, 2**64 - 1], dtype=jnp.uint64)
        kinds, arguments = jax.jit(env.decode_action)(actions)

        np.testing.assert_array_equal(kinds, [0, -1, -1, -1])
        np.testing.assert_array_equal(arguments, [-1, -1, -1, -1])
        assert jnp.issubdtype(kinds.dtype, jnp.signedinteger)


@pytest.mark.parametrize(("dtype", "symbols"), [(jnp.int8, 130), (jnp.uint8, 260)])
def test_small_input_dtype_can_address_a_larger_catalog(dtype: type, symbols: int) -> None:
    """La comparación no desborda si el catálogo supera el rango del entero de entrada."""
    env = DarkForestEnv(EnvConfig(n_symbols=symbols, max_detections=3))
    largest = jnp.iinfo(dtype).max
    actions = jnp.array([0, 3, largest - 1, largest], dtype=dtype)
    kinds, arguments = jax.jit(env.decode_action)(actions)

    np.testing.assert_array_equal(kinds, [0, 3, 3, 3])
    np.testing.assert_array_equal(arguments, [-1, 0, largest - 4, largest - 3])


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.float32, jnp.complex64])
def test_noninteger_arrays_are_rejected(dtype: type) -> None:
    """Rechaza tipos que no representan identificadores enteros, incluso bajo jit."""
    env = DarkForestEnv(EnvConfig())
    actions = jnp.array([0, 1], dtype=dtype)

    for decode in (env.decode_action, jax.jit(env.decode_action)):
        with pytest.raises(TypeError, match="tipo entero"):
            decode(actions)


@pytest.mark.parametrize("action", [True, 1.0, 1 + 0j, "crecer"])
def test_noninteger_scalars_are_rejected(action: object) -> None:
    """Devuelve un error en español para entradas escalares sin tipo entero."""
    env = DarkForestEnv(EnvConfig())

    with pytest.raises(TypeError, match="tipo entero"):
        env.decode_action(action)

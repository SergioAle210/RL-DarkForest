"""Prueba los pytrees con datos sintéticos, sin fijar dimensiones físicas del modelo."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.types import EnvState, Obs, StepInfo

SyntheticData = tuple[EnvState, Obs, StepInfo]


@pytest.fixture
def synthetic_data() -> SyntheticData:
    """Construye arrays pequeños cuyos tamaños solo pertenecen a estas pruebas."""
    state = EnvState(
        t=jnp.array(0),
        pos=jnp.array([[0.1, 0.2], [0.6, 0.7]]),
        resources=jnp.array([1.0, 2.0]),
        alive=jnp.array([True, True]),
        emission=jnp.array([0, 1]),
        refuge=jnp.array([False, False]),
        theta=jnp.zeros(3),
        last_burst=jnp.array(-1),
        last_meas=jnp.zeros((2, 1)),
        sig_origin=jnp.zeros(2, dtype=jnp.int32),
        sig_t=jnp.zeros(2, dtype=jnp.int32),
        sig_content=jnp.zeros((2, 2)),
        sig_valid=jnp.array([False, False]),
        prj_origin=jnp.zeros(2, dtype=jnp.int32),
        prj_target=jnp.zeros((2, 2)),
        prj_t=jnp.zeros(2, dtype=jnp.int32),
        prj_valid=jnp.array([False, False]),
    )
    obs = Obs(
        own=jnp.array([[1.0, 0.0], [2.0, 1.0]]),
        detections=jnp.zeros((2, 2, 3)),
        det_mask=jnp.array([[True, False], [False, False]]),
        own_meas=jnp.zeros((2, 1)),
        since_burst=jnp.zeros(2, dtype=jnp.int32),
        alive=jnp.array([True, True]),
        agent_id=jnp.arange(2),
    )
    info = StepInfo(
        death_cause=jnp.zeros(2, dtype=jnp.int32),
        attacker=jnp.full(2, -1, dtype=jnp.int32),
        preventive_attacks=jnp.array(0),
        shelters_correct=jnp.array(0),
        bursts=jnp.array(0),
    )
    return state, obs, info


@pytest.mark.parametrize("index", [0, 1, 2])
def test_structures_are_array_pytrees(synthetic_data: SyntheticData, index: int) -> None:
    """Comprueba que JAX puede separar y reconstruir cada contrato sin perder datos."""
    original = synthetic_data[index]
    leaves, structure = jax.tree_util.tree_flatten(original)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert type(restored) is type(original)
    for actual, expected in zip(jax.tree_util.tree_leaves(restored), leaves, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("index", "field"), [(0, "resources"), (1, "own"), (2, "bursts")])
def test_assignment_cannot_mutate_structures(
    synthetic_data: SyntheticData, index: int, field: str
) -> None:
    """Impide reasignar campos del estado, la observación y los diagnósticos."""
    value = synthetic_data[index]

    with pytest.raises(FrozenInstanceError):
        setattr(value, field, jnp.zeros_like(getattr(value, field)))


@pytest.mark.parametrize(("index", "field"), [(0, "resources"), (1, "own"), (2, "bursts")])
def test_replace_preserves_the_original(
    synthetic_data: SyntheticData, index: int, field: str
) -> None:
    """La actualización funcional genera otra estructura sin cambiar la anterior."""
    original = synthetic_data[index]
    before = np.asarray(getattr(original, field)).copy()
    updated = original.replace(**{field: getattr(original, field) + 1})

    assert updated is not original
    np.testing.assert_array_equal(getattr(original, field), before)
    np.testing.assert_array_equal(getattr(updated, field), before + 1)


def _transform_synthetic_arrays(state: EnvState, obs: Obs, info: StepInfo) -> SyntheticData:
    """Transforma arrays para probar JAX; no representa una transición del entorno."""
    return (
        state.replace(resources=state.resources + 2),
        obs.replace(own=obs.own * 2),
        info.replace(bursts=info.bursts + 1),
    )


def test_jit_accepts_and_returns_the_structures(synthetic_data: SyntheticData) -> None:
    """Comprueba actualizaciones compiladas de los tres tipos públicos."""
    state, obs, info = jax.jit(_transform_synthetic_arrays)(*synthetic_data)

    np.testing.assert_array_equal(state.resources, [3.0, 4.0])
    np.testing.assert_array_equal(obs.own, [[2.0, 0.0], [4.0, 2.0]])
    assert int(info.bursts) == 1
    np.testing.assert_array_equal(synthetic_data[0].resources, [1.0, 2.0])


def _stack_twice(array: jax.Array) -> jax.Array:
    """Añade dos muestras idénticas al lote de prueba."""
    return jnp.stack((array, array))


def test_vmap_batches_the_structures(synthetic_data: SyntheticData) -> None:
    """Vectoriza estructuras completas, incluidos escalares y máscaras."""
    batch = jax.tree_util.tree_map(_stack_twice, synthetic_data)
    state, obs, info = jax.jit(jax.vmap(_transform_synthetic_arrays))(*batch)

    np.testing.assert_array_equal(state.resources, [[3.0, 4.0], [3.0, 4.0]])
    assert state.pos.shape == (2, 2, 2)
    assert obs.detections.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(obs.det_mask, batch[1].det_mask)
    np.testing.assert_array_equal(info.bursts, [1, 1])

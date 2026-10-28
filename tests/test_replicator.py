"""Replicador con juegos de solución conocida: dilema del prisionero, caza del ciervo y piedra."""

import json

import numpy as np
import pytest

from analysis.replicator import (
    basin_shares,
    critical_mass,
    critical_mass_interval,
    fixed_points,
    is_stable,
    load_payoff_matrix,
    replicator_flow,
)

PRISONER = np.array([[3.0, 0.0], [5.0, 1.0]])  # cooperar, traicionar
STAG = np.array([[4.0, 0.0], [3.0, 2.0]])  # ciervo, liebre
ROCK_PAPER = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def has_point(points, target):
    """True si alguna fila coincide con el punto buscado."""
    return any(np.allclose(point, target, atol=1e-6) for point in points)


def test_prisoners_dilemma_flows_to_betrayal_and_only_that_vertex_is_stable():
    """Traicionar domina: el flujo converge ahí y es el único punto de reposo estable."""
    path = replicator_flow(PRISONER, np.array([0.9, 0.1]), steps=3000, dt=0.05)
    np.testing.assert_allclose(path[-1], [0, 1], atol=1e-3)
    points = fixed_points(PRISONER)
    assert len(points) == 2
    assert [is_stable(PRISONER, point) for point in points] == [False, True]


def test_stag_hunt_has_two_stable_vertices_and_an_unstable_interior_point():
    """El punto interior está en x = 2/3 y su cuenca de la liebre ocupa un tercio del símplex."""
    points = fixed_points(STAG)
    assert has_point(points, [2 / 3, 1 / 3])
    stable = [tuple(np.round(point, 3)) for point in points if is_stable(STAG, point)]
    assert sorted(stable) == [(0.0, 1.0), (1.0, 0.0)]
    shares, _ = basin_shares(STAG, n_starts=4000, steps=1500, seed=3)
    assert shares[0] == pytest.approx(1 / 3, abs=0.04)


def test_rock_paper_scissors_center_is_neutral_and_conserves_the_product():
    """El centro no es asintóticamente estable y el producto de las fracciones se conserva."""
    center = np.ones(3) / 3
    assert has_point(fixed_points(ROCK_PAPER), center)
    assert not is_stable(ROCK_PAPER, center)
    path = replicator_flow(ROCK_PAPER, np.array([0.6, 0.3, 0.1]), steps=4000, dt=0.01)
    product = path.prod(axis=1)
    assert product.max() - product.min() < 1e-4 * product[0]


def test_flow_stays_on_the_simplex_and_ignores_a_constant_shift():
    """Las fracciones suman 1 y sumar una constante a los pagos no cambia las trayectorias."""
    start = np.array([0.5, 0.3, 0.2])
    base = replicator_flow(ROCK_PAPER, start, steps=500, dt=0.02)
    shifted = replicator_flow(ROCK_PAPER + 7.0, start, steps=500, dt=0.02)
    np.testing.assert_allclose(base.sum(axis=1), 1, atol=1e-9)
    np.testing.assert_allclose(base, shifted, atol=1e-6)


def test_payoff_matrix_is_read_from_disk(tmp_path):
    """Lee la matriz y los nombres de las estrategias guardados por train.matriz_pagos."""
    path = tmp_path / "pagos.json"
    path.write_text(json.dumps({"modes": ["a", "b"], "matrix": STAG.tolist()}), encoding="utf-8")
    matrix, modes = load_payoff_matrix(path)
    np.testing.assert_array_equal(matrix, STAG)
    assert modes == ("a", "b")


def write_cell(directory, lethality, speed, matrix):
    """Guarda una celda sintética con el formato de train.pagos_rejilla."""
    cell = {
        "modes": ["coop_real", "solo"],
        "lethality": lethality,
        "signal_speed": speed,
        "matrix": matrix.tolist(),
        "se": np.zeros_like(matrix).tolist(),
    }
    name = f"pagos_lam{lethality:g}_c{speed:g}.json"
    (directory / name).write_text(json.dumps(cell), encoding="utf-8")


def test_regime_map_distinguishes_dominance_from_bistability(tmp_path):
    """Un dilema deja un régimen dominante único y una caza del ciervo uno biestable."""
    from analysis.phase_diagram import (
        critical_mass_table,
        load_payoff_grid,
        plot_critical_masses,
        regime_map,
    )

    write_cell(tmp_path, 0.25, 0.1, PRISONER)
    write_cell(tmp_path, 1.0, 0.1, STAG)
    cells = regime_map(load_payoff_grid(tmp_path))
    low, high = cells[(0.25, 0.1)], cells[(1.0, 0.1)]
    assert low["dominant"] == "solo" and not low["bistable"]
    assert low["stable_vertices"] == ["solo"]
    assert high["bistable"] and sorted(high["stable_vertices"]) == ["coop_real", "solo"]
    assert high["shares"][0] == pytest.approx(1 / 3, abs=0.05)
    table = critical_mass_table(load_payoff_grid(tmp_path))
    label = "cooperadores necesarios frente a solos"
    assert table[(1.0, 0.1)][label][0] == pytest.approx(2 / 3, abs=0.02)
    plot_critical_masses(table, tmp_path / "mapa.png")
    assert (tmp_path / "mapa.png").stat().st_size > 0


def test_critical_mass_matches_known_games_and_its_interval_brackets_the_value():
    """En la caza del ciervo vale 2/3; en el dilema es 0 para traicionar y 1 para cooperar."""
    assert critical_mass(STAG, invader=0, resident=1) == pytest.approx(2 / 3)
    assert critical_mass(PRISONER, invader=1, resident=0) == 0.0
    assert critical_mass(PRISONER, invader=0, resident=1) == 1.0
    exact = critical_mass_interval(STAG, np.zeros((2, 2)), 0, 1)
    assert exact == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    median, low, high = critical_mass_interval(STAG, np.full((2, 2), 0.3), 0, 1, n=2000)
    assert low < median < high

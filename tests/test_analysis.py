"""Métricas calculables sin ejecutar el entorno."""

import numpy as np

from analysis.metrics import summarize


def test_summary_uses_executed_actions_and_excludes_dead_agents():
    """Comprueba causas, costos y conteos sobre una trayectoria mínima conocida."""
    trajectory = {
        "alive": np.array([[True, True], [True, False], [True, False]]),
        "emission": np.array([[0, 0], [1, 0], [0, 0]]),
        "info_death_cause": np.array([[0, 2], [0, 0]]),
        "info_executed_kind": np.array([[3, 0], [4, -1]]),
        "info_preventive_attacks": np.array([1, 0]),
        "actions": np.array([[3, 0], [7, 3]]),
        "rewards": np.array([[0.2, -0.6], [0.1, 0.0]]),
    }
    report = summarize(trajectory)
    assert report["survival"] == 0.5
    assert report["death_attack"] == 0.5
    assert report["low_emission"] == 0.5
    assert report["symbols"] == report["shared_measurements"] == 1
    assert report["preventive_attacks"] == 1


def _episode(symbol: int, period: float, sends: int = 60) -> dict:
    """Episodio sintético donde una civilización transmite siempre el mismo símbolo."""
    steps = sends
    actions = np.full((steps, 2), 1)
    actions[:, 0] = 3 + symbol
    theta = np.tile(np.array([0.0, period, 0.5, 0.5, 0.45]), (steps + 1, 1))
    return {"actions": actions, "alive": np.ones((steps + 1, 2), bool), "theta": theta}


def test_mutual_information_detects_a_symbol_that_encodes_theta():
    """Un símbolo por cuartil del periodo da cerca de 2 bits; símbolos independientes, 0."""
    from analysis.metrics import message_theta_mutual_info

    informative = [_episode(i, 10.0 + 4 * i) for i in range(4)]
    assert message_theta_mutual_info(informative) > 1.8
    independent = []
    for i in range(4):
        episode = _episode(0, 10.0 + 4 * i, sends=400)
        episode["actions"][:, 0] = 3 + (np.arange(400) % 4)
        independent.append(episode)
    assert message_theta_mutual_info(independent) < 0.05


def test_regime_and_ablation_drop_use_survival_and_messages():
    """Sin mensajes y todos vivos es silencio; con mensajes es cooperación; sin vivos, colapso."""
    from analysis.metrics import channel_ablation_drop, classify_regime

    alive = np.ones((4, 2), bool)
    base = {
        "alive": alive,
        "info_executed_kind": np.zeros((3, 2), int),
        "info_preventive_attacks": np.zeros(3),
    }
    assert classify_regime(base) == "silencio_mutuo"
    talking = {**base, "info_executed_kind": np.full((3, 2), 3)}
    assert classify_regime(talking) == "cooperacion"
    dead = {**base, "alive": np.zeros((4, 2), bool)}
    assert classify_regime(dead) == "colapso"
    assert channel_ablation_drop(talking, dead) == 1.0


def test_coalition_graph_finds_two_groups_that_talk_inside():
    """Dos grupos de tres que se hablan entre sí y casi no con el otro dan dos comunidades."""
    from analysis.phase_diagram import coalition_graph

    sello = np.full((11, 6, 6), -1)
    for turno in range(1, 11):
        for grupo in ((0, 1, 2), (3, 4, 5)):
            for i in grupo:
                for j in grupo:
                    if i != j:
                        sello[turno:, i, j] = turno  # un mensaje nuevo cada turno
    sello[1:, 0, 3] = 1  # un solo mensaje suelto entre grupos
    resultado = coalition_graph({"message_t": sello})
    assert resultado["communities"] == [[0, 1, 2], [3, 4, 5]]
    assert resultado["inside_fraction"] > 0.95
    senales = np.zeros((10, 6, 6), int)
    senales[:, 0, 1] = senales[:, 1, 0] = 1
    solo_señales = coalition_graph({"info_receptions": senales}, fuente="senales")
    assert [0, 1] in solo_señales["communities"]

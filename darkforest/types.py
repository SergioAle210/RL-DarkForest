"""Estructuras de datos inmutables del entorno (pytrees de JAX)."""

from __future__ import annotations

import jax
from flax import struct

Array = jax.Array


@struct.dataclass
class EnvState:
    """Estado completo s_t (no observable por los agentes)."""

    t: Array  # () paso actual
    pos: Array  # (N, 2) posiciones en el toro [0,1]^2
    resources: Array  # (N,) r_i
    alive: Array  # (N,) bool v_i
    emission: Array  # (N,) 0 = baja, 1 = alta
    refuge: Array  # (N,) bool protegido del próximo estallido
    theta: Array  # (P,) parámetros ocultos de la anomalía
    last_burst: Array  # () turno del último estallido
    last_meas: Array  # (N, D) última medición propia
    # Colas de longitud fija con máscara de validez (formas estáticas para jit).
    sig_origin: Array  # (Qs,) int
    sig_t: Array  # (Qs,) int turno de emisión
    sig_content: Array  # (Qs, D+1) símbolo o medición
    sig_valid: Array  # (Qs,) bool
    prj_origin: Array  # (Qp,) int
    prj_target: Array  # (Qp, 2) posición objetivo
    prj_t: Array  # (Qp,) int
    prj_valid: Array  # (Qp,) bool


@struct.dataclass
class Obs:
    """Observación o_i de cada civilización, shape (N, ...)."""

    own: Array  # (N, 2) recursos y emisión propios
    detections: Array  # (N, M, F) rumbo, distancia, contenido, antigüedad
    det_mask: Array  # (N, M) bool, False = relleno
    own_meas: Array  # (N, D) última medición propia de la anomalía
    since_burst: Array  # (N,) turnos desde el último estallido
    alive: Array  # (N,) bool
    agent_id: Array  # (N,) identificador para parámetros compartidos


@struct.dataclass
class StepInfo:
    """Datos de diagnóstico (no entran a la política)."""

    death_cause: Array  # (N,) 0 = vivo, 1 = anomalía, 2 = ataque
    attacker: Array  # (N,) quién lanzó el ataque que mató (-1 si ninguno)
    preventive_attacks: Array  # () ataques contra quien nunca atacó
    shelters_correct: Array  # () refugios acertados en este estallido
    bursts: Array  # () 1 si hubo estallido

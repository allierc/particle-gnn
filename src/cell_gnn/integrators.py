"""Time-stepping integrators for cell-gnn simulations.

Provides Euler and RK4 stepping for both first_derivative (velocity)
and 2nd_derivative (acceleration) prediction modes.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import torch
from cell_gnn.cell_state import CellState


def euler_step(
    state: CellState,
    derivative: torch.Tensor,
    delta_t: float,
    prediction: str,
    bc_pos: Callable,
) -> CellState:
    """Forward Euler step.  Modifies *state* in-place and returns it.

    first_derivative:  vel = derivative;          pos += vel * dt
    2nd_derivative:    vel += derivative * dt;     pos += vel * dt
    """
    if prediction == "2nd_derivative":
        state.vel = state.vel + derivative * delta_t
    else:
        state.vel = derivative
    state.pos = bc_pos(state.pos + state.vel * delta_t)
    return state


# ------------------------------------------------------------------ #
#  RK4 – first-order system  dx/dt = f(x)
# ------------------------------------------------------------------ #

def _rk4_first_order(
    state: CellState,
    f: Callable[[CellState], torch.Tensor],
    dt: float,
    bc_pos: Callable,
    n_active: Optional[int] = None,
) -> Tuple[CellState, torch.Tensor]:
    """RK4 for dx/dt = f(x).  *f* returns velocity directly.

    k1 = f(x_n)
    k2 = f(x_n + dt/2 * k1)
    k3 = f(x_n + dt/2 * k2)
    k4 = f(x_n + dt * k3)
    x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    pos0 = state.pos.clone()

    def _mask(k):
        if n_active is not None:
            k[n_active:] = 0
        return k

    # k1
    k1 = _mask(f(state))

    # k2
    s2 = state.clone()
    s2.pos = bc_pos(pos0 + 0.5 * dt * k1)
    k2 = _mask(f(s2))

    # k3
    s3 = state.clone()
    s3.pos = bc_pos(pos0 + 0.5 * dt * k2)
    k3 = _mask(f(s3))

    # k4
    s4 = state.clone()
    s4.pos = bc_pos(pos0 + dt * k3)
    k4 = _mask(f(s4))

    # combine
    new_vel = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    state.vel = new_vel
    state.pos = bc_pos(pos0 + dt * new_vel)
    return state, k1


# ------------------------------------------------------------------ #
#  RK4 – second-order system  dx/dt = v,  dv/dt = f(x, v)
# ------------------------------------------------------------------ #

def _rk4_second_order(
    state: CellState,
    f: Callable[[CellState], torch.Tensor],
    dt: float,
    bc_pos: Callable,
    n_active: Optional[int] = None,
) -> Tuple[CellState, torch.Tensor]:
    """RK4 for coupled system  dx/dt = v, dv/dt = f(x, v).

    k1_x = v_n,                        k1_v = f(x_n, v_n)
    k2_x = v_n + dt/2 * k1_v,          k2_v = f(x_n + dt/2 * k1_x, v_n + dt/2 * k1_v)
    k3_x = v_n + dt/2 * k2_v,          k3_v = f(x_n + dt/2 * k2_x, v_n + dt/2 * k2_v)
    k4_x = v_n + dt * k3_v,            k4_v = f(x_n + dt * k3_x, v_n + dt * k3_v)

    x_{n+1} = x_n + dt/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v_{n+1} = v_n + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    """
    pos0 = state.pos.clone()
    vel0 = state.vel.clone()

    def _mask(k):
        if n_active is not None:
            k[n_active:] = 0
        return k

    # k1
    k1_v = _mask(f(state))
    k1_x = vel0.clone()
    if n_active is not None:
        k1_x[n_active:] = 0

    # k2
    s2 = state.clone()
    s2.pos = bc_pos(pos0 + 0.5 * dt * k1_x)
    s2.vel = vel0 + 0.5 * dt * k1_v
    k2_v = _mask(f(s2))
    k2_x = s2.vel.clone()
    if n_active is not None:
        k2_x[n_active:] = 0

    # k3
    s3 = state.clone()
    s3.pos = bc_pos(pos0 + 0.5 * dt * k2_x)
    s3.vel = vel0 + 0.5 * dt * k2_v
    k3_v = _mask(f(s3))
    k3_x = s3.vel.clone()
    if n_active is not None:
        k3_x[n_active:] = 0

    # k4
    s4 = state.clone()
    s4.pos = bc_pos(pos0 + dt * k3_x)
    s4.vel = vel0 + dt * k3_v
    k4_v = _mask(f(s4))
    k4_x = s4.vel.clone()
    if n_active is not None:
        k4_x[n_active:] = 0

    # combine
    state.pos = bc_pos(pos0 + dt / 6.0 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x))
    state.vel = vel0 + dt / 6.0 * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return state, k1_v


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def rk4_step(
    state: CellState,
    derivative_fn: Callable[[CellState], torch.Tensor],
    delta_t: float,
    prediction: str,
    bc_pos: Callable,
    n_active: Optional[int] = None,
) -> Tuple[CellState, torch.Tensor]:
    """Single RK4 step.

    Args:
        state:         current CellState (modified in-place)
        derivative_fn: callable(CellState) -> derivative tensor;
                       must handle edge-building internally
        delta_t:       time step
        prediction:    'first_derivative' or '2nd_derivative'
        bc_pos:        boundary-condition function for positions
        n_active:      if set, only the first n_active cells are updated

    Returns:
        (updated_state, k1) where k1 is the derivative at the initial state
    """
    if prediction == "2nd_derivative":
        return _rk4_second_order(state, derivative_fn, delta_t, bc_pos, n_active)
    else:
        return _rk4_first_order(state, derivative_fn, delta_t, bc_pos, n_active)


def integrate_step(
    state: CellState,
    derivative_fn: Callable[[CellState], torch.Tensor],
    derivative_at_state: Optional[torch.Tensor],
    delta_t: float,
    prediction: str,
    integration: str,
    bc_pos: Callable,
    n_active: Optional[int] = None,
) -> Tuple[CellState, torch.Tensor]:
    """Dispatch to Euler or RK4 based on *integration* mode.

    Args:
        derivative_fn:       callable(CellState) -> derivative.
                             Required for RK4; ignored for Euler.
        derivative_at_state: pre-computed derivative for Euler.
                             Ignored for RK4 (computed internally as k1).
        integration:         'Euler' or 'Runge-Kutta'
        n_active:            only update the first n_active cells (for ghost cells)

    Returns:
        (updated_state, derivative_at_initial_state)
    """
    if integration == "Runge-Kutta":
        return rk4_step(state, derivative_fn, delta_t, prediction, bc_pos, n_active)
    else:
        assert derivative_at_state is not None
        new_state = euler_step(state, derivative_at_state, delta_t, prediction, bc_pos)
        return new_state, derivative_at_state

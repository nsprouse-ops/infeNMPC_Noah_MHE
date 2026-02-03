# MHE_estimation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import copy
import pyomo.environ as pyo
from pyomo.contrib.mpc import ScalarData

from make_model import _make_finite_horizon_model
from indexing_tools import _get_derivative_and_state_vars, _get_variable_key_for_data


@dataclass
class MHEResult:
    M_eff: int
    xhat: Dict[str, float]                 # estimated state at current time (end of window)
    model: pyo.ConcreteModel               # the MHE model (optional to keep for debugging)
    solver_result: object                  # solver results


def build_mhe_histories_from_io_data_array(
    io_data_array: Sequence[Sequence[float]],
    CV_index: Sequence[str],
    MV_index: Sequence[str],
    M_desired: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], int]:
    """
    Build per-sampling-step histories for MHE from io_data_array.

    io_data_array format (based on your run_MPC.py):
      - Row k contains measurements y_k (CVs)
      - Row 0 has MV columns = None
      - Row k>=1 has MV columns = u_{k-1} (input applied between k-1 -> k)

    Returns:
      y_hist: {cv: [y_{k-M_eff}, ..., y_k]} length M_eff+1
      u_hist: {mv: [u_{k-M_eff}, ..., u_{k-1}]} length M_eff
      M_eff: effective window length used (min(M_desired, k))
    """
    if M_desired < 0:
        raise ValueError("M_desired must be >= 0")

    n = len(io_data_array)
    if n == 0:
        raise ValueError("io_data_array is empty.")

    n_cv = len(CV_index)
    n_mv = len(MV_index)

    k = n - 1
    M_eff = min(M_desired, k)

    # Measurements y: rows k-M_eff ... k (length M_eff+1)
    y_rows = list(range(k - M_eff, k + 1))

    # Inputs u: rows (k-M_eff+1) ... k (length M_eff)
    # because row r stores u_{r-1}, and row 0 has None
    u_rows = list(range(k - M_eff + 1, k + 1))

    # Build y histories
    y_hist: Dict[str, List[float]] = {name: [] for name in CV_index}
    for r in y_rows:
        row = io_data_array[r]
        if len(row) < n_cv + n_mv:
            raise ValueError(
                f"io_data_array row {r} has length {len(row)}, expected {n_cv + n_mv}"
            )
        for j, cv in enumerate(CV_index):
            y_hist[cv].append(float(row[j]))

    # Build u histories
    u_hist: Dict[str, List[float]] = {name: [] for name in MV_index}
    if M_eff > 0:
        for r in u_rows:
            row = io_data_array[r]
            for j, mv in enumerate(MV_index):
                val = row[n_cv + j]
                if val is None:
                    raise ValueError(
                        f"MV '{mv}' is None at io_data_array row {r}. "
                        f"Not enough MV history yet."
                    )
                u_hist[mv].append(float(val))

    return y_hist, u_hist, M_eff


def _make_options_for_mhe(options, M_eff: int):
    """
    Create a shallow copy of options with a horizon sized for MHE:
      - nfe_finite = M_eff
      - time horizon = M_eff * sampling_time (handled inside _make_finite_horizon_model)
    """
    opt = copy.copy(options)
    opt.nfe_finite = int(M_eff)
    return opt


def solve_mhe_no_arrival_cost(
    options,
    io_data_array: Sequence[Sequence[float]],
    M_desired: int,
    solver_name: str = "ipopt",
    tee: bool = False,
) -> Optional[MHEResult]:
    """
    Solve MHE with arrival cost = 0.

    Returns:
      MHEResult if MHE can run (k>=1), else None at k=0.

    Notes:
      - Uses existing dynamic model builder (_make_finite_horizon_model)
      - Penalizes measurement residuals at finite elements only
      - Fixes MV trajectory using u_hist (piecewise-constant per sampling step)
    """
    k = len(io_data_array) - 1
    print("DEBUG: k =", k)
    print("DEBUG: last row =", io_data_array[-1])
    print("DEBUG: M_desired =", M_desired)   #MHE will not run unless there is not at least one control input

    # Need at least one applied MV (k>=1) to do meaningful window dynamics
    k = len(io_data_array) - 1
    if k < 1:
        return None

    # Build histories (per sampling step)
    # We need CV_index and MV_index; easiest is to build a small model once to read them.
    # But your options/model setup already implies these sets are fixed across builds,
    # so we can build the MHE model first and then fill parameters.
    #
    # We'll build the model after we know M_eff (from history), so first make a tiny pass:
    # Use a temporary model to read indices cleanly.
    tmp_opt = _make_options_for_mhe(options, M_eff=1)
    tmp = pyo.ConcreteModel()
    tmp = _make_finite_horizon_model(tmp, tmp_opt)
    CV_index = list(tmp.CV_index)
    MV_index = list(tmp.MV_index)

    y_hist, u_hist, M_eff = build_mhe_histories_from_io_data_array(
        io_data_array=io_data_array,
        CV_index=CV_index,
        MV_index=MV_index,
        M_desired=M_desired,
    )

    # If M_eff ends up 0 (e.g., M_desired=0), still not useful without arrival cost.
    if M_eff < 1:
        return None

    # Build an MHE model with M_eff finite elements
    mhe_opt = _make_options_for_mhe(options, M_eff=M_eff)
    m = pyo.ConcreteModel()
    m = _make_finite_horizon_model(m, mhe_opt)

    # Finite element times: should be [0, Ts, 2Ts, ..., M_eff*Ts]
    fe_times = list(m.time.get_finite_elements())
    if len(fe_times) != M_eff + 1:
        raise RuntimeError(
            f"Expected {M_eff+1} finite elements, got {len(fe_times)}. "
            f"Check discretization."
        )

    # --- Measurement parameters y_meas(cv, t_fe) ---
    m.y_meas = pyo.Param(m.CV_index, m.time, mutable=True, initialize=0.0)

    # Load measurement history only at finite elements
    # y_hist[cv] has length M_eff+1 aligned with fe_times
    for cv in m.CV_index:
        series = y_hist[cv]
        if len(series) != len(fe_times):
            raise ValueError(
                f"y_hist['{cv}'] length {len(series)} != finite elements {len(fe_times)}"
            )
        for t_fe, val in zip(fe_times, series):
            m.y_meas[cv, t_fe] = float(val)

    # --- Fix MV trajectory (piecewise-constant per sampling step) ---
    # u_hist[mv] has length M_eff aligned with intervals.
    # We fix MV at finite elements:
    #   - at t0, use u_hist[mv][0]
    #   - at t_j (j>=1), use u_hist[mv][j-1]
    #   - at final time, hold last input
    for mv in m.MV_index:
        u_series = u_hist[mv]
        if len(u_series) != M_eff:
            raise ValueError(
                f"u_hist['{mv}'] length {len(u_series)} != M_eff {M_eff}"
            )

        for j, t_fe in enumerate(fe_times):
            idx = min(max(j - 1, 0), M_eff - 1)
            getattr(m, mv)[t_fe].fix(float(u_series[idx]))
    print("DEBUG: M_eff =", M_eff)
    print("DEBUG: y_hist lens =", {k: len(v) for k, v in y_hist.items()})
    print("DEBUG: u_hist lens =", {k: len(v) for k, v in u_hist.items()})

    # --- Objective: sum of squared measurement errors at finite elements only ---
    def _mhe_obj_rule(mm):
        expr = 0
        for t_fe in fe_times:
            for cv in mm.CV_index:
                yhat = getattr(mm, cv)[t_fe]
                ymeas = mm.y_meas[cv, t_fe]
                # Use stage_cost_weights if you have them; else weight=1
                w = 1.0
                if hasattr(mm, "cv_cost"):
                    # your model_equations defines cv_cost Param on CV_index
                    w = mm.cv_cost[cv]
                expr += w * (yhat - ymeas) ** 2
        return expr

    m.mhe_obj = pyo.Objective(rule=_mhe_obj_rule)

    # --- Solve ---
    solver = pyo.SolverFactory(solver_name)
    res = solver.solve(m, tee=tee)

    # --- Extract xhat at final finite element time ---
    tf = fe_times[-1]
    unmeasured = set(m.Unmeasured_index)

    xhat = {}
    for v in m.state_vars:
        name = v.local_name
        if name in unmeasured:
            xhat[name] = pyo.value(v[tf])

    return MHEResult(M_eff=M_eff, xhat=xhat, model=m, solver_result=res)


"""The constrained covariate-balancing objective in Methods M3."""

import cvxpy as cp
import numpy as np


def fit_control_weights(control_matrix, treatment_means, solver="MOSEK"):
    """Minimize the Euclidean imbalance with nonnegative weights summing to one."""
    weights = cp.Variable(control_matrix.shape[0])
    problem = cp.Problem(
        cp.Minimize(cp.norm(control_matrix.T @ weights - treatment_means, 2)),
        [weights >= 0, cp.sum(weights) == 1],
    )
    problem.solve(solver=solver, canon_backend=cp.SCIPY_CANON_BACKEND)
    if (
        problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        or weights.value is None
    ):
        raise RuntimeError(f"Covariate balancing did not converge: {problem.status}")
    result = np.asarray(weights.value).ravel()
    if (
        not np.isfinite(result).all()
        or result.min() < -1e-6
        or abs(result.sum() - 1) > 1e-5
    ):
        raise RuntimeError("The solver returned infeasible control weights.")
    result = np.maximum(result, 0)
    return result / result.sum()

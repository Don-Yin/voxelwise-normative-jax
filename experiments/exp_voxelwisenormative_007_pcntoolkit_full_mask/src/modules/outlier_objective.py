"""Re-evaluate the exact warped BLR objective and refine from both archived fits."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from shared.voxelwisenormative.batched_blr import PENALTY

HYP_NAMES = ("log_noise_precision", "warp_shift", "log_warp_shape", "log_weight_precision")


@dataclass(frozen=True)
class Objective:
    """One voxel with the common design and training-standardised responses."""

    phi: np.ndarray
    gram: np.ndarray
    phit: np.ndarray
    y: np.ndarray
    yt: np.ndarray

    def evaluate(self, theta: np.ndarray):
        """Return marginal NLL, penalised objective, exact gradient and held-out z."""
        logbeta, eps, logb, logalpha = theta
        beta, b, alpha = np.exp([logbeta, logb, logalpha])
        n, d = self.phi.shape
        u = b * (np.arcsinh(self.y) + eps)
        ywarp = np.sinh(u)
        chol = cho_factor(beta * self.gram + alpha * np.eye(d))
        weights = cho_solve(chol, beta * (self.phi.T @ ywarp))
        residual = ywarp - self.phi @ weights
        quad = residual @ residual
        logjac = logb + np.logaddexp(u, -u) - np.log(2) - 0.5 * np.log1p(self.y**2)
        logdet = 2 * np.log(np.diag(chol[0])).sum()
        nll = (n * (np.log(2 * np.pi) - logbeta) - d * logalpha + beta * quad
               + alpha * (weights @ weights) + logdet) / 2 - logjac.sum()
        norm = np.linalg.norm(theta)
        tangent = beta * residual * np.cosh(u) - np.tanh(u)
        grad = np.array([
            (-n + beta * (quad + np.trace(cho_solve(chol, self.gram)))) / 2,
            b * tangent.sum(),
            tangent @ u - n,
            (-d + alpha * ((weights @ weights) + np.trace(cho_solve(chol, np.eye(d))))) / 2,
        ])
        if norm:
            grad += PENALTY * theta / norm
        variance = 1 / beta + np.einsum("ij,ji->i", self.phit, cho_solve(chol, self.phit.T))
        z = (np.sinh(b * (np.arcsinh(self.yt) + eps)) - self.phit @ weights) / np.sqrt(variance)
        return nll, nll + PENALTY * norm, grad, z

    def describe(self, theta: np.ndarray, row: int) -> dict:
        """Report comparable objective, warp and noise parameters on their natural scales."""
        nll, value, grad, z = self.evaluate(theta)
        return {"theta": theta.tolist(), "marginal_nll": float(nll), "penalised_nll": float(value),
                "gradient_inf": float(np.linalg.norm(grad, ord=np.inf)), "z_at_worst_row": float(z[row]),
                "noise_sd_warped": float(np.exp(-theta[0] / 2)), "warp_shift": float(theta[1]),
                "warp_shape": float(np.exp(theta[2])), "weight_precision": float(np.exp(theta[3]))}

    def refine(self, start: np.ndarray, bounds: list[tuple[float, float]], row: int) -> dict:
        """Improve one endpoint with a common analytic gradient; retain stop evidence."""
        def fun(theta):
            """Evaluate a guarded trial step without changing the archived fits."""
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    _, value, gradient, _ = self.evaluate(theta)
                    if np.isfinite(value) and np.isfinite(gradient).all():
                        return value, gradient
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                pass
            return 1e200, np.zeros_like(theta)

        result = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-15, "gtol": 3e-7, "maxiter": 1500, "maxls": 60})
        hess = np.stack([(self.evaluate(result.x + d * 1e-5)[2] - self.evaluate(result.x - d * 1e-5)[2]) / 2e-5
                         for d in np.eye(len(start))], axis=1)
        at_bound = any(min(abs(x - lo), abs(x - hi)) <= 1e-6 for x, (lo, hi) in zip(result.x, bounds, strict=True))
        return self.describe(result.x, row) | {
            "success": bool(result.success), "status": int(result.status), "task": str(result.message),
            "iterations": int(result.nit), "minimum_hessian_eigenvalue": float(np.linalg.eigvalsh((hess + hess.T) / 2)[0]),
            "at_safety_bound": at_bound,
        }


def line_barrier(objective: Objective, first: np.ndarray, last: np.ndarray) -> float:
    """Measure any rise above both endpoints along their straight line, not all possible paths."""
    values = [objective.evaluate(first + t * (last - first))[1] for t in np.linspace(0, 1, 41)]
    return float(max(values) - max(values[0], values[-1]))


def safety_bounds(first: np.ndarray, last: np.ndarray) -> list[tuple[float, float]]:
    """Limit overflow during refinement while keeping both archived endpoints interior."""
    baseline = ((-25, 25), (-8, 8), (-7, 2), (-25, 25))
    return [(min(lo, float(a) - 1, float(b) - 1), max(hi, float(a) + 1, float(b) + 1))
            for a, b, (lo, hi) in zip(first, last, baseline, strict=True)]

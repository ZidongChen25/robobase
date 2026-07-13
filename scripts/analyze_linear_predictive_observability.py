#!/usr/bin/env python3
"""Analyze linear closed-loop action observability and predictive uncertainty.

The deterministic system is

    x[t + 1] = A x[t] + B a[t],     a[t] = K x[t],

with closed-loop matrix ``A_c = A + B K``.  A horizon-H action chunk is

    [a[0]; ...; a[H - 1]] = O_H x[0],
    O_H = [K; K A_c; ...; K A_c**(H - 1)].

For uncertainty analysis, ``Q`` is additive state/process covariance and
``R`` is additive action/output covariance.  In particular, ``R`` affects the
predicted action covariance but is not fed back through ``B``:

    P[t + 1] = A_c P[t] A_c.T + Q,
    S_a[t]   = K P[t] K.T + R.

By default the CLI uses a two-state canonical system.  Custom A, B, and K
matrices can be supplied as JSON.  The sweep axes are constructed to be
independent: changing action observability updates A to preserve A_c, while
changing the requested closed-loop spectral radius rescales A_c and then
updates A.  Example:

    python scripts/analyze_linear_predictive_observability.py \
      --spectral-radii 0.5 0.9 1.1 \
      --action-observability-scales 0.0 0.1 1.0 \
      --noise-scales 0.0 1.0 10.0 \
      --horizons 1 2 4 8 \
      --output-csv /tmp/linear_observability.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LinearSystem:
    """Discrete linear plant and a deterministic linear expert."""

    A: np.ndarray
    B: np.ndarray
    K: np.ndarray

    def __post_init__(self) -> None:
        a = _as_matrix("A", self.A)
        b = _as_matrix("B", self.B)
        k = _as_matrix("K", self.K)
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"A must be square; got {a.shape}")
        state_dim = a.shape[0]
        if b.shape[0] != state_dim:
            raise ValueError(
                f"B must have {state_dim} rows; got shape {b.shape}"
            )
        action_dim = b.shape[1]
        if k.shape != (action_dim, state_dim):
            raise ValueError(
                "K must have shape (action_dim, state_dim) = "
                f"{(action_dim, state_dim)}; got {k.shape}"
            )
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)
        object.__setattr__(self, "K", k)

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def action_dim(self) -> int:
        return self.B.shape[1]

    @property
    def closed_loop(self) -> np.ndarray:
        return closed_loop_matrix(self.A, self.B, self.K)


@dataclass(frozen=True)
class PredictiveCovariances:
    """Per-step state and action marginal predictive covariances.

    ``state[t]`` is P[t] for t=0,...,H, while ``action[t]`` is S_a[t]
    for t=0,...,H-1.
    """

    state: np.ndarray
    action: np.ndarray


def _as_matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix; got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix.copy()


def closed_loop_matrix(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Return the expert-induced closed-loop matrix A + B K."""

    return np.asarray(A, dtype=np.float64) + np.asarray(
        B, dtype=np.float64
    ) @ np.asarray(K, dtype=np.float64)


def spectral_radius(matrix: np.ndarray) -> float:
    """Return max(abs(eigenvalue)) for a square matrix."""

    matrix = _as_matrix("matrix", matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square; got {matrix.shape}")
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def action_observability_matrix(
    A_closed: np.ndarray,
    K: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Construct O_H mapping x[0] to H deterministic expert actions."""

    A_closed = _as_matrix("A_closed", A_closed)
    K = _as_matrix("K", K)
    if A_closed.shape[0] != A_closed.shape[1]:
        raise ValueError(f"A_closed must be square; got {A_closed.shape}")
    if K.shape[1] != A_closed.shape[0]:
        raise ValueError(
            "K state dimension must match A_closed; got "
            f"K={K.shape}, A_closed={A_closed.shape}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be positive; got {horizon}")

    blocks = []
    transition_power = np.eye(A_closed.shape[0], dtype=np.float64)
    for _ in range(horizon):
        blocks.append(K @ transition_power)
        transition_power = transition_power @ A_closed
    return np.vstack(blocks)


def action_observability_metrics(
    observability: np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> dict[str, object]:
    """Compute rank, spectrum, Gramian volume, and conditioning metrics."""

    observability = _as_matrix("observability", observability)
    state_dim = observability.shape[1]
    singular_values = np.linalg.svd(observability, compute_uv=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    if rank_tolerance is None:
        rank_tolerance = (
            max(observability.shape) * np.finfo(np.float64).eps * leading
        )
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be non-negative")

    rank = int(np.sum(singular_values > rank_tolerance))
    padded_singular_values = np.zeros(state_dim, dtype=np.float64)
    padded_singular_values[: singular_values.size] = singular_values
    fully_observable = rank == state_dim

    if fully_observable:
        smallest = float(padded_singular_values[-1])
        condition = leading / smallest if smallest > 0.0 else math.inf
        gramian_logdet = float(
            2.0 * np.log(padded_singular_values).sum()
        )
    else:
        smallest = 0.0
        condition = math.inf
        gramian_logdet = -math.inf

    nonzero = singular_values[singular_values > rank_tolerance]
    gramian_log_pdet = (
        float(2.0 * np.log(nonzero).sum()) if nonzero.size else -math.inf
    )
    gramian = observability.T @ observability
    return {
        "rank": rank,
        "fully_observable": fully_observable,
        "singular_values": padded_singular_values,
        "sigma_max": leading,
        "sigma_min": smallest,
        "condition": condition,
        "gramian": gramian,
        "gramian_trace": float(np.trace(gramian)),
        "gramian_logdet": gramian_logdet,
        "gramian_log_pdet": gramian_log_pdet,
        "rank_tolerance": float(rank_tolerance),
    }


def _validate_covariance(
    name: str,
    covariance: np.ndarray,
    dimension: int,
) -> np.ndarray:
    covariance = _as_matrix(name, covariance)
    if covariance.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape {(dimension, dimension)}; "
            f"got {covariance.shape}"
        )
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    tolerance = max(1.0, float(np.linalg.norm(covariance, ord=2))) * 1e-10
    if float(np.linalg.eigvalsh(covariance)[0]) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return covariance


def predictive_covariances(
    A_closed: np.ndarray,
    K: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    horizon: int,
) -> PredictiveCovariances:
    """Recursively compute state P[t] and action S_a[t] covariances."""

    A_closed = _as_matrix("A_closed", A_closed)
    K = _as_matrix("K", K)
    if A_closed.shape[0] != A_closed.shape[1]:
        raise ValueError(f"A_closed must be square; got {A_closed.shape}")
    state_dim = A_closed.shape[0]
    if K.shape[1] != state_dim:
        raise ValueError(
            "K state dimension must match A_closed; got "
            f"K={K.shape}, A_closed={A_closed.shape}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be positive; got {horizon}")

    action_dim = K.shape[0]
    current = _validate_covariance("P0", P0, state_dim)
    Q = _validate_covariance("Q", Q, state_dim)
    R = _validate_covariance("R", R, action_dim)
    state_covariances = [current]
    action_covariances = []
    for _ in range(horizon):
        action_covariance = K @ current @ K.T + R
        action_covariances.append(
            0.5 * (action_covariance + action_covariance.T)
        )
        current = A_closed @ current @ A_closed.T + Q
        current = 0.5 * (current + current.T)
        state_covariances.append(current)
    return PredictiveCovariances(
        state=np.stack(state_covariances),
        action=np.stack(action_covariances),
    )


def canonical_system() -> LinearSystem:
    """Return a two-mode base system used by the standalone sweep.

    The second closed-loop eigenvalue differs from the first, so observing both
    state coordinates through K=[1, 1] produces rank growth between H=1 and
    H=2.  Sweep transformations preserve or explicitly control A_c.
    """

    A_closed = np.diag([0.9, 0.45])
    B = np.asarray([[0.2], [0.1]], dtype=np.float64)
    K = np.asarray([[1.0, 1.0]], dtype=np.float64)
    A = A_closed - B @ K
    return LinearSystem(A=A, B=B, K=K)


def set_closed_loop_spectral_radius(
    system: LinearSystem,
    target_radius: float,
) -> LinearSystem:
    """Rescale A_c to a target radius while preserving B and K."""

    if not np.isfinite(target_radius) or target_radius < 0.0:
        raise ValueError(
            f"target_radius must be finite and non-negative; got {target_radius}"
        )
    current_radius = spectral_radius(system.closed_loop)
    if current_radius == 0.0:
        if target_radius != 0.0:
            raise ValueError(
                "cannot rescale a zero-radius closed loop to a non-zero radius"
            )
        target_closed_loop = system.closed_loop
    else:
        target_closed_loop = system.closed_loop * (
            target_radius / current_radius
        )
    target_A = target_closed_loop - system.B @ system.K
    return LinearSystem(A=target_A, B=system.B, K=system.K)


def set_action_observability_scale(
    system: LinearSystem,
    scale: float,
    *,
    state_index: int = -1,
) -> LinearSystem:
    """Scale one state's action readout while exactly preserving A_c."""

    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"scale must be finite and non-negative; got {scale}")
    index = state_index % system.state_dim
    K = system.K.copy()
    K[:, index] *= scale
    A = system.closed_loop - system.B @ K
    return LinearSystem(A=A, B=system.B, K=K)


def _covariance_logdet(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1e-12
    if np.any(eigenvalues <= tolerance):
        return -math.inf
    return float(np.log(eigenvalues).sum())


def _json_matrix(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix, dtype=np.float64).tolist(), separators=(",", ":"))


def run_sweep(
    base_system: LinearSystem,
    *,
    horizons: Sequence[int],
    spectral_radii: Sequence[float],
    action_observability_scales: Sequence[float],
    noise_scales: Sequence[float],
    P0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    observability_state_index: int = -1,
    rank_tolerance: float | None = None,
) -> list[dict[str, object]]:
    """Evaluate the Cartesian product of the requested independent axes."""

    horizons = tuple(int(horizon) for horizon in horizons)
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError(f"horizons must be non-empty and positive; got {horizons}")
    if not spectral_radii:
        raise ValueError("spectral_radii must be non-empty")
    if not action_observability_scales:
        raise ValueError("action_observability_scales must be non-empty")
    if not noise_scales:
        raise ValueError("noise_scales must be non-empty")

    max_horizon = max(horizons)
    rows: list[dict[str, object]] = []
    for target_radius in spectral_radii:
        radius_system = set_closed_loop_spectral_radius(
            base_system, float(target_radius)
        )
        for observability_scale in action_observability_scales:
            system = set_action_observability_scale(
                radius_system,
                float(observability_scale),
                state_index=observability_state_index,
            )
            actual_radius = spectral_radius(system.closed_loop)
            for noise_scale in noise_scales:
                noise_scale = float(noise_scale)
                if not np.isfinite(noise_scale) or noise_scale < 0.0:
                    raise ValueError(
                        "noise scales must be finite and non-negative; "
                        f"got {noise_scale}"
                    )
                covariances = predictive_covariances(
                    system.closed_loop,
                    system.K,
                    P0,
                    np.asarray(Q, dtype=np.float64) * noise_scale,
                    np.asarray(R, dtype=np.float64) * noise_scale,
                    max_horizon,
                )
                for horizon in horizons:
                    observability = action_observability_matrix(
                        system.closed_loop,
                        system.K,
                        horizon,
                    )
                    metrics = action_observability_metrics(
                        observability,
                        rank_tolerance=rank_tolerance,
                    )
                    state_covariance = covariances.state[horizon]
                    action_covariance = covariances.action[horizon - 1]
                    rows.append(
                        {
                            "target_spectral_radius": float(target_radius),
                            "closed_loop_spectral_radius": actual_radius,
                            "action_observability_scale": float(
                                observability_scale
                            ),
                            "noise_scale": noise_scale,
                            "horizon": horizon,
                            "state_dim": system.state_dim,
                            "action_dim": system.action_dim,
                            "observability_rank": metrics["rank"],
                            "fully_observable": metrics["fully_observable"],
                            "singular_values": json.dumps(
                                np.asarray(metrics["singular_values"]).tolist(),
                                separators=(",", ":"),
                            ),
                            "sigma_max": metrics["sigma_max"],
                            "sigma_min": metrics["sigma_min"],
                            "observability_condition": metrics["condition"],
                            "gramian_trace": metrics["gramian_trace"],
                            "gramian_logdet": metrics["gramian_logdet"],
                            "gramian_log_pdet": metrics["gramian_log_pdet"],
                            "state_cov_trace_step_h": float(
                                np.trace(state_covariance)
                            ),
                            "state_cov_logdet_step_h": _covariance_logdet(
                                state_covariance
                            ),
                            "action_cov_trace_step_h_minus_1": float(
                                np.trace(action_covariance)
                            ),
                            "action_cov_logdet_step_h_minus_1": (
                                _covariance_logdet(action_covariance)
                            ),
                            "state_covariance_step_h": _json_matrix(
                                state_covariance
                            ),
                            "action_covariance_step_h_minus_1": _json_matrix(
                                action_covariance
                            ),
                            "closed_loop_matrix": _json_matrix(
                                system.closed_loop
                            ),
                            "action_matrix_K": _json_matrix(system.K),
                        }
                    )
    return rows


def write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    """Write sweep records with a stable header."""

    if not rows:
        raise ValueError("cannot write an empty sweep")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_json_matrix(text: str, name: str) -> np.ndarray:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a JSON matrix: {exc}"
        ) from exc
    try:
        return _as_matrix(name, value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep closed-loop stability, action observability, and noise for "
            "a discrete linear expert system."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("linear_predictive_observability.csv"),
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--spectral-radii",
        nargs="+",
        type=float,
        default=None,
        help="Target closed-loop spectral radii. Default preserves the base system.",
    )
    parser.add_argument(
        "--action-observability-scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="Multipliers for one selected state column of K.",
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="Joint multipliers for the supplied/base Q and R matrices.",
    )
    parser.add_argument(
        "--observability-state-index",
        type=int,
        default=-1,
        help="State column of K controlled by the observability scale.",
    )
    parser.add_argument("--initial-state-variance", type=float, default=1.0)
    parser.add_argument("--process-noise-variance", type=float, default=0.01)
    parser.add_argument("--action-noise-variance", type=float, default=0.001)
    parser.add_argument(
        "--a-matrix",
        type=lambda text: _parse_json_matrix(text, "A"),
        default=None,
        help='Custom A as JSON, e.g. "[[0.8,0.0],[0.0,0.6]]".',
    )
    parser.add_argument(
        "--b-matrix",
        type=lambda text: _parse_json_matrix(text, "B"),
        default=None,
        help='Custom B as JSON, e.g. "[[1.0],[0.0]]".',
    )
    parser.add_argument(
        "--k-matrix",
        type=lambda text: _parse_json_matrix(text, "K"),
        default=None,
        help='Custom K as JSON, e.g. "[[-0.1,-0.2]]".',
    )
    parser.add_argument(
        "--p0-matrix",
        type=lambda text: _parse_json_matrix(text, "P0"),
        default=None,
    )
    parser.add_argument(
        "--q-matrix",
        type=lambda text: _parse_json_matrix(text, "Q"),
        default=None,
    )
    parser.add_argument(
        "--r-matrix",
        type=lambda text: _parse_json_matrix(text, "R"),
        default=None,
    )
    parser.add_argument("--rank-tolerance", type=float, default=None)
    return parser


def _system_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> LinearSystem:
    custom = (args.a_matrix, args.b_matrix, args.k_matrix)
    if all(value is None for value in custom):
        return canonical_system()
    if any(value is None for value in custom):
        parser.error("--a-matrix, --b-matrix, and --k-matrix must be used together")
    return LinearSystem(A=args.a_matrix, B=args.b_matrix, K=args.k_matrix)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    system = _system_from_args(parser, args)
    spectral_radii = args.spectral_radii
    if spectral_radii is None:
        spectral_radii = [spectral_radius(system.closed_loop)]

    if args.initial_state_variance < 0.0:
        parser.error("--initial-state-variance must be non-negative")
    if args.process_noise_variance < 0.0:
        parser.error("--process-noise-variance must be non-negative")
    if args.action_noise_variance < 0.0:
        parser.error("--action-noise-variance must be non-negative")

    P0 = (
        args.p0_matrix
        if args.p0_matrix is not None
        else np.eye(system.state_dim) * args.initial_state_variance
    )
    Q = (
        args.q_matrix
        if args.q_matrix is not None
        else np.eye(system.state_dim) * args.process_noise_variance
    )
    R = (
        args.r_matrix
        if args.r_matrix is not None
        else np.eye(system.action_dim) * args.action_noise_variance
    )
    rows = run_sweep(
        system,
        horizons=args.horizons,
        spectral_radii=spectral_radii,
        action_observability_scales=args.action_observability_scales,
        noise_scales=args.noise_scales,
        P0=P0,
        Q=Q,
        R=R,
        observability_state_index=args.observability_state_index,
        rank_tolerance=args.rank_tolerance,
    )
    write_csv(rows, args.output_csv)
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Port of the paper's optimizer: DIRECT-L over min-jerk moves (main_opt_NLOPT.py of Nah et al. 2023)."""

import argparse
import os

import nlopt
import numpy as np

from targets.whip import Whip


def run(
    whip: Whip,
    maxeval: int = 600,
    stopval: float = 1e-5,
    save_path: str | None = None,
    verbose: bool = False,
) -> dict:
    """Minimize the whip cost over the paper's move parameters, recording every evaluation in order."""
    lb, ub = whip.direct_bounds
    pars_history, y_history = [], []

    def objective(pars: np.ndarray, grad: np.ndarray) -> float:
        _, q0, dq0 = whip.direct_reference(pars)
        y = float(whip.evaluate_reference(q0, dq0))
        pars_history.append(pars.copy())
        y_history.append(y)
        if verbose:
            print(f"Iteration {len(y_history)}: current = {y:.8f}, best = {min(y_history):.8f}")
        if save_path is not None and len(y_history) % 50 == 0:
            np.savez(save_path, pars=np.asarray(pars_history), y=np.asarray(y_history))
        return y

    opt = nlopt.opt(nlopt.GN_DIRECT_L, lb.size)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(maxeval)
    opt.set_stopval(stopval)
    opt.set_min_objective(objective)
    pars_opt = opt.optimize(0.5 * (lb + ub))

    pars_history, y_history = np.asarray(pars_history), np.asarray(y_history)
    if save_path is not None:
        np.savez(save_path, pars=pars_history, y=y_history)
    return dict(pars=pars_history, y=y_history, pars_opt=pars_opt, y_opt=opt.last_optimum_value())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--planar", action="store_true", help="The 2-DOF planar model with its single target.")
    parser.add_argument("--maxeval", type=int, default=600)
    parser.add_argument("--stopval", type=float, default=1e-5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    name = "whip2d" if args.planar else f"whip{args.target}"
    save_dir = f"{args.results_dir}/{name}/paper_direct_l"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/run.npz"

    results = run(
        Whip(planar=True) if args.planar else Whip(target=args.target),
        maxeval=args.maxeval,
        stopval=args.stopval,
        save_path=save_path,
        verbose=args.verbose,
    )
    if args.verbose:
        print(f"Optimum {results['y_opt']:.8f} at {results['pars_opt']}")
        print(f"Results saved to {save_path}")

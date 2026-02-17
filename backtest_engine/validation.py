"""Walk-Forward Analysis and CSCV (Combinatorially Symmetric Cross-Validation)."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np


class WalkForward:
    """Walk-Forward analysis with rolling or anchored windows.

    Parameters
    ----------
    n_bars : Total number of bars in the dataset.
    is_ratio : Fraction of window used for in-sample (e.g. 0.7).
    n_splits : Number of IS/OOS splits.
    anchored : If True, IS always starts from bar 0. If False, rolling window.
    """

    def __init__(
        self,
        n_bars: int,
        is_ratio: float = 0.7,
        n_splits: int = 5,
        anchored: bool = False,
    ):
        self.n_bars = n_bars
        self.is_ratio = is_ratio
        self.n_splits = n_splits
        self.anchored = anchored
        self.splits = self._compute_splits()

    def _compute_splits(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Compute (IS_start, IS_end), (OOS_start, OOS_end) for each split."""
        splits = []

        if self.anchored:
            # Anchored: IS always starts from 0, OOS grows
            oos_total = int(self.n_bars * (1.0 - self.is_ratio))
            oos_per_split = max(1, oos_total // self.n_splits)

            for i in range(self.n_splits):
                oos_start = self.n_bars - oos_total + i * oos_per_split
                oos_end = min(oos_start + oos_per_split, self.n_bars)
                is_start = 0
                is_end = oos_start
                if is_end <= is_start or oos_end <= oos_start:
                    continue
                splits.append(((is_start, is_end), (oos_start, oos_end)))
        else:
            # Rolling: fixed-size window slides forward
            total_window = self.n_bars // self.n_splits
            is_size = int(total_window * self.is_ratio)
            oos_size = total_window - is_size

            for i in range(self.n_splits):
                is_start = i * total_window
                is_end = is_start + is_size
                oos_start = is_end
                oos_end = min(oos_start + oos_size, self.n_bars)
                if oos_end <= oos_start:
                    continue
                splits.append(((is_start, is_end), (oos_start, oos_end)))

        return splits

    def run(
        self,
        param_grid: list[dict[str, Any]],
        evaluate_fn: Callable[[dict[str, Any], int, int], float],
    ) -> dict:
        """Run Walk-Forward optimization.

        Parameters
        ----------
        param_grid : List of parameter dicts to test.
        evaluate_fn : Function(params, start_bar, end_bar) → metric (higher=better).

        Returns
        -------
        Dict with 'splits' (per-split results) and 'oos_metrics' (aggregated).
        """
        results = []

        for is_range, oos_range in self.splits:
            # IS optimization: find best params
            best_metric = -np.inf
            best_params = param_grid[0]

            for params in param_grid:
                metric = evaluate_fn(params, is_range[0], is_range[1])
                if metric > best_metric:
                    best_metric = metric
                    best_params = params

            # OOS evaluation with best IS params
            oos_metric = evaluate_fn(best_params, oos_range[0], oos_range[1])

            results.append({
                "is_range": is_range,
                "oos_range": oos_range,
                "best_params": best_params,
                "is_metric": best_metric,
                "oos_metric": oos_metric,
            })

        oos_metrics = np.array([r["oos_metric"] for r in results])

        return {
            "splits": results,
            "oos_metrics": oos_metrics,
            "oos_mean": float(np.mean(oos_metrics)),
            "oos_std": float(np.std(oos_metrics)),
            "oos_positive_frac": float(np.mean(oos_metrics > 0)),
        }


class CSCV:
    """Combinatorially Symmetric Cross-Validation (Bailey 2015).

    Computes PBO (Probability of Backtest Overfitting).

    Parameters
    ----------
    n_splits : Number of equal partitions (must be even, typically 10).
    """

    def __init__(self, n_splits: int = 10):
        if n_splits % 2 != 0:
            raise ValueError("n_splits must be even")
        self.n_splits = n_splits

    def run(
        self,
        param_grid: list[dict[str, Any]],
        evaluate_fn: Callable[[dict[str, Any], int, int], float],
        n_bars: int,
    ) -> dict:
        """Run CSCV analysis.

        Parameters
        ----------
        param_grid : List of parameter dicts to test.
        evaluate_fn : Function(params, start_bar, end_bar) → metric.
        n_bars : Total number of bars.

        Returns
        -------
        Dict with 'pbo', 'logit_distribution', 'n_combinations'.
        """
        S = self.n_splits
        half = S // 2
        n_params = len(param_grid)

        # Step 1: Compute metric for each (param, partition) pair
        partition_size = n_bars // S
        # metrics[p][s] = metric for param p in partition s
        metrics = np.empty((n_params, S), dtype=np.float64)

        for p, params in enumerate(param_grid):
            for s in range(S):
                start = s * partition_size
                end = start + partition_size if s < S - 1 else n_bars
                metrics[p, s] = evaluate_fn(params, start, end)

        # Step 2: Enumerate C(S, S/2) combinations
        partition_indices = list(range(S))
        combos = list(combinations(partition_indices, half))
        n_combos = len(combos)

        # Step 3: For each combination, compute IS/OOS performance
        logit_values = []
        n_overfit = 0

        for combo in combos:
            is_set = set(combo)
            oos_set = set(partition_indices) - is_set

            # IS metric per param = sum over IS partitions
            is_metrics = np.zeros(n_params, dtype=np.float64)
            oos_metrics = np.zeros(n_params, dtype=np.float64)

            for s in is_set:
                is_metrics += metrics[:, s]
            for s in oos_set:
                oos_metrics += metrics[:, s]

            # Best param in IS
            best_is = np.argmax(is_metrics)

            # Rank of best-IS param in OOS
            oos_sorted = np.argsort(oos_metrics)[::-1]  # descending
            rank_in_oos = 0
            for i, idx in enumerate(oos_sorted):
                if idx == best_is:
                    rank_in_oos = i
                    break

            # Relative rank: 0 = best, 1 = worst
            if n_params > 1:
                relative_rank = rank_in_oos / (n_params - 1)
            else:
                relative_rank = 0.0

            # Logit: log(rank / (1 - rank)), clamp to avoid inf
            eps = 1e-10
            clamped = max(eps, min(1.0 - eps, relative_rank))
            logit = np.log(clamped / (1.0 - clamped))
            logit_values.append(logit)

            # Overfit if rank > median (relative_rank > 0.5)
            if relative_rank > 0.5:
                n_overfit += 1

        pbo = n_overfit / n_combos
        logit_arr = np.array(logit_values)

        return {
            "pbo": pbo,
            "logit_distribution": logit_arr,
            "logit_mean": float(np.mean(logit_arr)),
            "logit_std": float(np.std(logit_arr)),
            "n_combinations": n_combos,
            "n_overfit": n_overfit,
        }

"""
Coupon schedule utilities for USD-settled differential swaps.

Generates semiannual (or custom frequency) coupon dates and accrual fractions.
"""

from __future__ import annotations

import numpy as np


def make_schedule(
    maturity: float,
    freq: float = 0.5,
    start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a coupon schedule for a swap with semiannual frequency.

    Parameters
    ----------
    maturity : float  Swap maturity in years.
    freq     : float  Coupon frequency in years (default 0.5 = semiannual).
    start    : float  Schedule start time (T_0, default 0.0).

    Returns
    -------
    dates : np.ndarray  Shape (n+1,).  T_0, T_1, ..., T_n  (payment/accrual dates).
    deltas: np.ndarray  Shape (n,).    Accrual fractions δ_j = T_j - T_{j-1}.
    """
    n = int(round(maturity / freq))
    dates = start + np.arange(n + 1) * freq
    deltas = np.diff(dates)
    return dates, deltas


def outstanding_coupons(t: float, dates: np.ndarray) -> np.ndarray:
    """
    Return indices j such that the j-th coupon (paid at dates[j]) is not yet settled.

    A coupon j is outstanding if t < dates[j]  (payment has not occurred).

    Parameters
    ----------
    t     : float       Current time.
    dates : np.ndarray  Full schedule T_0..T_n (length n+1).

    Returns
    -------
    np.ndarray  Integer indices into dates[1:] that are outstanding.
    """
    # Coupon j corresponds to payment at dates[j] (1-indexed from dates array).
    payment_dates = dates[1:]
    return np.where(payment_dates > t + 1e-12)[0]

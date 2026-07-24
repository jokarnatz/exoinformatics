"""
exoinformatics.py
=================
Exoplanet informatics utilities for computing planetary entropy microstates.

Author: David Kipping, Columbia University
Reference: "Do planets remember how they formed?", MNRAS (arXiv)
Refactored for Python 3 with improved readability, type hints, and robustness.
"""

import math

import numpy as np

__all__ = [
    "heaviside",
    "kronecker",
    "eulerian",
    "M0",
    "W0",
    "Omega0",
    "omegai",
    "OmegaI",
    "findk",
    "WI",
    "omegac",
    "OmegaC",
    "WC",
    "randomswap",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def heaviside(x: float) -> int:
    """Return the Heaviside step function value: 0 if *x* < 0, else 1."""
    return 0 if x < 0 else 1


def kronecker(x, y) -> int:
    """Return the Kronecker delta: 1 if *x* == *y*, else 0."""
    return 1 if x == y else 0


def eulerian(n: int, q: int) -> int:
    """
    Compute the Eulerian number ``<n, q>``.

    This counts the number of permutations of *n* elements that have
    exactly *q* ascents.
    """
    return sum(
        (-1) ** j * math.comb(n + 1, j) * (q - j + 1) ** n
        for j in range(q + 1)
    )


# ---------------------------------------------------------------------------
# Tally (T) microstate functions
# ---------------------------------------------------------------------------

def M0(N: int, T: int) -> int:
    """Return the tally microstate index for *N* planets with tally score *T*."""
    return int(0.5 * (N + T + 1))


def W0(N: int, M: int) -> int:
    """Return the tally microstate occupancy for *N* planets in state *M*."""
    return eulerian(N, M - 1)


def Omega0(N: int) -> int:
    """Return the number of distinct tally macrostates for *N* planets."""
    return N


# ---------------------------------------------------------------------------
# Integral (I) microstate functions
# ---------------------------------------------------------------------------

def omegai(N: int, M: int) -> int:
    """Return the number of integral sub-microstates for *N* planets in tally state *M*."""
    if N > 1:
        return (M - 1) * (N - M) + 1
    return 1


def OmegaI(N: int) -> float:
    """Return the total number of integral microstates for *N* planets."""
    return (15 * N ** 2 - 2 * N ** 3 - 7 * N) / 6


def findk(N: int, M: int, I: float) -> int:
    """
    Return the integral sub-microstate index *k* for a given integral score *I*.

    Parameters
    ----------
    N:
        Number of planets.
    M:
        Tally microstate index.
    I:
        Integral score.

    Returns
    -------
    int
        Sub-microstate index *k* (1-indexed).

    Raises
    ------
    ValueError
        If no sub-microstate matches *I* within a tolerance of 0.1.
    """
    length = omegai(N, M)
    absmax = 0.5 * (N - 1) ** 2 - float(np.minimum(M - 1, N - M)) ** 2
    if M <= 0.5 * N:
        iscore = sorted([-absmax + 2 * k for k in range(length)], reverse=True)
    else:
        iscore = sorted([absmax - 2 * k for k in range(length)], reverse=True)

    for k, score in enumerate(iscore):
        if abs(score - I) < 0.1:
            return k + 1

    raise ValueError(
        f"No integral sub-microstate found for N={N}, M={M}, I={I}."
    )


def WI(N: int, M: int, k: int) -> int:
    """
    Return the integral microstate occupancy for *N* planets in tally state *M*,
    sub-microstate *k*.

    Returns -1 for unsupported values of *N* (currently only N ≤ 10 are tabulated).
    """
    if M == 1 or M == N:
        return 1
    if k == 1:
        return math.comb(N - 1, M - 1)
    if M == 2 or M == N - 1:
        return math.comb(omegai(N, M) + 1, k) - 1

    if N == 5:
        if M == 3:
            return 22 if k == 3 else 16  # k == 2 or k == 4

    elif N == 6:
        if M in (3, 4):
            if k == 4:
                return 80
            if k in (3, 5):
                return 66
            return 35  # k == 2 or k == 6

    elif N == 7:
        if M == 4:
            if k in (5, 6):
                return 494
            if k in (4, 7):
                return 382
            if k in (3, 8):
                return 222
            return 90  # k == 2 or k == 9
        if M in (3, 5):
            if k == 5:
                return 269
            if k in (4, 6):
                return 233
            if k in (3, 7):
                return 149
            return 64  # k == 2 or k == 8

    elif N == 8:
        if M in (4, 5):
            if k == 7:
                return 2785
            if k in (6, 8):
                return 2540
            if k in (5, 9):
                return 1918
            if k in (4, 10):
                return 1175
            if k in (3, 11):
                return 560
            return 189  # k == 2 or k == 12
        if M in (3, 6):
            if k == 6:
                return 855
            if k in (5, 7):
                return 765
            if k in (4, 8):
                return 540
            if k in (3, 9):
                return 288
            return 105  # k == 2 or k == 10

    elif N == 9:
        if M == 5:
            if k == 9:
                return 23402
            if k in (8, 10):
                return 21916
            if k in (7, 11):
                return 17956
            if k in (6, 12):
                return 12764
            if k in (5, 13):
                return 7754
            if k in (4, 14):
                return 3918
            if k in (3, 15):
                return 1568
            return 448  # k == 2 or k == 16
        if M in (4, 6):
            if k in (8, 9):
                return 13536
            if k in (7, 10):
                return 11736
            if k in (6, 11):
                return 8767
            if k in (5, 12):
                return 5561
            if k in (4, 13):
                return 2913
            if k in (3, 14):
                return 1198
            return 350  # k == 2 or k == 15
        if M in (3, 7):
            if k == 7:
                return 2632
            if k in (6, 8):
                return 2400
            if k in (5, 9):
                return 1806
            if k in (4, 10):
                return 1091
            if k in (3, 11):
                return 503
            return 160  # k == 2 or k == 12

    elif N == 10:
        if M in (5, 6):
            if k == 11:
                return 171224
            if k in (10, 12):
                return 162826
            if k in (9, 13):
                return 139841
            if k in (8, 14):
                return 108019
            if k in (7, 15):
                return 74485
            if k in (6, 16):
                return 45297
            if k in (5, 17):
                return 23836
            if k in (4, 18):
                return 10521
            if k in (3, 19):
                return 3690
            return 924  # k == 2 or k == 20
        if M in (4, 7):
            if k == 10:
                return 63770
            if k in (9, 11):
                return 60213
            if k in (8, 12):
                return 50592
            if k in (7, 13):
                return 37597
            if k in (6, 14):
                return 24427
            if k in (5, 15):
                return 13610
            if k in (4, 16):
                return 6299
            if k in (3, 17):
                return 2295
            return 594  # k == 2 or k == 18
        if M in (3, 8):
            if k == 8:
                return 7934
            if k in (7, 9):
                return 7332
            if k in (6, 10):
                return 5756
            if k in (5, 11):
                return 3776
            if k in (4, 12):
                return 2005
            if k in (3, 13):
                return 817
            return 231  # k == 2 or k == 14

    return -1


# ---------------------------------------------------------------------------
# Change-point (C) microstate functions
# ---------------------------------------------------------------------------

def omegac(N: int, M: int) -> int:
    """Return the number of change-point sub-microstates for *N* planets in tally state *M*."""
    if N > 1:
        return max(2 * min(M - 1, N - M), 1) - kronecker(0.5 * (N - 1), M - 1.0)
    return 1


def OmegaC(N: int) -> int:
    """Return the total number of change-point microstates for *N* planets."""
    return ((N - 1) ** 2 + 3) // 2


def WC(N: int, M: int, D: int) -> int:
    """
    Return the change-point microstate occupancy for *N* planets in tally state *M*,
    change-point level *D*.

    Returns -1 for unsupported values of *N* (currently only N ≤ 10 are tabulated).
    """
    if M == 1 or M == N:
        return 1
    if D == 1:
        return 2 * math.comb(N - 1, M - 1) - kronecker(M, 1) - kronecker(M, N)
    if D == 2 and omegac(N, M) == 2:
        return W0(N, M) - (2 * math.comb(N - 1, M - 1) - kronecker(M, 1) - kronecker(M, N))

    oc = omegac(N, M)

    if N == 5:
        return 22 if D == 2 else 32  # D == 3

    elif N == 6:
        if D == 2:
            return 71
        if D == 3:
            return 150
        return 61  # D == 4

    elif N == 7:
        if D == 2:
            return 200 if oc == 4 else 220
        if D == 3:
            return 482 if oc == 4 else 888
        if D == 4:
            return 479 if oc == 4 else 724
        return 544  # D == 5

    elif N == 8:
        if D == 2:
            return 521 if oc == 4 else 629
        if D == 3:
            return 1316 if oc == 4 else 3472
        if D == 4:
            return 2414 if oc == 4 else 4897
        if D == 5:
            return 5166
        return 1385  # D == 6

    elif N == 9:
        if D == 2:
            return 1290 if oc == 4 else (1734 if oc == 6 else 1794)
        if D == 3:
            return 3292 if oc == 4 else (11240 if oc == 6 else 16032)
        if D == 4:
            return 9970 if oc == 4 else (25568 if oc == 6 else 32250)
        if D == 5:
            return 30552 if oc == 6 else 58860
        if D == 6:
            return 19028 if oc == 6 else 31242
        return 15872  # D == 7

    elif N == 10:
        if D == 2:
            return 3083 if oc == 4 else (4659 if oc == 6 else 4999)
        if D == 3:
            return 7818 if oc == 4 else (32682 if oc == 6 else 60030)
        if D == 4:
            return 32867 if oc == 4 else (115053 if oc == 6 else 173526)
        if D == 5:
            return 144972 if oc == 6 else 408438
        if D == 6:
            return 157658 if oc == 6 else 359838
        if D == 7:
            return 252750
        return 50521  # D == 8

    return -1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def randomswap(radii: list) -> list:
    """
    Return a copy of *radii* with a random adjacent pair of elements swapped.

    Parameters
    ----------
    radii:
        Ordered list of planetary radii.

    Returns
    -------
    list
        New list with one adjacent pair swapped.
    """
    last = len(radii) - 1
    swap1 = np.random.randint(0, last + 1)

    if swap1 == 0:
        swap2 = 1
    elif swap1 == last:
        swap2 = last - 1
    else:
        swap2 = swap1 + int(2.0 * (np.random.randint(0, 2) - 0.5))

    newradii = list(radii)
    newradii[swap1], newradii[swap2] = newradii[swap2], newradii[swap1]
    return newradii

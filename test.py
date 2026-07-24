"""
test.py - Example usage of the exoinformatics module.

Computes the tally (W_T), integral (W_I), and change-point (W_C) microstate
occupancies for a random array of five planetary radii.
"""

import numpy as np
import exoinformatics


def main():
    # Inputs: radii in arbitrary units, rank-ordered by semi-major axis
    # Outputs: W values (microstate occupancies); entropy S = log W
    radii = np.random.rand(5)
    print("random radii =", radii)
    N = len(radii)

    # Calculate the tally score T
    tally = [2 * exoinformatics.heaviside(radii[i + 1] - radii[i]) - 1 for i in range(N - 1)]
    tally_history = [0] * N
    for i in range(N - 1):
        tally_history[i + 1] = int(np.sum(tally[: i + 1]))
    T = int(np.sum(tally))

    # Calculate the integral score
    integral = [0.5 * (tally_history[i + 1] + tally_history[i]) for i in range(N - 1)]
    area = np.sum(integral)

    # Calculate the change-point score D
    delta = [
        1 - exoinformatics.kronecker(tally[i], tally[i + 1])
        for i in range(N - 2)
    ]
    D = int(np.sum(delta))

    # Compute microstate indices and occupancies
    m0 = exoinformatics.M0(N, T)
    w0 = exoinformatics.W0(N, m0)
    k = exoinformatics.findk(N, m0, area)

    print(
        "W_T, W_I, W_C =",
        w0,
        exoinformatics.WI(N, m0, k),
        exoinformatics.WC(N, m0, D),
    )


if __name__ == "__main__":
    main()

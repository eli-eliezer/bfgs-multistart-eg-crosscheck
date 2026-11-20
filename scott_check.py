"""Scott measure Q_m from RDM purities (Eq. (2) with the extension Eq. (3)); tolerance 1e-12."""
import itertools
from math import comb
import numpy as np
from fractions import Fraction
from eg_crosscheck import gd_state, table2_states, TOL


def purity_sum(psi, m):
    N = psi.ndim; tot = 0.0
    for S in itertools.combinations(range(N), m):
        Sc = [i for i in range(N) if i not in S]
        M = np.transpose(psi, list(S) + Sc).reshape(2 ** m, -1); r = M @ M.conj().T
        assert np.allclose(r, r.conj().T, atol=TOL) and abs(np.trace(r).real - 1) < TOL
        assert np.linalg.eigvalsh(r).min() > -TOL
        tot += np.trace(r @ r).real
    return tot


def scott(psi, m):
    N = psi.ndim
    return 2 ** m / (2 ** m - 1) * (1 - purity_sum(psi, m) / comb(N, m))


if __name__ == "__main__":
    for name, c in table2_states().items():
        psi = gd_state(len(next(iter(c))), c)
        print(f"{name:20s} Q2={scott(psi,2):.6f} ({Fraction(scott(psi,2)).limit_denominator(5000)})"
              f"  Q3={scott(psi,3):.6f} ({Fraction(scott(psi,3)).limit_denominator(5000)})")

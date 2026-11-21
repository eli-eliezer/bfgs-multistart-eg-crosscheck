"""Check of the explicit expression of sum_{|S|=m} Tr rho_S^2 for GD states (Theorem 3, Eq. (7))
against direct partial traces. RNG: numpy.random.default_rng(seed=2026); 50 random instances per (N, l);
coefficients i.i.d. complex Gaussian, normalised. Checks: Hermiticity, unit trace, PSD of every RDM (tol 1e-12).
Reports the largest absolute deviation between Eq. (7) and the direct computation."""
import itertools
import numpy as np
from math import comb

TOL = 1e-12
C = lambda a, b: comb(a, b) if 0 <= b <= a else 0


def direct(c, N, m):
    psi = np.zeros(2 ** N, complex)
    for x, v in c.items():
        psi[x] = v
    psi = psi.reshape([2] * N); tot = 0.0
    for S in itertools.combinations(range(N), m):
        Sc = [i for i in range(N) if i not in S]
        M = np.transpose(psi, list(S) + Sc).reshape(2 ** m, -1); r = M @ M.conj().T
        assert np.allclose(r, r.conj().T, atol=TOL) and abs(np.trace(r) - 1) < TOL
        assert np.linalg.eigvalsh(r).min() > -TOL
        tot += np.trace(r @ r).real
    return tot


def formula(c, N, m):
    """Eq. (7): ordered pairs (x, z); inner sum over A in D_x, B in D_z with |A| = |B| = j."""
    tot = 0j
    for x in c:
        for z in c:
            Dx = [i for i in range(N) if (x >> i) & 1 and not (z >> i) & 1]
            Dz = [i for i in range(N) if (z >> i) & 1 and not (x >> i) & 1]
            d = len(Dx)
            for j in range(d + 1):
                w = C(N - 2 * d, m - 2 * j)
                if w == 0:
                    continue
                for A in itertools.combinations(Dx, j):
                    for B in itertools.combinations(Dz, j):
                        F = sum(1 << i for i in Dx + Dz if i not in A + B)
                        tot += w * c[x] * c[z] * np.conj(c[x ^ F] * c[z ^ F])
    return tot


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    worst = 0.0
    for N, l in [(3, 1), (4, 1), (4, 2), (5, 2), (6, 2), (6, 3), (7, 3)]:
        xs = [x for x in range(2 ** N) if bin(x).count("1") == l]
        dev = 0.0
        for _ in range(50):
            v = rng.normal(size=len(xs)) + 1j * rng.normal(size=len(xs)); v /= np.linalg.norm(v)
            c = dict(zip(xs, v))
            for m in range(1, N):
                b = formula(c, N, m)
                assert abs(b.imag) < TOL
                dev = max(dev, abs(direct(c, N, m) - b.real))
        worst = max(worst, dev)
        print(f"(N,l)=({N},{l}): 50 instances, all m, max |Eq.(7) - direct| = {dev:.2e}")
    print(f"overall max deviation = {worst:.2e}")

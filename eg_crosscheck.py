"""Geometric measure E_G = 1 - max_phi |<phi|psi>|^2 over normalised N-qubit product states.

Two independent methods:
  (1) BFGS multistart (SciPy), parametrisation |phi_k> = cos(t_k)|0> + exp(i f_k) sin(t_k)|1>,
      2N real parameters, gtol = 1e-10, 200 uniformly random starts (seed fixed);
  (2) alternating (higher-order power) iteration: each single-qubit factor is replaced by the
      normalised contraction of psi with all other factors, until |change in overlap| < 1e-14.
A start is counted as reaching the optimum if its value is within 1e-8 of the best value.
String labels: qubit 1 is the leftmost bit.
Sanity checks: the state is normalised, its density matrix is Hermitian, unit trace, PSD (tol 1e-12).
"""
import itertools
import numpy as np
from scipy.optimize import minimize

TOL = 1e-12


def gd_state(N, coeffs):
    """coeffs: dict 'bitstring' -> amplitude. Returns tensor of shape (2,)*N."""
    psi = np.zeros((2,) * N, complex)
    for s, v in coeffs.items():
        psi[tuple(int(b) for b in s)] = v
    nrm = np.linalg.norm(psi)
    assert abs(nrm - 1) < TOL, nrm
    v = psi.reshape(-1); rho = np.outer(v, v.conj())
    assert np.allclose(rho, rho.conj().T, atol=TOL) and abs(np.trace(rho) - 1) < TOL
    assert np.linalg.eigvalsh(rho).min() > -TOL
    return psi


def overlap(psi, factors):
    t = psi
    for f in factors:                       # contract leading axis each time
        t = np.tensordot(f.conj(), t, axes=([0], [0]))
    return t


def factors_from(params, N):
    th, ph = params[:N], params[N:]
    return [np.array([np.cos(a), np.exp(1j * b) * np.sin(a)]) for a, b in zip(th, ph)]


def lam2_bfgs(psi, n_starts=200, seed=2026):
    N = psi.ndim; rng = np.random.default_rng(seed)
    f = lambda p: -abs(overlap(psi, factors_from(p, N))) ** 2
    vals = []
    for _ in range(n_starts):
        p0 = np.concatenate([rng.uniform(0, np.pi / 2, N), rng.uniform(0, 2 * np.pi, N)])
        r = minimize(f, p0, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
        vals.append(-r.fun)
    vals = np.array(vals); best = vals.max()
    return best, int(np.sum(vals > best - 1e-8))


def lam2_alternating(psi, n_starts=200, seed=7, tol=1e-14, maxit=10000):
    """Returns (best value, number of starts within 1e-8 of the best)."""
    N = psi.ndim; rng = np.random.default_rng(seed); vals = []
    for _ in range(n_starts):
        fac = [x / np.linalg.norm(x) for x in rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2))]
        old = -1.0
        for _ in range(maxit):
            for k in range(N):
                t = np.moveaxis(psi, k, 0)
                for f in fac[:k] + fac[k + 1:]:
                    t = np.tensordot(t, f.conj(), axes=([1], [0]))
                fac[k] = t / np.linalg.norm(t)          # optimal k-th factor given the others
            val = abs(overlap(psi, fac)) ** 2
            if abs(val - old) < tol:
                break
            old = val
        vals.append(val)
    vals = np.array(vals); best = vals.max()
    return best, int(np.sum(vals > best - 1e-8))


def table2_states():
    st = {}
    w = lambda N, l: ["".join(b) for b in itertools.product("01", repeat=N) if b.count("1") == l]
    s52 = w(5, 2)
    dom = {s: 1 / np.sqrt(18) for s in s52}; dom["11000"] = 1 / np.sqrt(2)
    dmin = dict(dom); dmin["00011"] *= -1
    st["(5,2) dominant +"], st["(5,2) dominant -"] = dom, dmin
    u42 = {s: 1 / np.sqrt(6) for s in w(4, 2)}; u42m = dict(u42); u42m["0110"] *= -1
    st["(4,2) uniform +"], st["(4,2) uniform -"] = u42, u42m
    u52 = {s: 1 / np.sqrt(10) for s in s52}; u52m = dict(u52); u52m["00011"] *= -1
    st["(5,2) uniform +"], st["(5,2) uniform -"] = u52, u52m
    return st


if __name__ == "__main__":
    for name, c in table2_states().items():
        N = len(next(iter(c)))
        psi = gd_state(N, c)
        b, hits = lam2_bfgs(psi)
        a, hits_alt = lam2_alternating(psi)
        print(f"{name:20s} Lambda2_BFGS={b:.12f} hits={hits}/200  Lambda2_alt={a:.12f} hits={hits_alt}/200  "
              f"|diff|={abs(a-b):.1e}  E_G={1-b:.6f}")

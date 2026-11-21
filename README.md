# BFGS multistart and \(E_G\) cross-check

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24422765.svg)](https://doi.org/10.5281/zenodo.24422765)

Reproducible code accompanying the article **“Explicit (ℓ+1)-Partite Reduced Density
Matrices of Generalized Dicke States and Their Determination of the Scott and
Geometric Entanglement Measures”**, published in _Zero: Jurnal Sains, Matematika dan Terapan_.

## Contents

| File | Purpose |
|---|---|
| `eg_crosscheck.py` | Computes geometric entanglement \(E_G\) for Table 2 states using 200-start BFGS and alternating/power iteration. |
| `scott_check.py` | Computes Scott measures \(Q_2\) and \(Q_3\) from reduced-density-matrix purities. |
| `purity_formula_check.py` | Checks the Theorem 3 purity-sum formula against direct partial traces. |
| `cost_count.py` | Compares the term count of the explicit formula with direct partial-trace costs. |

## Numerical settings

- Product-state parametrisation:
  \[
  |\phi_k\rangle=\cos(t_k)|0\rangle+e^{if_k}\sin(t_k)|1\rangle,
  \]
  using \(2N\) real parameters \(t_1,\ldots,t_N,f_1,\ldots,f_N\).
- BFGS via `scipy.optimize.minimize`: `gtol=1e-10`, `maxiter=10000`, 200 uniformly random starts, seed `2026`.
- A BFGS start reaches the best value when its objective value is within \(10^{-8}\) of the best BFGS value.
- Alternating iteration: 200 random starts, seed `7`, with convergence when the overlap changes by less than \(10^{-14}\).
- States and RDMs are checked for normalisation, Hermiticity, unit trace, and positive semidefiniteness at tolerance \(10^{-12}\).

## Reproduce

Python 3.10+ is required.

```bash
python3 -m pip install numpy scipy
python3 eg_crosscheck.py
python3 scott_check.py
python3 purity_formula_check.py
python3 cost_count.py
```

The scripts were also tested with Python 3.10.11, NumPy 2.2.6, and SciPy 1.13.1.

## Citation

Please cite the accompanying article and this software repository. Release `v1.0`
provides the reproducible source snapshot associated with this release.

## License

This project is licensed under the [MIT License](LICENSE).

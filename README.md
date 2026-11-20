# bfgs-multistart-eg-crosscheck

Code accompanying the article *Explicit (ℓ+1)-Partite Reduced Density Matrices of Generalized Dicke States
and Their Determination of the Scott and Geometric Entanglement Measures* (Zero: Jurnal Sains, Matematika dan Terapan).

| File | Purpose |
|---|---|
| `eg_crosscheck.py` | Geometric measure $E_G$ of the states in Table 2 by two independent methods: BFGS multistart and alternating (higher-order power) iteration |
| `scott_check.py` | Scott measures $Q_2$ and $Q_3$ of the same states from RDM purities |
| `purity_formula_check.py` | Check of the general expression of $\sum_{\vert S\vert=m}\mathrm{Tr}\,\rho_S^2$ in terms of the coefficients (Theorem 3) against direct partial traces |

## Numerical settings
- Product states: $\vert\phi_k\rangle=\cos t_k\vert0\rangle+e^{i f_k}\sin t_k\vert1\rangle$, i.e. $2N$ real parameters.
- BFGS (`scipy.optimize.minimize`): `gtol=1e-10`, `maxiter=10000`, 200 random starts, seed 2026. A start counts as reaching the optimum if it is within $10^{-8}$ of the best value.
- Alternating iteration: 200 random starts, seed 7, stopped when the overlap changes by less than $10^{-14}$.
- Every state and RDM is checked for normalisation, Hermiticity, unit trace and positive semidefiniteness at tolerance $10^{-12}$.

## Run
```
python eg_crosscheck.py
python scott_check.py
python purity_formula_check.py
```
Tested with Python 3.13, NumPy 2.5, SciPy 1.18.

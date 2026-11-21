"""Number of terms in Eq. (7) versus the cost of direct partial traces.
T(N, l) = C(N,l) * sum_{d=0}^{l} C(l,d) C(N-l,d) C(2d,d)   (ordered pairs (x,z) and choices of A, B), independent of m.
Direct: forming all C(N,m) RDMs from the 2^N state vector costs about C(N,m) * 2^(N+m) multiply-adds.
The formula for T is checked by explicit enumeration for small (N, l)."""
import itertools
from math import comb


def T_formula(N, l):
    return comb(N, l) * sum(comb(l, d) * comb(N - l, d) * comb(2 * d, d) for d in range(l + 1))


def T_enum(N, l):
    xs = [x for x in range(2 ** N) if bin(x).count("1") == l]; n = 0
    for x in xs:
        for z in xs:
            d = bin(x & ~z).count("1")
            n += sum(comb(d, j) ** 2 for j in range(d + 1))
    return n


if __name__ == "__main__":
    for N, l in [(3, 1), (4, 2), (5, 2), (6, 3), (7, 3), (8, 2)]:
        assert T_formula(N, l) == T_enum(N, l)
    print("formula verified")
    print("  N  l  m   terms Eq.(7)   direct C(N,m)2^(N+m)")
    for l in (1, 2, 3):
        for N in (10, 16, 20, 30):
            m = l + 1
            print(f"{N:3d} {l:2d} {m:2d} {T_formula(N, l):14,d} {comb(N, m) * 2 ** (N + m):22,d}")

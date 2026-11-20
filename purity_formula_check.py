"""Check of the general expression of sum_{|S|=m} Tr rho_S^2 for GD states in terms of c_x (Theorem 3). Tolerance 1e-12."""
import itertools, numpy as np
from math import comb
rng=np.random.default_rng(7)
C=lambda a,b: comb(a,b) if 0<=b<=a else 0

def direct(c,N,m):
    psi=np.zeros(2**N,complex)
    for x,v in c.items(): psi[x]=v
    psi=psi.reshape([2]*N); tot=0
    for S in itertools.combinations(range(N),m):
        Sc=[i for i in range(N) if i not in S]
        M=np.transpose(psi,list(S)+Sc).reshape(2**m,-1); r=M@M.conj().T
        assert np.allclose(r,r.conj().T,atol=1e-12) and abs(np.trace(r)-1)<1e-12
        assert np.linalg.eigvalsh(r).min()>-1e-12
        tot+=np.trace(r@r).real
    return tot

def formula(c,N,m):
    tot=0
    for x in c:
        for z in c:
            Dx=[i for i in range(N) if (x>>i)&1 and not (z>>i)&1]
            Dz=[i for i in range(N) if (z>>i)&1 and not (x>>i)&1]
            d=len(Dx)
            for j in range(d+1):
                w=C(N-2*d,m-2*j)
                if w==0: continue
                for A in itertools.combinations(Dx,j):
                    for B in itertools.combinations(Dz,j):
                        F=[i for i in Dx+Dz if i not in A+B]   # D \ T dibalik
                        mask=sum(1<<i for i in F)
                        tot+=w*c[x]*c[z]*np.conj(c[x^mask]*c[z^mask])
    return tot

for N,l in [(3,1),(4,1),(4,2),(5,2),(6,2),(6,3),(7,3)]:
    xs=[x for x in range(2**N) if bin(x).count("1")==l]
    v=rng.normal(size=len(xs))+1j*rng.normal(size=len(xs)); v/=np.linalg.norm(v)
    c=dict(zip(xs,v))
    for m in range(1,N):
        a,b=direct(c,N,m),formula(c,N,m)
        assert abs(b.imag)<1e-12 and abs(a-b.real)<1e-12,(N,l,m,a,b)
    print(N,l,"ok")

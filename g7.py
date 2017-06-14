from random import random
from random import choice
from math import sqrt, exp, e, pi, log
import lib
from scipy.stats import norm, chi2
from scipy.stats import chisquare, binom
from collections import Counter

def _____discretizacion(F, N, a, b, nint=1000): #TODO
    assert a < b
    binsize = (b - a)/nint
    Np = [0]*nint
    for v in N:
        if a <= v < b:
            Np[int((v - a)/binsize)] += 1
    #return Np
    

def estadisticoT(p, N, n=None):
    assert(len(p) == len(N) != 0)
    n = sum(N) if n == None else n
    assert(n != 0)
    k = len(p)
    T = 1/n * sum((N[i] - n*p[i])**2/p[i] for i in range(k))
    return T

def pval_disc(p, N, param_desc=0):
    # == chisquare(N, [e*sum(N) for e in p])
    T = estadisticoT(p, N)
    k = len(p)
    pval = 1 - chi2.cdf(T, df=k - 1 - param_desc)
    return T, pval

def pval_disc_sim(p, N, ns=1000):
    assert(len(p) == len(N))
    n = sum(N) # tamaño muestra inicial
    k = len(p) # cantidad categorias
    t = estadisticoT(p, N)
    def genNp():
        M = (lib.discretaX(range(k), p) for _ in range(n))
        d = dict(Counter(M))
        Np = [0] * k
        for i in d.keys():
            Np[i] = d[i]
        return Np
    simpval = sum(estadisticoT(p, genNp()) >= t for _ in range(ns))/ns
    return t, simpval

def pval_disc_sim_binom(p, N, ns=1000):
    assert(len(p) == len(N))
    n = sum(N) # tamaño muestra inicial
    k = len(p) # cantidad categorias
    t = estadisticoT(p, N)
    def genNp():
        Nacum = lib.binomial(n, p[0])
        yield Nacum
        pacum = 0
        for i in range(1, k):
            pacum += p[i-1]
            N = lib.binomial(n - Nacum, p[i]/(1 - pacum))
            yield N
            Nacum += N
    simpval = sum(estadisticoT(p, list(genNp())) >= t for _ in range(ns))/ns
    return t, simpval


def ejemplo2_1():
    N = [2, 7, 20, 22, 24, 23, 0, 2]
    p = [binom.pmf(k, n=7, p=.5) for k in range(8)]
    T, pval = pval_disc(p, N)

def ej1a():
    p, N = [1/4, 1/2, 1/4], [141, 291, 132]
    T, pval = pval_disc(p, N)
    #return "T: {0}, pval: {1}".format(T, pval)
    return T, pval

def ej1b(ns=1000):
    p, N = [1/4, 1/2, 1/4], [141, 291, 132]
    return pval_disc_sim_binom(p, N, ns=ns)

def ej2(ns=1000):
    p, N = [1/6]*6, [158, 172, 164, 181, 160, 165]
    chi2 = pval_disc(p, N)
    print("chi2: {0}".format(chi2))
    sim = pval_disc_sim(p, N, ns=ns)
    print("simulacion: {0}".format(sim))

def KS(F, M, ns=500):
    # KS(lambda x: 1 - exp(-x/100), [66, 72, 81, 94, 112, 116, 124, 140, 145, 155])
    # == 0.012
    M.sort()
    n = len(M)
    def est_d(l):
        def elem():
            for i, e in enumerate(l):
                yield (i + 1)/n - e
                yield e - i/n
        return max(elem())
    
    d = est_d([F(y) for y in M])
    pval = sum(est_d(sorted(random() for _ in range(n))) >= d for _ in range(ns))/ns
    return d, pval

def ejemplo2_2():
    return KS(lambda x: 1 - exp(-x/100), 
        [66, 72, 81, 94, 112, 116, 124, 140, 145, 155], ns=500)
    # == 0.012

def ej3():
    muestra = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
    return KS(lambda x: x, muestra, ns=10000)

def ej4():
    muestra = [86, 133, 75, 22, 11, 144, 78, 122, 8, 146, 33, 41, 99]
    return KS(lambda x: 1 - exp(-x/50), muestra, ns=100000)

def ej6():
    return KS(lambda x: 1 - exp(-x),
        [lib.exponencial(1) for _ in range(10)], ns=100000)

def ejemplo2_3():
    """
    6 dias 0 acc
    2 dias 1 acc
    1 dia  2
    9 dias 3
    7 dias 4
    4 dias 5
    1 dia  8
    total_acc = 6*0 + 2*1 + 1*2 + 9*3 + 7*4 + 4*5 + 1*8 = 87
    total_dias = 30
    """
    n = 30
    lamb = 87/n
    g = lib.poissonseriesgen(lamb)
    P = [next(g) for _ in range(5)]
    N = [6, 2, 1, 9, 7, 5] 
    #p= [P(0), P(1), P(2), P(3), P(4), P(X>=5)]
    # P(X>=5) = 1 - P(X <= 4) 
    p = [P[0], P[1], P[2] ,P[3] ,P[4], 1 - sum(P[:5])]
    t, pval_chi = pval_disc(p, N, param_desc=1)
    print("t: {0}".format(t))
    print("chi-test pval: {0}".format(pval_chi))
    M = [lib.Poisson(lamb) for _ in range(n)]
    def genNp():
        k = 6
        d = dict(Counter(M))
        Np = [0] * k
        for i in d.keys():
            Np[i] = d[i]
        return Np
    lamb = sum(M)/30
    g = lib.poissonseriesgen(lamb)
    P = [next(g) for _ in range(5)]
    p = [P[0], P[1], P[2] ,P[3] ,P[4], 1 - sum(P[:5])]
    print(p)
    print(genNp())
    t = estadisticoT(p, genNp())
    print("t: {0}".format(t))


def pval_disc_sim_pd(M, randvar, estimar_parametros, f, n, sizimg, param_desc, ns=10):
        def estadisticoTdeM(M):
            pest = estimar_parametros(M)
            d = dict(Counter(M))
            N = [0] * sizimg
            for i in d.keys():
                N[i] = d[i]
            p = [f(k, *pest) for k in range(sizimg)]
            return estadisticoT(p, N, n=n), pest
        
        k = len(M)
        t, pest = estadisticoTdeM(M)
        print("t: {0}".format(t))
        print("chi-test: {0}".format(1 - chi2.cdf(t, df=k - 1 - param_desc)))
        def genEst(k):
            M = [randvar(pest) for _ in range(k)]
            #M = binom.rvs(8, pest, size=k)
            return estadisticoTdeM(M)[0]
        return sum(genEst(k) >= t for _ in range(ns))/ns


def ej5(ns=10000):
    M = [6,7,3,4,7,3,7,2,6,3,7,8,2,1,3,5,8,7]
    return pval_disc_sim_pd(M, lambda p: binom.rvs(n=8, p=p),
        lambda M: ((sum(M)/len(M))/8,), 
        lambda k, p: binom.pmf(k, n=8, p=p), len(M), 9, 1, ns=ns)

def pval_cont_sim_pd(M, estimar_parametros, F, randvar, ns):
        print('M orig: m:{0}, v:{1}'.format(lib.media(M), lib.varianza(M)))
        M.sort()
        k = len(M)
        def est_d(l):
            def elem():
                for i, e in enumerate(l):
                    yield (i + 1)/k - e
                    yield e - i/k
            return max(elem())
        pest = estimar_parametros(M)
        d = est_d([F(y, *pest) for y in M])
        print("d: {0}, pest: {1}".format(d, pest))
        #return sum(est_d(sorted(random() for _ in range(k))) >= d for _ in range(ns))/ns
        def generar_M_y_Estimar_d():
            M = sorted(randvar(*pest) for _ in range(k))
            #print('M gen: m:{0}, v:{1}'.format(lib.media(M), lib.varianza(M)))
            pest_sim = estimar_parametros(M)
            return est_d([F(y, *pest_sim) for y in M])
        #return generar_M_y_Estimar_d()
        return sum(generar_M_y_Estimar_d() >= d for _ in range(ns))/ns


def ej7(ns=100000):
    M = [1.6,10.3,3.5,13.5,18.4,7.7,24.3,10.7,8.4,4.9,7.9,12,16.2,6.8,14.7]
    return pval_cont_sim_pd(M,
        lambda M: (1/lib.media(M),),
        lambda x, p: 1 - exp(-x*p), 
        lambda lam: lib.exponencial(lam), ns)

def ej8(ns=100000):
    M = [91.9, 97.8, 111.4, 122.3, 105.4, 95.0, 103.8, 99.6, 96.6, 119.3, 104.8, 101.7]
    uest = lib.media(M)
    vest = lib.varianza(M)
    sest = sqrt(vest)
    M = [(m - uest)/sest for m in M]
    return KS(lambda x: norm(0, 1).cdf(x), M, ns=ns)
    #return pval_cont_sim_pd(M,
    #    lambda M: (lib.media(M), lib.varianza(M, adjusted=False)),
    #    lambda x, u, v: norm(u, v).cdf(x), 
    #    lambda u, v: norm.rvs(u, v), ns)

def rangos(n, m, r):
    if n == 1 and m == 0:
        if r <= 0: 
            return 0
        else:
            return 1
    elif n == 0 and m == 1:
        if r < 0: return 0
        else: return 1
    else:
        if n == 0:
            return rangos(0, m - 1, r)
        elif m == 0:
            return rangos(n - 1, 0, r-n)
        else:
            return n/(n+m)*rangos(n-1,m,r-n-m)+m/(n+m)*rangos(n,m-1,r)
from random import random
from random import choice
from math import sqrt, exp, e, pi, log
import lib
from scipy.stats import norm, chi2
from scipy.stats import chisquare, binom
from collections import Counter

"""
alpha = 1 - confianza/100
pval <= alpha => Rechazar

geom
p(k) = p*q**k   u=1/p   v=(1-p)/p**2

poisson
p(x=i) = exp(-lamb)*lamb**i/i!    i>=0      u=v=lamb

MLE
L(par) = prod_i=1^n p_par(Xi)

uniforme
f(x) = 1/(b-a)  u=(a+b)/2   v=(b-a)**2/2
"""

def determinar(pval=None, confianza=None):
    """
    confianza: de que no voy a rechazar una hipotesis correcta
    hay un (100 - confianza)% de prov que rechaze una hipotesis correcta
    """
    alpha = 1 - confianza/100
    return "Wacale!" if pval <= alpha else "No Rechazo"

def _____discretizacion(F, N, a, b, nint=1000): #TODO
    assert a < b
    binsize = (b - a)/nint
    Np = [0]*nint
    for v in N:
        if a <= v < b:
            Np[int((v - a)/binsize)] += 1
    #return Np
    

def estadisticoT(p, N, n=None):
    """
    H0 Cierta => T ~ chi2_k-1 as n -> inf
    muy grande => se rechaza
    N: cantidad de ocurrencias de cada categoria
    p: probabilidad de cada categoria
    n: tamaño de la muestra, default: sum(N)
    """
    assert(len(p) == len(N) != 0)
    n = sum(N) if n == None else n
    assert(n != 0)
    k = len(p)
    T = 1/n * sum((N[i] - n*p[i])**2/p[i] for i in range(k))
    return T

def pval_disc(p, N, param_desc=0):
    """
    Prueba de bondad de ajuste, datos discretos, estimacion chi2
    p: lista de probabilidades de cada categoria
    N: lista de cantidades de cada categoria

    pval  = P(chi2_k-1-m >= T)
    """
    # == chisquare(N, [e*sum(N) for e in p])
    T = estadisticoT(p, N)
    k = len(p)
    pval = 1 - chi2.cdf(T, df=k - 1 - param_desc)
    return T, pval

def pval_disc_sim(p, N, ns=1000):
    """
    Prueba de bondad de ajuste, datos discretos, parametros especificados
    """
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
    """
    Determinar si N ~ Bin(7, .5)
    """
    N = [2, 7, 20, 22, 24, 23, 0, 2]
    p = [binom.pmf(k, n=7, p=.5) for k in range(8)]
    T, pval = pval_disc(p, N)

def ej1a():
    """
    3 categorias, cantidad de encontradas en c/u
    """
    p, N = [1/4, 1/2, 1/4], [141, 291, 132]
    T, pval = pval_disc(p, N)
    #return "T: {0}, pval: {1}".format(T, pval)
    return T, pval

def ej1b(ns=1000):
    p, N = [1/4, 1/2, 1/4], [141, 291, 132]
    return pval_disc_sim_binom(p, N, ns=ns)

def ej2(ns=1000):
    """
    idem
    """
    p, N = [1/6]*6, [158, 172, 164, 181, 160, 165]
    pval_chi2 = pval_disc(p, N)
    print("chi2: {0}".format(pval_chi2))
    sim = pval_disc_sim(p, N, ns=ns)
    print("simulacion: {0}".format(sim))

def KS(F, M, ns=500):
    """
    Para datos continuos.
    F: CDF de la distr
    M: Muestra
    ns: Numero de simulaciones
    """
    # KS(lambda x: 1 - exp(-x/100), [66, 72, 81, 94, 112, 116, 124, 140, 145, 155])
    # == 0.012
    M.sort()
    n = len(M)
    def est_d(l):
        """
        # D = max_1<=j<=n {j/n - F}
        D = sup_x | Fe(x) - F(x) |
        """
        def elem():
            for i, e in enumerate(l):
                yield (i + 1)/n - e
                yield e - i/n
        return max(elem())
    
    d = est_d([F(y) for y in M])
    pval = sum(est_d(sorted(random() for _ in range(n))) >= d for _ in range(ns))/ns
    return d, pval

def ejemplo2_2():
    """
    los datos son exponenciales con media 100 => lamb = 1/100
    f(x) = 1/100*exp(-1/100*x)
    F(x) = int_0^x f(x) = 1 - exp(-x/100)
    """
    return KS(lambda x: 1 - exp(-x/100), 
        [66, 72, 81, 94, 112, 116, 124, 140, 145, 155], ns=500)
    # == 0.012

def ej3():
    """
    f(x) = 1
    """
    muestra = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
    return KS(lambda x: x, muestra, ns=10000)

def ej4():
    """
    media 50 => lamb= 1/50
    """
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


def pval_disc_sim_pd(M, randvar, estimar_parametros, f, sizimg, param_desc, ns=10):
    """
    VA discreta, parametros desconocidos
    M: muestra
    randvar: funcion cuyos parametros son *estimar_parametros(M) y genera una variate
    estimar_parametros: funcion: Muestra -> (p1, p2, ...)
    f: funcion: PMF 
    sizimg: tamaño de Img(randvar) (cantidad de categorias)
    param_desc: cantidad de param desc
    ns: numero de simulaciones
    """
    def estadisticoTdeM(M):
        pest = estimar_parametros(M)
        d = dict(Counter(M))
        N = [0] * sizimg
        for i in d.keys():
            N[i] = d[i]
        p = [f(k, *pest) for k in range(sizimg)]
        return estadisticoT(p, N, n=len(M)), pest
    
    k = len(M)
    t, pest = estadisticoTdeM(M)
    print("t: {0}".format(t))
    print("chi-test: {0}".format(1 - chi2.cdf(t, df=k - 1 - param_desc)))
    def genEst(k):
        #M = [randvar(pest) for _ in range(k)]
        M = binom.rvs(4, pest[0], size=k)
        return estadisticoTdeM(M)[0]
    return sum(genEst(k) >= t for _ in range(ns))/ns


def ej5(ns=10000):
    """
    los sig datos corresponden a una X ~ binomial(n=8,p=nose)
    media(X) = np => p = media(X)/n
    """
    M = [6,7,3,4,7,3,7,2,6,3,7,8,2,1,3,5,8,7]
    return pval_disc_sim_pd(M, lambda p: binom.rvs(n=8, p=p),
        lambda M: (lib.media(M)/8,),
        lambda k, p: binom.pmf(k, n=8, p=p), 9, 1, ns=ns)

def pval_cont_sim_pd(M, estimar_parametros, F, randvar, ns, usandouniformes=False):
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
        if usandouniformes:
            return sum(est_d(sorted(random() for _ in range(k))) >= d for _ in range(ns))/ns
        def generar_M_y_Estimar_d():
            M = sorted(randvar(*pest) for _ in range(k))
            #print('M gen: m:{0}, v:{1}'.format(lib.media(M), lib.varianza(M)))
            pest_sim = estimar_parametros(M)
            return est_d([F(y, *pest_sim) for y in M])
        return sum(generar_M_y_Estimar_d() >= d for _ in range(ns))/ns


def ej7(ns=100000, usandouniformes=False):
    """
    de una exponencial
    """
    M = [1.6,10.3,3.5,13.5,18.4,7.7,24.3,10.7,8.4,4.9,7.9,12,16.2,6.8,14.7]
    return pval_cont_sim_pd(M,
        lambda M: (1/lib.media(M),),
        lambda x, p: 1 - exp(-x*p), 
        lambda lam: lib.exponencial(lam), ns, usandouniformes=usandouniformes)

def ej8(ns=1000):
    M = [91.9, 97.8, 111.4, 122.3, 105.4, 95.0, 103.8, 99.6, 96.6, 119.3, 104.8, 101.7]
    #uest = lib.media(M)
    #vest = lib.varianza(M)
    #sest = sqrt(vest)
    #M = [(m - uest)/sest for m in M]
    #return KS(lambda x: norm(0, 1).cdf(x), M, ns=ns)
    return pval_cont_sim_pd(M,
        lambda M: (lib.media(M), lib.varianza(M, adjusted=False)),
        lambda x, u, v: norm(u, v).cdf(x), 
        lambda u, v: norm.rvs(u, v), ns)


def rango(x, nm_list):
    return sum(nm_list.index(k) + 1 for k in x)

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

def valor_p(n, m, r):
    return 2 * min([rangos(n, m, r), 1 - rangos(n, m, r - 1)])

def by_simulation(x, y):
    n = len(x)
    m = len(y)
    return sum(int((n+m)*random()) + 1 for j in range(n))

def ej9(ns=100000):
    x = [65.2, 67.1, 69.4, 78.4, 74, 80.3]
    y = [59.4, 72.1, 68, 66.2, 58.5]
    n = len(x)
    m = len(y)
    xy = sorted(x + y)
    r = rango(x, xy)
    pval = valor_p(n, m, r)
    print("pval exacto: \t{0}".format(pval))
    pvalsim = sum(by_simulation(x, y) > r for _ in range(ns))/ns
    print("pval simulacion: \t{0}".format(pvalsim))

def test_pval(pvalcreator, confianza, ns=1000):
    """
    pvalcreator: pvalores done la hipotesis nula es siempre cierta,
        es decir los datos tienen distr F.
    """
    def rechaze():
        return determinar(pval=pvalcreator(), confianza=confianza) == "Wacale!"
    prob_rech = sum(rechaze() for _ in range(ns))/ns
    print("si los pvalores provienen siempre de muestras de la distr H0, entonces")
    print("el porcentaje de rechazo {0} debe ser cercano a ".format(prob_rech*100))
    print("(100 - confianza) que vale {0}".format(100-confianza))

def test_ej1a(ns=100):
    p = [1/4, 1/2, 1/4] 
    def gen():
        N = [0]*3
        for i in range(564):
            i = lib.discretaX([0,1,2], p)
            N[i] += 1
        return N
    def pvalcreator():
        N = gen()
        #t, pval = pval_disc(p, N, param_desc=0)
        t, pval = pval_disc_sim_binom(p, N, ns=1000)
        return pval
    test_pval(pvalcreator, 75, ns=ns)

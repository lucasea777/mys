from random import random
from random import choice
from math import sqrt, exp, e, pi, log
from scipy.stats import norm, chi2, binom
from collections import Counter

#import lib
def exponencial(lamb, U=None): #testme
    U = random() if U is None else U
    return -log(U)/lamb

def normal(mu=0, sigma=1):
    while True:
        U1, U2 = random(), random()
        X = exponencial(1)
        if log(U1) < -(X-1)**2/2:
            return (-1 if U2 < .5 else 1)*sigma * X + mu

def media(list):
    return sum(list)/len(list)

def varianza(data, adjusted=True):
    """Return sum of square deviations of sequence data.
    """
    c = media(data)
    return sum((x-c)**2 for x in data)/(len(data) - adjusted)

def media_gen(data_gen):
    """
    returns: ((X̄_1, d1), (X̄_2, d2), ..., (X̄_N, dN))
    """
    A = 0
    for i, d in enumerate(data_gen):
        A = (A*i + d)/(i+1)
        yield A, d

def media_var(data_gen):
    """
    returns: ((X̄_1, None, d1), (X̄_2, S²_2, d2), ..., (X̄_N, S²_N, dN))
    """
    mgen = media_gen(data_gen)
    d1, d1 = next(mgen)
    yield d1, None, d1
    X2, d2 = next(mgen)
    XA = X2 # X anterior
    A = (d1 - X2)**2 + (d2 - X2)**2
    yield X2, A, d2
    for i, (X, d) in enumerate(mgen, 2):
        A = (1 - 1/i)*A + (1 + i)*(X - XA)**2
        XA = X
        yield X, A, d

def binomial(n, p):
    U = random()
    i = 0
    if(p == 1):
        return n
    c = p/(1 - p)
    prob = (1 - p) ** n
    F = prob
    while U >= F:
        prob = c * ((n-i)/(i+1)) * prob
        F += prob
        i += 1
    return i

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
    def genNp(p):
        Nacum = binomial(len(M), p[0])
        yield Nacum
        pacum = 0
        for i in range(1, 5):
            pacum += p[i-1]
            N = binomial(len(M) - Nacum, p[i]/(1 - pacum))
            yield N
            Nacum += N
    def estadisticoTdeM(M):
        pest = estimar_parametros(M)
        d = dict(Counter(M))
        N = [0] * sizimg
        for i in d.keys():
            N[i] = d[i]
        p = [f(k, *pest) for k in range(sizimg)]
        #N = list(genNp(p))
        return estadisticoT(p, N, n=len(M)), pest
    
    k = len(M)
    t, pest = estadisticoTdeM(M)
    #print("t: {0}".format(t))
    #print("chi-test: {0}".format(1 - chi2.cdf(t, df=k - 1 - param_desc)))
    def genEst(k):
        M = [randvar(*pest) for _ in range(k)]
        #M = binom.rvs(4, pest[0], size=k)
        return estadisticoTdeM(M)[0]
    #print(genEst(k))
    #return 0
    return sum(genEst(k) > t for _ in range(ns))/ns

def ej1(ns=100, confianza=95, M=[1,1,0,0,4,0,1,3,0,1,2,1,1,0,1,1,0,2,1,1]):
    """
    los sig datos corresponden a una X ~ binomial(n=4,p=nose)
    media(X) = np => p = media(X)/n
    """
    def sim():
        #M = [6,7,3,4,7,3,7,2,6,3,7,8,2,1,3,5,8,7]
        #return pval_disc_sim_pd(M, lambda p: binomial(4, p),
        return pval_disc_sim_pd(M, lambda p: binom.rvs(n=4, p=p),
            lambda M: (media(M)/4,),
            lambda k, p: binom.pmf(k, n=4, p=p), 5, 1, ns=ns)
    pval = sim()
    return pval
    print("p-val: {0}".format(pval))
    print("alpha: {0}".format(1 - confianza/100))
    if pval <= (1 - confianza/100):
        print("Se rechaza la hipotesis, es decir, no se puede afirmar")
    else:
        print("no se puede afirmar, pero tampoco rechazar")

def _pej1():
    """
    Determinar si N ~ Bin(7, .5)
    """
    M = [1,1,0,0,4,0,1,3,0,1,2,1,1,0,1,1,0,2,1,1]
    d = dict(Counter(M))
    Np = [0] * 5
    for i in d.keys():
        Np[i] = d[i]
    print(Np)
    assert(Np == [6, 10, 2, 1, 1])
    p = [binom.pmf(k, n=4, p=media(M)/4) for k in range(5)]
    T, pval = pval_disc(p, Np)
    return T, pval

def ej3(confianza=95):
    """
    los datos son exponenciales con media 1/11 => lamb = 11
    f(x) = 11*exp(-11*x)
    F(x) = int_0^x f(x) = 1 - exp(-11*x)
    """
    def sim():
        return KS(lambda x: 1 - exp(-11*x), 
            [.06, .02, .18, .17, .08, .13, .22, .07, .12, .21, .03], ns=500)
    d, pval = sim()
    print("estadisto KS: {0}".format(d))
    print("p-val: {0}".format(pval))
    print("alpha: {0}".format(1 - confianza/100))
    if pval <= (1 - confianza/100):
        print("Se rechaza la hipotesis, es decir, no se puede afirmar")
    else:
        print("no se puede afirmar, pero tampoco rechazar")

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

def valor_p_distribucion_R(n, m, r):
    N = n + m
    rr = (r - (n * (N + 1)) / 2.0) / (sqrt((n * m * (N + 1)) / 12.0))
    if r <= n * (N + 1) / 2.0:
        return 2 * norm.cdf(rr)
    else:
        return 2 * (1 - norm.cdf(rr))

def ej4(ns=10000, confianza=95):
    x = [141, 132, 154, 142, 143, 150, 134, 140]
    y = [133, 138, 136, 125, 135, 130, 127, 131, 116, 128]
    n = len(x)
    m = len(y)
    xy = sorted(x + y)
    r = rango(x, xy)
    pval = valor_p(n, m, r)
    print("pval exacto: \t{0}".format(pval))
    print("pval, aproximacion normal: \t{0}".format(valor_p_distribucion_R(n, m, r)))
    print("alpha: {0}".format(1 - confianza/100))
    if pval <= (1 - confianza/100):
        print("Se rechaza la hipotesis, es decir, no se puede afirmar")
    else:
        print("no se puede afirmar, pero tampoco rechazar")

def ej2b(d=.01):
    """
    sigma**2 = 1
    """
    def chi2_():
        return 9 * varianza([normal() for _ in range(9)])
    def gen():
        while True:
            yield 0 < chi2_() < 5
            #yield 0 < chi2.rvs(df=9) < 5

    def ej2b():
        for n, (m, V, _) in enumerate(media_var(gen()), 1):
            if n >= 30 and sqrt(V/n) < d:
                return n, m, sqrt(V/n)
    
    n, m, s = ej2b()
    print("p = {0}".format(m))

def ej2c(N=10000):
    """
    p = P(0 < chi2 < 5)
    Estimar p con bootstrap
    """
    # asumo que es una muestra de la chi2
    Xi = [6.422, 7.968, 2.287, 5.679, 5.740, 7.254 ,3.126, 3.443, 7.702, 5.680]
    n = len(Xi)
    xb = sum(Xi)/n
    def Y():
        #return 9 * varianza([choice(Xi) for _ in range(9)])
        return choice(Xi)
    return sum(0 < Y() < 5 for _ in range(N))/N


from g7 import pval_disc_sim_binom, test_pval, pval_disc
import lib
def test_ej1(ns=100):
    p = .2625
    def pvalcreator():
        #pval = ej1(ns=100, M=[binomial(4, p) for _ in range(20)])
        pval = ej1(ns=100, M=binom.rvs(n=4, p=p, size=20))
        #t, pval = pval_disc_sim_binom(p, N, ns=1000)
        return pval
    test_pval(pvalcreator, 95, ns=ns)
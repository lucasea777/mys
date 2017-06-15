from random import random
from math import sqrt, exp, e, pi, log

mcarlo = lambda g,n: sum(g(random()) for _ in range(n))/n

def _discretaX(x, p, U=None):
    U = random() if U is None else U
    i = 0
    F = p[0]

    while U >= F:
        i += 1
        F += p[i]

    return x[i]

def discretaX(x, p, U=None):
    return transf_inv(iter(x), iter(p), U=U)

def exponencial(lamb, U=None): #testme
    U = random() if U is None else U
    return -log(U)/lamb

def udiscreta(a, b):
    return int(random()*(b - a + 1)) + a

def permutacion(x): #FIXME
    N = len(x)
    for j in range(N-1, 0, -1):
        i = udiscreta(0, j)
        # i = int(j * random())
        # print("i:{0} j:{1}".format(i, j))
        x[j], x[i] = x[i], x[j]
    return x


def media(list):
    return sum(list)/len(list)

def ds(data):
    return sqrt(varianza(data))

def varianza(data, adjusted=True):
    """Return sum of square deviations of sequence data.
    """
    c = media(data)
    return sum((x-c)**2 for x in data)/(len(data) - adjusted)

def Poisson(l, U=None):
        U = random() if U is None else U
        i = 0
        p = exp(-l)
        F = p
        while F <= U:
                i += 1
                p *= l/i
                F += p
        return i

def poissonseriesgen(l):
    p = exp(-l)
    i = 0
    while True:
        yield p
        p *= l/(i+1)
        i += 1

def Poisson2(l, U=None):
    """
    >>> U=random(); l=udiscreta(0,4);Poisson(l, U=U) == Poisson2(l, U=U)
    True
    """
    return transf_inv((x for x in range(10**10)), poissonseriesgen(l), U=U)


def transf_inv(xgen, pgen, U=None): # bug cuando n == 2 ??
    U = random() if U is None else U
    F = next(pgen)
    x = next(xgen)
    while F <= U:
        x = next(xgen)
        F += next(pgen)
    return x

def met_comp(p1, p2, alfa, U=None):
    U = random() if U is None else U
    if U < alfa:
        return p1
    else:
        return p2

def met_comp_cont(p, Fto1, U=None):
    assert sum(p) == 1
    U = random() if U is None else U
    acum = p[0]
    i = 0
    while acum <= U:
        i += 1
        acum += p[i]
    return Fto1[i](U)

def eventosPoisson(T, lamb):
    t = 0
    S = []
    while True:
        U = random()
        if t - log(U)/lamb > T:
            break
        else:
            t += - log(U)/lamb
        S.append(t)
    return S

def noHomo(T, lambf, lambc, lenlimit=None):
    t = 0
    S = []
    while True:
        t += exponencial(lambc)
        if t > T:
            break
        V = random()
        if V < lambf(t)/lambc:
            S.append(t)
        if lenlimit and len(S) == lenlimit:
            break
    return S

def N(T, lamb):
    return len(eventosPoisson(T, lamb))

def NnoHomo(T, lambf, lambc):
    return len(noHomo(T, lambf, lambc))

def normal(mu=0, sigma=1):
    while True:
        U1, U2 = random(), random()
        X = exponencial(1)
        if log(U1) < -(X-1)**2/2:
            return (-1 if U2 < .5 else 1)*sigma * X + mu

#def normal_ayr(mu=0, sigma=1):
#    return ayr(lambda: exponencial(1), 
#        lambda x: 1/sqrt(2*pi)*exp(-x**2/2), 
#        lambda x: exp(-x), sqrt(e/pi))*sigma + mu

def raizn(n):
    M = 0
    for _ in range(n):
        U = random()
        if M < U:
            M = U
    return M

def ayr(genY,f,g,c):
    """
    Tengo Y con dens g, tq:
    f(y)/g(y) <= c     paratodo y pert R tq f(y) != 0
    Este metodo genera una X con dens f
    """
    while True:
        Y = genY()
        U = random()
        if U < f(Y)/(c*g(Y)):
            return Y

def gamma(n=3, lamb=2):
    U = 1
    for _ in range(n):
        U *= random()
    return -log(U)/lamb

def __media_gen(x1):
    A = 0
    d = x1
    i = 0
    while True:
        A = (A*i + d)/(i+1)
        d = yield A
        i += 1

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

def intdeconfXb(RANDVAR, n, confianza, show=False):
        """
        μ ∊ X̄ ± Ƶ_α/2 * S/√n

        n: tamaño muestra
        confianza: en %
        """
        #print("S(N̄) = √V(N̄) = S(N)/√n = {0}".format(S))
        # V(N̄) = "la varianza del estimador" = V(N)/n
        # SNb = sqrt(V/n) # V(N̄) = V(N)/n ^ S(N̄) = √V(N̄) => S(N̄) = √(V(N)/n)
        for X, VN, _ in lib.media_var((RANDVAR() for _ in range(n))):
            pass
        VNb = VN/n                   # V(N̄) = V(N)/n
        SNb = sqrt(VNb)              # S(N̄) = √(V(N)/n) = S/√n
        alpha = 1 - confianza/100
        z = norm.ppf(alpha/2)
        if show:
            print("μ ∊ {0} ± {1} con {2}% de confianza".format(X, z * SNb, confianza))
            print("V(N̄) = {0}".format(VNb))
        return X, z * SNb, VNb

def binomial(n, p): # pasar a transf_inv
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

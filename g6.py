from random import random
from random import choice
from math import sqrt, exp, e, pi, log
import lib
from scipy.stats import norm

#mcarlo = lambda g,n: sum(g(random()) for _ in range(n))/n

def ej1(d=0.1):
    def gen():
        while True:
            yield lib.normal()

    for n, (m, V, _) in enumerate(lib.media_var(gen()), 1):
        if n >= 30 and sqrt(V/n) < d:
            return n, m, sqrt(V/n)

def ej2(d=.01):
    def gen():
        while True:
            yield exp(random()**2)

    for n, (m, V, _) in enumerate(lib.media_var(gen()), 1):
        if n >= 100 and sqrt(V/n) < d:
            return n, m, sqrt(V/n)

def ej3(confianza=95, n=1000):
    """
    l = (g6.ej3(n=1000, confianza=90) for _ in range(1000))
    sum(j[0]+j[1] <= e <= j[0]-j[1] for j in l)/1000*100
    89.9
    """
    def sumuntilone():
        n = s = 0
        while s <= 1:
            s += random()
            n += 1
        return n

    for X, V, _ in lib.media_var(sumuntilone() for _ in range(n)):
        pass

    VNb = V/n
    SNb = sqrt(VNb) # S(N̄)
    #print("V(N) = V(N1, ..., Nn) = {0}".format(V))
    #print("V(N̄) = V(N)/n = {0}".format(VNb))
    #print("S(N̄) = √V(N̄) = S(N)/√n = {0}".format(S))

    alpha = 1 - confianza/100
    z = norm.ppf(alpha/2) # z_[alpha/2] = {x: I(x) = alpha/2}
    #print("X̄ ∊ {0} ± {1} con {2}% de confianza".format(X, z*S, confianza))
    return X, z * SNb

def ej4(n=1000, confianza=95):
    # sum(j[0]+j[1] <= e <= j[0]-j[1] for j in (g6.ej4(n=1000, confianza=98) for _ in range(1000)))/1000*100
    # 98
    def primermenor():
        n = A = 0
        while True:
            n += 1
            U = random()
            if U < A:
                return n
            A = U
    
    return lib.intdeconfXb(primermenor, n, confianza)
    
def ej5(l=0.1, confianza=95):
    # sum(j[0]+j[1] <= pi <= j[0]-j[1] for j in (ej5(l=0.1, confianza=82) for _ in range(1000)))/1000*100
    # 82
    def isInside():
        while True:
            yield sqrt(random()**2 + random()**2) < 1
    
    alpha = 1 - confianza/100
    z = norm.ppf(alpha/2)
    for n, (m, V, _) in enumerate(lib.media_var(isInside()), 1):
        if n >= 100 and -2*z*sqrt(V/n) < l:
            return m * 4, -4*z*sqrt(V/n)


def ej6(N=1000, a=-5, b=5):
    Xi = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
    n = len(Xi)
    xb = sum(Xi)/n
    def Y():
        return sum(choice(Xi) for _ in range(n))/n
    return sum(a + xb < Y() < b + xb for _ in range(N))/N
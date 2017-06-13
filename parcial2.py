from random import random
from math import e, factorial, sqrt, pi, exp, log

def exponencial(lamb, U=None):
    U = random() if U is None else U
    return -log(U)/lamb

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

def ej4(U=None):
    """
    X=F^-1(U) tiene distribucion f(x) para U~U[0,1]
    F(x) = int from -inf to x f(x)
    """
    U = random() if U is None else U
    if U < 1/2:
        return sqrt(2)*sqrt(U)
    else:
        return -sqrt(2-2*U)+2

def ej2():
    """retorno lista de tiempos de llegada"""
    return noHomo(8, lambda x: x, 8)
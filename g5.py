import lib, libnat
from random import random
from lib import mcarlo
from math import e, factorial, sqrt, pi, exp, log
from itertools import chain
from functools import partial

def ej1(U=None):
    """
            ⎧   0     for x < 2
            ⎪                  
            ⎪ x                
            ⎪ ─ - 1   for x ≤ 3
    F(x) =  ⎨ 2                
            ⎪                  
            ⎪  x               
            ⎪- ─ + 1  for x ≤ 6
            ⎩  6    
    X=F^-1(U) tiene distribucion f(x) para U~U[0,1]
    F(x) = int from -inf to x f(x)
    """
    U = random() if U is None else U
    if U < 1/4:
        return 2*sqrt(U) + 2
    else:
        return -2*sqrt(-3*U + 3) + 6

def ej2(a, b, U=None):
    """X=F^-1(U) tiene distribucion f(x) para U~U[0,1]"""
    U = random() if U is None else U
    return (-log(1-U)/a)**(1/b)

def ej3():
    """"""
    def Fm1(U):
        if U < 1/4:
            return 2*sqrt(U) + 2
        else:
            return -2*sqrt(-3*U + 3) + 6
    return lib.met_comp_cont([1/3, 2/3], [Fm1]*2)


def ej5():
    """  
    X = max{X_i}
    F_X(y) = P(x1<=y,...,xn<=y)
           = prod P(X_i<=y) from i=1 to n
           = prod Fi(y) from i=1 to n
    caso particular X~U(0,1)
    F_X(y) = y^n


    """
    pass

def ej6a(n=3):
    """
    F(x) = x^n    0<=x<=1
    F(x) = prod Fi(x) from i=1 to n
    a) Generar n Unif y retornar maximo
    """
    return lib.raizn(n)

def ej6b(n=3):
    """
    F(x) = x^n    0<=x<=1
    F(x) = prod Fi(x) from i=1 to n
    b) Tinv F^-1(U) = U**(1/n)
    """
    U = random()
    return U**(1/n)

def ej6c(n=3):
    """
    F(x) = x^n    0<=x<=1
    f(x) = n*x**(n-1)
    g(x) = 1 <=> X ~ U(0,1)
    c = n
    """
    return lib.ayr(lambda: random(), lambda x: n*x**(n-1), lambda x: 1, n)

def ej7a():
    """
    f(x) = xe^-x,  x>=0
    """
    return lib.gamma(n=2, lamb=1)

def ej7b():
    """
    f(x) = xe^-x,  x>=0
    Y ~ g(x) ~ exponencial(1/2)
    c = 1.5
    """
    return lib.ayr(lambda: lib.exponencial(1/2), lambda x: x*e**-x, 
        lambda x: 1/2*e**-(x/2), 1.5)

def ej12():
    lambf = lambda t: 3 + 4/(t + 1)
    lambc = 7
    T = 10000000
    lenlimit = 10
    return lib.noHomo(T, lambf, lambc, lenlimit=lenlimit)
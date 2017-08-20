# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
# from sympy import solve, symbols, simplify, Eq, Integral, Piecewise, pprint, oo
# x, y = symbols("x y")
# r = Integral(Piecewise((0, x<2),((x-2)/2, x<=3),((2-x/3)/2, x<=6)), (x, -oo, x)).doit()
# solve(Eq(r.args[2][0], y), x)
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, e , exp, log, sqrt, pi
from random import random
from sympy import solve, symbols, simplify, Eq, Integral, Piecewise, pi, E, stats
from sympy import pprint, oo, init_printing, mpmath
x, y = symbols("x y")

import g4, g5, g6, lib

def g6ej1(d=.1):
    compare_disc(lambda x: 0, 
        lambda: g6.ej1(d=d)[0], (), 10000, 50, 150)


def ej4PDF(x):
    if 0 <= x <= 1:
        return x
    elif 1 < x <= 2:
        return 2 - x
    else:
        return 0

import alonso_lucas
def parej4():
    compare_cont(ej4PDF, alonso_lucas.ej4, (), 100000, 0, 2, 50)


def gammaPDF(x, a=3, b=2):
    if x < 0:
        return 0
    return b**a/mpmath.gamma(a)*x**(a-1)*exp(-b*x)

def g4ej5():
    compare_disc(lambda x: (1/2)**(x+1)+((1/2)*2**(x-1))/3**x, g4.ej5, (), 50000, 1, 20)

def g5ej6(n=3):
    compare_cont(lambda x: n*x**(n-1), g5.ej6a, (n,), 1000000, 0, 1, 20)
    compare_cont(lambda x: n*x**(n-1), g5.ej6b, (n,), 1000000, 0, 1, 20)
    compare_cont(lambda x: n*x**(n-1), g5.ej6c, (n,), 1000000, 0, 1, 20)

def g5ej7():
    compare_cont(lambda x: x*e**-x, g5.ej7a, (), 1000000, 0, 4, 20)
    compare_cont(lambda x: x*e**-x, g5.ej7b, (), 1000000, 0, 4, 20)


def gamma():
    compare_cont(gammaPDF, lib.gamma, (), 1000000, 0, 1, 20)


def g4ej4(l=4, k=22, N=50000, a=0):
    sumaabajo = sum(l**j/factorial(j) for j in range(k))*exp(-l)
    g = lib.poissonseriesgen(l)
    vals = [next(g) for _ in range(a, k+1)]
    compare_disc(lambda x: vals[x]/sumaabajo,
        g4.ej4, (l, k), N, a, k)

def g5ej1():
    def fd(x):
        if 2<=x<=3:
            return (x-2)/2
        elif 3<=x<=6: 
            return (2-x/3)/2
        else:
            return 0
    compare_cont(fd, g5.ej1, (), 1000000, 1, 7, 20)

def g2ej1(N=100000): #bien
    n = 0
    for _ in range(N):
        n += lib.N(4, .3) == 0
    return n/N

def g2ej5(N=100000, s=2): #bien
    return sum(lib.N(s/60, 3) <= 1 for _ in range(N))/N

def g3ej6a(N=100000): #bien --> 
    l_t = lambda t: 3 + 4/(t + 1)
    return sum(lib.NnoHomo(1, l_t, 7) == 5 for _ in range(N))/N

def g3ej6b(N=10000): #bien --> 0.166714
    l_t = lambda t: 3 + 4/(t + 1)
    n = 0
    cumplecond = 0
    while cumplecond < N:
        S = lib.noHomo(2, l_t, 7)
        if len(S) == 8:
            cumplecond += 1
            n += len(list(filter(lambda t: t >= 1, S))) == 5
    return n/cumplecond

def g5ej2(a=2, b=5):
    compare_cont(lambda x: a*b*x**(b-1)*exp(-a*x**b),
        g5.ej2, (a,b), 1000000, 0, 3, 20)
    #compare_cont(lambda x: exp(-x), np.random.exponential, (), 1000000, 0, 5, 20)

def normal(mu=2, sigma=3):
    fd = lambda x: 1/(sigma*sqrt(2*pi))*exp(-(x-mu)**2/(2*sigma**2))
    compare_cont(lambda x: fd(x), lib.normal, (mu, sigma), 1000000, -6, 12, 40)
    #compare_cont(lambda x: fd(x), lib.normal_ayr, (mu, sigma), 1000000, -6, 12, 40)
    #Z = stats.Normal('Z', 0, 1)
    #compare_cont(lambda x: fd(x), stats.sample, (Z,), 5000, -1, 1, 25)

def compare_cont(fd, va, params, N, a, b, res, show=True):
    s = list([0]*res)
    binsize = (b-a)/res
    #g = iter([1.1, 1.6, 1.9, 2.3, 2.4, 2.1, 2.61, 2.9])
    #plt.hist([va(*params) for _ in range(N)])
    #plt.show()
    #va = lambda x: next(g)
    for _ in range(N):
        v = va(*params)
        #v = next(g)
        if a <= v < b:
            s[int((v-a)/binsize)] += 1
    x = np.linspace(a + binsize/2, b - binsize/2, num=res)
    y = np.array(s)/N/binsize
    #plt.scatter(x, y)
    #plt.grid(True)
    #plt.plot((100, 100), (0, 1))
    fig, ax = plt.subplots()
    ax.bar(x, y, width=binsize, facecolor='green', alpha=0.75)

    def graph(a, b):
        x = np.arange(a, b+.1, 0.1)
        y = [fd(x) for x in list(x)]
        plt.plot(x, y)
        plt.ylabel('form ej4')

    graph(a, b)
    if show:
        plt.show()
    else:
        print("plt.show()")

def test_N():
    g = lib.poissonseriesgen(1)                            
    vals = [next(g) for _ in range(0, 20)]
    #compare_disc(lambda x: vals[x], lib.N, (1,1), 1000000, 0, 7)
    compare_disc(lambda x: vals[x], lib.NnoHomo, (1, lambda x: 1, 1.5), 1000000, 0, 7)

def ____test_noHomo(): # MAL
    c = 4*log(8) + 21 # = Integral(3+4/(x+1), (x, 0, 7)).doit()
    # Pero porque necesito dividir por esta constante ????
    l_t = lambda t: (3 + 4/(t + 1))/c
    def ppgen():
        for _ in range(100000000):
            for xi in lib.noHomo(7, l_t, 7):
                yield xi
    pp = ppgen()
    va = lambda: next(pp)
    compare_cont(lambda x: l_t(x), va, (), 100000, 0, 7, 20)

def test_noHomo():
    l_t = lambda t: 3 + 4/(t + 1)
    def ppgen():
        for xi in lib.noHomo(7, l_t, 7):
            yield xi
    compare_process(l_t, ppgen, 100000, 7, 20)

def test_noHomo2():
    l_t = lambda t: t
    def ppgen():
        for xi in lib.noHomo(8, l_t, 8):
            yield xi
    compare_process(l_t, ppgen, 100000, 8, 20)

def test_sin():
    from math import sin
    l_t = lambda t: sin(t) + 2
    def ppgen():
        for xi in lib.noHomo(7, l_t, 3):
            yield xi
    compare_process(l_t, ppgen, 100000, 7, 20)

def compare_process(fd, procgen, N, b, res):
    a = 0
    s = list([0]*res)
    binsize = (b-a)/res
    for _ in range(N):
        for v in procgen():
            if a <= v < b:
                s[int((v-a)/binsize)] += 1
    x = np.linspace(a + binsize/2, b - binsize/2, num=res)
    y = np.array(s)/N/binsize
    #plt.scatter(x, y)
    #plt.grid(True)
    fig, ax = plt.subplots()
    ax.bar(x, y, width=binsize, facecolor='green', alpha=0.75)

    def graph(a, b):
        x = np.arange(a, b+.1, 0.1)
        y = [fd(x) for x in list(x)]
        plt.plot(x, y)
        plt.ylabel('form ej4')

    graph(a, b)
    plt.show()


def compare_disc(fd, va, params, N, a, b, show=True):
    s = {s:0 for s in range(a, b+1)}
    for _ in range(N):
        v = va(*params)
        if a <= v < b+1:
            s[v] += 1
    (keys,values) = zip(*s.items())
    plt.scatter(keys, np.array(values)/N, color='red')
    #plt.ylabel('ej5')
    #plt.figure()

    def graph(x_range):  
        x = np.array(x_range)  
        y = [fd(x) for x in list(x)]
        plt.plot(x, y)  
        plt.ylabel('form ej4')

    graph(range(a, b+1))
    plt.show()


#ax.set_xticks(survived_df.index+0.4)  # set the x ticks to be at the middle of each bar since the width of each bar is 0.8
#ax.set_xticklabels(survived_df.Groups)  #replace the name of the x ticks with your Groups name
#plt.show()

def p3ej1():
    from scipy.stats import binom
    M = [1,1,0,0,4,0,1,3,0,1,2,1,1,0,1,1,0,2,1,1]
    p = lib.media(M)/4
    M = binom.rvs(n=4, p=p, size=len(M)).tolist()
    N = [M.count(i) for i in range(5)]
    #compare_disc(lambda x: binom.pmf(x, n=4, p=p), lambda: next(g), (), 5, 0, 4)
    print(N)
    plt.plot(range(5), [n/len(M) for n in N])
    plt.plot(range(5), [binom.pmf(x, n=4, p=p) for x in range(5)])
    plt.show()
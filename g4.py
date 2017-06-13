import lib, libnat
from random import random
from lib import mcarlo
from math import e, factorial, sqrt, pi, exp
from itertools import chain
from functools import partial

def ej1_play():
		perm = lib.permutacion(list(range(1, 101)))
		return sum(1 for (i, c) in enumerate(perm) if i + 1 == c)

def ej1(n):
	"""
	https://www.johndcook.com/blog/2010/04/06/subfactorial/
	n: numero de muestras
	"""
	sample = [ej1_play() for _ in range(n)]
	print("n: {0}".format(n))
	print("sum: {0}".format(sum(sample)))
	print("media: {0}".format(lib.media(sample)))
	print("varianza: {0}".format(lib.varianza(sample)))


def ej2(n):
    """sum exp(k/N) from k=1 to N con N=10**4"""
	#return e**(1/10**4)*10**4*mcarlo(lambda k: e**k, n)
    N = 10**4
    return N*sum(exp(lib.udiscreta(1, N)/N) for _ in range(n))/n

def roll_dice():
	return lib.udiscreta(1, 6)

def ej3(n):
	def rolluntil():
		count = 0
		all = set(range(2,13))
		while all != set():
			all.discard(roll_dice() + roll_dice())
			count += 1
		return count

	sample = [rolluntil() for _ in range(n)]
	print("n: {0}".format(n))
	print("sum: {0}".format(sum(sample)))
	print("media: {0}".format(lib.media(sample)))
	print("varianza: {0}".format(lib.varianza(sample)))

def ej4(l, k, U=None):
    """
    
    """
    U = random() if U is None else U
    g = lib.poissonseriesgen(l)
    next(g)
    sumaabajo = sum(l**j/factorial(j) for j in range(k))*exp(-l)
    return lib.transf_inv((x for x in range(10**10)),
        chain(iter([exp(-l)/sumaabajo]), g), U=U)

def ej5():
    p1gen = (2**-j for j in range(1, 10**10))
    p2gen = (1/2 * (2/3)**j for j in range(1, 10**10))
    Ngen1 = (n for n in range(1, 10**10))
    Ngen2 = (n for n in range(1, 10**10))
    return lib.met_comp(
        lib.transf_inv(Ngen1, p1gen),
        lib.transf_inv(Ngen2, p2gen),
        1/2)

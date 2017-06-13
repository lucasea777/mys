from random import random
from math import e, pi, sin

monte = lambda g, n: sum(g(random()) for _ in range(n))/n

def ej4b(n):
	g = lambda x: x**2*(e**(-x**2))
	h = lambda s: g(1/s - 1)/s**2
	return 2 * monte(h, n)

def ej4a(n):
	g = lambda x : sin(x) + 1/3 * sin(3*x)
	h = lambda s : g(pi*s)*pi
	return monte(h, n)


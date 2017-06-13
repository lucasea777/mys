import numpy as np
from itertools import permutations
import random
from statistics import stdev, variance
from sympy import Sum, E, symbols
k = symbols('k')

def discretaX(x, p, U=None):
	return np.random.choice(elements, p=p)

def udiscreta(a, b):
	return random.randint(a, b)

def permutacion(x):
	return list(np.random.permutation(x))

def ds(iter):
	return stdev(iter)

def varianza(iter):
	return variance(iter)

def media(list):
	return sum(list)/len(list)

def ej2():
	return Sum(E**(k/10000), (k, 1, 10000)).doit().evalf()


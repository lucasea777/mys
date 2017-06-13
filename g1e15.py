import random
import numpy
def a():
    return sum(numpy.random.choice([-1, 69], 1000000, p=[.99, .01]))/1000000

def b():
    return sum(numpy.random.choice([-1, 69], 60, p=[.99, .01])) < -15

def getprob(n, playpartial):
    c = sum(playpartial() for _ in range(n))
    return c/n

# count(100000, b)
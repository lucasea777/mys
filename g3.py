from random import random
from math import e, pi, sqrt
E = lambda f,n: sum(f(random()) for _ in range(n))/n # Expected value
ej3a = lambda n: E(lambda x: (1-x**2)**(3/2), n)
mcarloab = lambda a,b,f,n: E(f(a + (b - a)*y)*(b-a), n)
mcarlo0Inf = lambda f,n: E(lambda x: f(1/x - 1)/x**2, n)
ej3b = lambda n : mcarlo0Inf(lambda x: x*(1+x**2)**(-2), n)

ej3c = lambda n : 2 * mcarlo0Inf(lambda x: e**(-x**2), n)
# http://www.wolframalpha.com/input/?i=integrate+e%5E(-x%C2%B2)
E2 = lambda f,n: sum(f(random(), random()) for _ in range(n))/n
mcarlomv0Inf = lambda f,n: sum(f(random(), random()) for _ in range(n))/n
ej3d = lambda n : E2(lambda x,y: e**((x+y)**2), n)
# http://www.wolframalpha.com/input/?i=Integrate%5Be%5E(x%2By)%C2%B2,+%7By,+0,+1%7D,+%7Bx,+0,+1%7D%5D
I = lambda x, y: y < x
w = lambda x,y : I(x,y)*e**-(x+y)
ej3e = lambda n : E2(lambda x,y: w(1/x + 1, 1/y + 1)/(x**2*y**2), n)
# http://www.wolframalpha.com/input/?i=Integrate%5Be%5E-(x%2By),%7Bx,+0,+inf%7D,++%7By,+0,+x%7D%5D
def showtable():
    print('n' + ' '*5, *map(lambda i: chr(i) + ' '*10, range(97, 97+4)), sep='\t')
    ejs = [ej3a, ej3b, ej3c, ej3d]
    reales = [3/16*pi, 1/2, sqrt(pi), 4.899158851087022]
    for i in range(2,7):
        print(10**i, *["%.6f" % ej(10**i) for ej in ejs], sep='\t', end='\n')
    print("real", *["%.6f" % r for r in reales], sep='\t', end='\n')
# showtable()
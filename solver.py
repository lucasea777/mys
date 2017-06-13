from sympy import solve, symbols, simplify, Eq, Integral, Piecewise, E, stats
from sympy import pprint, oo, init_printing, Derivative, pi as PI
from math import factorial, e , exp, log, sqrt, pi
from random import random
init_printing(pretty_print=True)
x, y, a, b = symbols("x y a b")
r = Integral(Piecewise((0, x<2),((x-2)/2, x<=3),((2-x/3)/2, x<=6)), (x, -oo, x)).doit()
# log(a*b) = log(a) + log(b)
# (f o g)' = (f' o g)*g'
# solve(Eq(r.args[2][0], y), x)
# subs
"""
         ___________
        ╱  2  
-b +- ╲╱  b  - 4⋅a⋅c
────────────────────
         2⋅a         
"""
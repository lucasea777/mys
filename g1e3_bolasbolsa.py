import random
from functools import partial
def play(f, s):
    b = {'az1', 'az2', 'az3', 'az4', 'az5', 'az6', 'az7', 'az8', 'az9',
        'am1', 'am2', 'am3', 'am4', 'am5', 'am6', 'am7'}
    typee = lambda x: x[:2]
    fs = random.choice(tuple(b))
    b.remove(fs)
    if typee(fs) == f:
        ss = random.choice(tuple(b))
        b.remove(ss)
        if typee(ss) == s:
            return True
    return False

def count(n, playpartial):
    c = sum(playpartial() for _ in range(n))
    return c/n

#count(1000000, functools.partial(play, 'az', 'am'))
# aprox 0.2625
# count(1000000, functools.partial(play, 'az', 'az'))
# aprox 0.30
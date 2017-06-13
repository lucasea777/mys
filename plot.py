# http://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html
# http://bugra.github.io/work/notes/2014-06-26/law-of-large-numbers-central-limit-theorem/
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import g5
from random import random
def fd(x):
    if 2<=x<=3:
        return (x-2)/2
    elif 3<=x<=6: 
        return (2-x/3)/2
    else:
        return 0
#mu, sigma = 100, 15
#x = mu + sigma*np.random.randn(10000)
x = [g5.ej1() for _ in range(50)]

# the histogram of the data
n, bins, patches = plt.hist(x, 5, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
y = [fd(x) for x in bins]
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()
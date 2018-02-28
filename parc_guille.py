from random import random
from math import e, sqrt, factorial, pi, cos,sin, log
from numpy import std, median
from numpy.lib.scimath import logn
from scipy.stats import chi2, norm,  binom

def suma(array,i):
    suma = 0
    for j in range(0,i + 1):
        suma = suma + array[j]

    return suma


def media(array):
    res = 0
    for a in array:
        res = res + a
    return res/len(array)

def s_cuadrado(array):
    m = media(array)
    v = 0
    for a in array:
        v = v + (a - m)**2
    return v/(len(array)-1)

#Ejercicio1
####################################################################


def generador(array,n):
    res = []
    for i in range(0,n+1):
        res.append(0)
        for a in array:
            if (a == i):
                res[i] = res[i] + 1 
    return res

def binomial(n,p):
    U=random()
    i=0
    if( p == 1):
        return n

    c = p/(1-p)
    prob = (1-p)** n
    F=prob
    
    while U >= F:
        prob= c* ((n-i)/(i+1)) * prob
        F= F+prob
        i= i+1

    return i


def gen_ns(dist,tam):
    k = len(dist)
    array = []
    n = tam
    p = dist[0]
    array.append(binomial(n, p))
    n = n - array[0]

    for j in range(1,k):
        p = dist[j]/ (1 - suma(dist,j-1))
        array.append(binomial(n,p))
        n = n - array[j]

    return array

def calc_t(arreglo_Ns, dist,tam_muestra):

    i = 0 
    t = 0
    M = len(arreglo_Ns)

    if (sum(arreglo_Ns) != tam_muestra):
        return -1

    while i < M :
        t = t + ((arreglo_Ns[i] - tam_muestra*dist[i])**2/(tam_muestra*dist[i]))
        i = i + 1
     
    return t

def ej1(ns=10000, muestra=[1,1,0,0,4,0,1,3,0,1,2,1,1,0,1,1,0,2,1,1]):

    N = generador(muestra,4)
    p_est =  media(muestra)/4  
    tam_muestra = len(muestra)

    i = 0
    count = 0
    dist= []
    
    for k in range(0,5):
        dist.append(binom.pmf(k, n=4, p=p_est))
     
    t_obs = calc_t(N,dist, tam_muestra)

    while (i < ns):
        array = gen_ns(dist,tam_muestra)
        t_sim = calc_t(array,dist,tam_muestra)

        if( t_sim > t_obs):
            count = count+1
        i = i + 1


    return count/ns

#Ejercicio2
#####################################################################

def media_rec(media,last_elem, n):
      return media + (last_elem - media)/n

def var_rec(media, var, last_elem, n):
    return (media_rec(media,last_elem,n), (1-(1/(n-1))) * var + n * ( media_rec(media,last_elem,n) - media)**2 )


def normal(mu,sigma):
    while True:
        Y1=-logn(e,random())
        Y2=-logn(e,random())
        if Y2 >= (Y1-1)**2/2:
            break
    if random() < 0.5:
        return Y1*sigma + mu
    else:
        return -Y1*sigma + mu


def ej2_b():
    count = 0
    array = []
    
    # primeras dos iteraciones
    for _  in range(0,2):
        muestra = []

        for i in range(0,10):
             muestra.append(normal(0,1))

        chi_cuadrado = (len(muestra)- 1)* s_cuadrado(muestra)
        array.append(chi_cuadrado)

        if( 0 < chi_cuadrado and chi_cuadrado < 5):
            count = count + 1
    
    med_var =[media(array),s_cuadrado(array)]
    n = 2

   
    while( n <= 30 or sqrt(med_var[1]/n) > 0.1):
        muestra = []

        for i in range(0,10):
            muestra.append(normal(0,1))

        chi_cuadrado = (len(muestra)- 1) * s_cuadrado(muestra)
        n = n+1
        
        med_var = var_rec(med_var[0], med_var[1], chi_cuadrado, n)

        if( 0 < chi_cuadrado and chi_cuadrado < 5):
            count = count + 1

        
    return count/n


def ej2_check():
    return chi2.cdf(5,9) -chi2.cdf(0, 9)



### ejercicio 2b
def dist_emp(x):
    muestra = [6.442,7.968,2.287,5.679,5.740,7.254,3.126,3.443,7.702]
    muestra.sort()
    
    i = 0
    prob = 0

    while x >= muestra[i] and i < 9:
        prob = (i+1)/9
        i = i + 1

def med_emp(array):
    res = 0
    for i in range(0,len(array)):
        res = res + array[i] * 1/9
    return res

def var_emp(array):
    res = 0
    media = med_emp(array)
    for i in range(0,len(array)):
        res = (array[i] - media)**2
    
    return res/len(array)



#Ejercicio3
#####################################################################

def acum_exp(x,lamb):
    res = 0
    if x > 0:
       res = 1-e**(-lamb*x)
    return res


def exponencial(lamb):
    U = random()
    return -log(U)/lamb

def c_D1(muestra, f_acum, par1):
    muestra.sort()
    n = len(muestra)
    max_dif = 1/n - f_acum(muestra[0], par1)

    for j in range(1,len(muestra)):
         if(((j+1)/n - f_acum(muestra[j],par1)) > max_dif):
            max_dif = (j+1)/n - f_acum(muestra[j],par1)

    for j in range(0,len(muestra)):
        if((f_acum(muestra[j],par1) - j/n)  > max_dif):
            max_dif = f_acum(muestra[j],par1) - j/n

    return max_dif

def sim1_exp(Nsim, m, d,f_acum, f_gen, par1):
    pvalor = 0

    for _ in range(Nsim):

        muestra_sim = []

        for j in range(m):
            muestra_sim.append(f_gen(par1))
           
        par_est= 1/media(muestra_sim)

        muestra_sim.sort()

        lista = []
        for j in range(m):
            lista.append((j+1)/m - f_acum(muestra_sim[j],par_est))
            lista.append(f_acum(muestra_sim[j],par_est) - j/m)

        if max(lista) > d:
            pvalor = pvalor + 1 
            

    return pvalor/Nsim



######################################################################



def ej3():

    muestra = [0.06,0.02,0.18,0.17,0.08,0.13,0.22,0.07,0.12,0.21,0.03]

    par1 = 1/media(muestra)

    d = c_D1(muestra, acum_exp, par1)
    

    return d, sim1_exp(10000, len(muestra), d ,acum_exp, exponencial, par1)


#Ejercicio 4
######################################################################
def rangos(n,m,r):

    if n ==1 and m==0:
        if r <=0 :
            return 0
        else:
            return 1
    elif n==0 and  m==1:
        if r<0:
            return 0
        else:
            return 1
    else:   
        if n==0:
            return rangos(0,m-1,r)
        elif m==0:
            return rangos(n-1,0,r-n)
        else:
            return n/(n+m)*rangos(n-1,m,r-n-m)+m/(n+m)*rangos(n,m-1,r)


def calculo_r(muestra1,muestra2):
    n = len(muestra1)
    r=0    
    lista = muestra1 + muestra2

    lista.sort()

    
    for i in range(0,len(muestra1)):
       r = r + (lista.index(muestra1[i]) + 1)
    
    return r

def ej4_a():

    muestra1= [141,132,154,142,143,150,134,140]
    muestra2= [133,138,136,125,135,130,127,131,116,128]

    

    n = len(muestra1)
    m = len(muestra2)
    r = calculo_r(muestra1,muestra2)
    ps =[rangos(n,m,r),1 - rangos(n,m,r-1)]

    return 2 * min(ps)


def ej4_b():

    muestra1= [141,132,154,142,143,150,134,140]
    muestra2= [133,138,136,125,135,130,127,131,116,128]

    n =len(muestra1)
    m = len(muestra2)
    r = calculo_r(muestra1,muestra2)


    r_estrella = (r - n*(n+m+1)/2)/(sqrt(n*m*(n+m+1)/12))


    if (r_estrella <= 0 ):

        pvalor = 2* norm.cdf(r_estrella)
    else:
        pvalor =  2* (1 - norm.cdf(r_estrella))

    return pvalor


from g7 import pval_disc_sim_binom, test_pval, pval_disc
import lib
def test_ej1a(ns=10000):
    p = [1/4, 1/2, 1/4] 
    def gen():
        N = [0]*3
        for i in range(564):
            i = lib.discretaX([0,1,2], p)
            N[i] += 1
        return N
    def pvalcreator():
        N = gen()
        t, pval = pval_disc(p, N, param_desc=0)
        #t, pval = pval_disc_sim_binom(p, N, ns=1000)
        return pval
    test_pval(pvalcreator, 75, ns=ns)

def test_ej1(ns=1000):
    p = .2625
    def pvalcreator():
        pval = ej1(ns=100, muestra=[binomial(4, p) for _ in range(20)])
        #t, pval = pval_disc_sim_binom(p, N, ns=1000)
        return pval
    test_pval(pvalcreator, 75, ns=ns)
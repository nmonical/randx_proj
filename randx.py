# import packages
import numpy as np
import math
from scipy.stats import norm
from scipy import stats

""" 
desert_island generates Unif(0,1) random numbers which are then used in the generation of random variates from other distributions. 
    n = number of uniform random numbers required for the generation of one random number from the goal distribution
    size = number of random numbers to be be generated from the goal distribution
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M6L3)
"""
def desert_island(n,size,seed):
    # initialize X_i with seed value
    X_i_minus_1=seed    
    # initialize list which will be used to store Unif(0,1) RVs
    output=[]   
    # run iterations of "desert island" algorithm to produce necessary number of Unif(0,1) RVs
    for i in range(size):   
        sample=[]
        for j in range(n):
            X_i = 16807*X_i_minus_1%(2**31-1)
            sample.append(X_i/(2**31-1))
            X_i_minus_1=X_i
        output.append(sample)
    # return list of Unif(0,1) RVs
    return output

"""
rand_unif uses the Inverse Transform Method to generate uniform random numbers between a and b. 
    a = minimum value
    b = maximum value
    size = number of random numbers to be be generated
    seed = starting seed
"""
def rand_unif(a,b,size, seed):
    # define error messages
    if not isinstance(a,(float, int)):
        raise ValueError('a must be a number')
    if not isinstance(b,(float, int)):
        raise ValueError('b must be a number')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Unif(a,b) RVs
    unif_output=[]
    # run iterations of inverse transform algorithm to generate Unif(a,b) RVs
    for i in rand_unifs:
        unif_output.append(a+(b-a)*i[0])
    # return list of Unif(a,b) RVs
    return unif_output

"""
rand_triangular uses the Inverse Transform Method to generate random numbers from a triangular distribution between a and b. 
    a = minimum value
    b = maximum value
    c = mode value
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from math.wm.edu
"""

def rand_tria(a,c,b,size, seed):
    # define error messages
    if not isinstance(a, (float, int)):
        raise TypeError('a must be numeric')
    if not isinstance(b, (float, int)):
        raise TypeError('b must be numeric')
    if not isinstance(c, (float, int)):
        raise TypeError('c must be numeric')    
    if c > b or c < a:
        raise ValueError('c must be between a and b')
    if a > b:
        raise ValueError('a must be less than b')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Tria(a,c,b) RVs
    tri_output=[]
    # run iterations of inverse transform algorithm to generate Tria(a,c,b) RVs
    for i in rand_unifs:
        if i[0] < (c-a)/(b-a):
            tri_output.append(a+((b-a)*(c-a)*i[0])**(1/2))
        else:
            tri_output.append(b-((b-a)*(b-c)*(1-i[0]))**(1/2))
    # return list of Tria(a,c,b) RVs
    return tri_output

"""
rand_exp uses the Inverse Transform Method to generate random numbers from an exponential distribution with parameter lambda(l). 
    l = lambda (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L2)
"""

def rand_exp(l, size, seed):
    # define error messages
    if not isinstance(l, (float, int)):
        raise TypeError('l must be numeric')
    if l < 0:
        raise ValueError('l must be greater than or equal to zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Exp(l) RVs
    exp_output=[]
    # run iterations of inverse transform algorithm to produce necessary number of Exp(l) RVs
    for i in rand_unifs:
        exp_output.append((-1/l)*np.log(1-i[0]))
    # return list of Exp(l) RVs
    return exp_output

"""
rand_weibull uses Inverse Transform Method to generate random numbers from a Weibull distribution with parameters lambda(l), and beta(b). 
    l = lambda (rate)
    b = beta (shape)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L2)
"""
def rand_weib(l, b, size, seed):
    # define error messages
    if not isinstance(l,(float,int)):
        raise TypeError('l must be numeric')
    if l <= 0:
        raise ValueError('l must be greater than zero')
    if not isinstance(b,(float,int)):
        raise TypeError('b must be numeric')
    if b <= 0:
        raise ValueError('b must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Weib(l,b) RVs
    weib_output=[]
    # run iterations of inverse transform algorithm to produce necessary number of Weib(l,b) RVs
    for i in rand_unifs:
        weib_output.append(l*(-math.log(1-i[0]))**(1/b))
    # return list of Weib(l,b) RVs
    return weib_output

"""
rand_erlang uses the Inverse Transform Method and Convolution to generate random numbers from an Erlang distribution with parameters lambda(l), and number(n). 
    n: number (shape)
    l: lambda (rate)
    size: number of random numbers to be be generated
    seed: integer used to initialize random number generator
Algorithm from Goldsman Lecture Notes (M7L6)
"""
def rand_erlang(n, l, size, seed):
    # define error messages
    if not isinstance(n,int):
        raise TypeError('n must be an integer')
    if n <= 0:
        raise ValueError('n must be greater than zero')
    if not isinstance(l,(float,int)):
        raise TypeError('l must be numeric')
    if l <= 0:
        raise ValueError('l must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')   
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(n,size,seed)
    # initialize list which will be used to store Erlang(n,l) RVs
    erlang_output=[]
    # run iterations to first get product of n Unif(0,1) RVs, and then apply inverse transform algorithm to produce necessary number of Erlang(n,l) RVs
    for i in rand_unifs:
        prod=1
        for j in i:
            prod*=j
        erlang_output.append((-1/l)*math.log(prod))
    # return list of Erlang(n,l) RVs
    return erlang_output
"""
rand_gamma uses the Inverse Transform and Acceptance-Rejection Methods to generate random numbers from a gamma distribution with parameters alpha(a) and beta(b). 
    a = alpha (shape)
    b = beta (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Simulation and Modeling Analysis, Law, Averill M., page 454-456
"""
def rand_gamma(a, b, size, seed):
    # define error messages
    if not isinstance(a,(float,int)):
        raise TypeError('a must be numeric')
    if a <= 0:
        raise ValueError('a must be greater than zero')
    if not isinstance(b,(float,int)):
        raise TypeError('b must be numeric')
    if b <= 0:
        raise ValueError('b must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')   
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(2,size,seed)
    # initialize list which will be used to store Gamma(a,b) RVs
    gamma_output=[]
    # run iterations of inverse transform algorithms and then applies acceptance-rejection method to produce necessary number of Gamma(a,b) RVs
    c=(math.e+a)/math.e
    if a==1:
        for i in rand_unifs:
            gamma_output.append(b*(-1/1)*np.log(1-i[0]))
    if a < 1:
        for i in rand_unifs:
            if c*i[0] > 1:
                Y=-(math.log((c-c*i[0])/a))
                if i[1] <= Y**(a-1):
                    gamma_output.append(Y)
            else:
                Y=(c*i[0])**(1/a)
                if i[1] <= math.exp(-Y):
                    gamma_output.append(b*Y)
    else:
        v=1/(2*a-1)**(1/2)
        w=a-math.log(4)
        q=a+1/v
        x=(a+1)/1
        y=4.5
        z=1+math.log(y)
        for i in rand_unifs:
            V=v*math.log(i[0]/(1-i[0]))
            Y=a*math.exp(V)
            Z=i[0]**2*i[1]
            W=w+q*V-Y
            if W+z-y*Z>=0:
                gamma_output.append(b*Y)
            elif W>=math.log(Z):
                gamma_output.append(b*Y)
        # return list of Gamma(a,b) RVs
        return gamma_output
    
"""
rand_nor uses the Inverse Transform Method to generate random numbers from a normal distribution with parameters mean (m), and variance (v). 
    m = mean 
    v = variance
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L3)
"""
def rand_nor(m, v, size, seed):
    # define error messages
    if not isinstance(m,(float,int)):
        raise TypeError('m must be numeric')
    if not isinstance(v,(float,int)):
        raise TypeError('v must be numeric')
    if v < 0:
        raise ValueError('v must be greater than or equal to zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Nor(m,v) RVs
    nor_output=[]
    # run iterations of inverse transform algorithm to produce necessary number of Nor(m,v) RVs
    for i in rand_unifs:
        nor_output.append(norm.ppf(i[0], m, v))
    # return list of Nor(m,v) RVs
    return nor_output

"""
rand_bern uses the Inverse Transform Method to generate random numbers from a Bernoulli distribution with parameter, probability of success (p). 
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L4)
"""
def rand_bern(p, size, seed):
    # define error messages
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Bern(p) RVs
    bern_output=[]
    # run iterations of inverse transform algorithm to produce necessary number of Bern(p) RVs
    for i in rand_unifs:
        if i[0] <= p:
            bern_output.append(1)
        else:
            bern_output.append(0)
    # return list of Bern(p) RVs
    return bern_output
    
"""
rand_bin uses the Inverse Transform and Convolution Methods to generate random numbers from a binomial distribution with parameters n and p. 
    n = number of trials
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L6)
"""
def rand_bin(n, p, size, seed):
    # define error messages
    if not isinstance(n,int):
        raise TypeError('n must be an integer')
    if n <= 0:
        raise ValueError('n must be greater than zero')
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(n,size,seed)
    # initialize list which will be used to store Bin(n,p) RVs
    bin_output=[]
    # run iterations of inverse transform algorithm and then sum them to produce necessary number of Bin(n,p) RVs
    for i in rand_unifs:
        total=0
        for j in i:
            if j <= p:
                total+=1
        bin_output.append(total)
    # return list of Bin(n,p) RVs
    return bin_output

"""
rand_geom uses the Inverse Transform Method to generate random numbers from a geometric distribution with parameter, probability of success (p). 
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L4)
"""
def rand_geom(p, size, seed):
    # define error messages
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(1,size,seed)
    # initialize list which will be used to store Geom(p) RVs
    geom_output=[]
    # run iterations of inverse transform algorithm to produce necessary number of Geom(p) RVs
    for i in rand_unifs:
        geom_output.append(math.ceil(math.log(i[0])/math.log(1-p)))
    # return list of Geom(p) RVs
    return geom_output

"""
rand_negbin uses Inverse Transform Theory and Convolution to generate random numbers from a negative binomial distribution with parameters n and p. 
    n = number of successes
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L6)
"""
def rand_negbin(n, p, size, seed):
    # define error messages
    if not isinstance(n,int):
        raise TypeError('n must be an integer')
    if n <= 0:
        raise ValueError('n must be greater than zero')
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(n,size,seed)
    # initialize list which will be used to store NegBin(n,p) RVs
    negbin_output=[]
    # run iterations of inverse transform algorithm and sum them to produce necessary number of NegBin(n,p) RVs
    for i in rand_unifs:
        count=0
        for j in i:
            count+=math.ceil(math.log(j)/math.log(1-p))
        negbin_output.append(count)
    # return list of NegBin(n,p) RVs
    return negbin_output

"""
rand_poisson uses Acceptance-Rejection Method to generate random numbers from a poisson distribution with parameter lambda(l). 
    l = lambda (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L10)
"""
def rand_pois(l, size, seed):
    # define error messages
    if not isinstance(l,(float,int)):
        raise TypeError('l must be numeric')
    if l <= 0:
        raise ValueError('l must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    # generate Unif(0,1) RVs
    rand_unifs=desert_island(l+10,size,seed)
    # initialize list which will be used to store Pois(l) RVs
    pois_output=[]
    # run iterations of acceptance-rejection method to produce necessary number of Pois(l) RVs
    for i in rand_unifs:
        p=1
        X=-1
        a=math.exp(-l)
        for j in i:
            if p >= a:
                p=p*j
                X=X+1
            else:
                break
        pois_output.append(X)
    # return list of Pois(l)) RVs
    return pois_output





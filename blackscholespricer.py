from scipy.stats import norm
from scipy import *
import pandas as pd
from scipy.optimize import newton
from scipy.optimize import root


def d_j(j, S, K, r, v, T):
    num = log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*T
    denom = (v*(T**0.5))
    return num/denom

def vanilla_price(S, K, r, v, T,type):
    if (type == 'C'):
        price = S*norm.cdf(d_j(1, S, K, r, v, T))-K*exp(-r*T) * norm.cdf(d_j(2, S, K, r, v, T))
    if (type == 'P'):
        price = -S*norm.cdf(-d_j(1, S, K, r, v, T))+K*exp(-r*T) * norm.cdf(-d_j(2, S, K, r, v, T))
    return price

def imp_vol(observed_price, S,K,r,T,type): # S0,K,r,v,T,type are defined outside
    def func(v):
        return (observed_price)-vanilla_price(S,K,r,v,T,type)
    iv = newton(func, 1)
    return iv


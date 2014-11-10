import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from pylab import *


def callprice(S,K,r,T,lambd,rho,eta,v0,v_bar):

    alpha=0.75
    
    def Heston_cf(w):
        T1=(lambd-rho*eta*1j*w)**2
        T2 = (w**2+1j*w)*eta**2
        D = sqrt(T1+T2)
    
        num = lambd-rho*eta*1j*w-D
        denom = lambd-rho*eta*1j*w+D
        G = num/denom
    
        E1 = exp(1j*w*(log(S)+r*T))
        E2 = exp((v0)/(eta**2)*((1-exp(-D*T))/(1-G*exp(-D*T)))*(lambd-rho*eta*1j*w-D))
        E3 = exp((lambd*v_bar)/(eta**2)*(T*(lambd-rho*eta*1j*w-D)-2*log((1-G*exp(-D*T))/(1-G))))
        return E1*E2*E3


    def Heston_psi(v): # alpha to be defined outside
        num = exp(-r*T)*Heston_cf(v-(alpha+1)*1j)
        denom = (alpha**2)+(alpha)-(v**2)+(1j*(2*alpha+1)*v)
        return num/denom

    def CM_Integrand(v):
        return (exp(-1j*v*log(K))*Heston_psi(v)).real



    # by integration

    #   factor = exp(-alpha*log(K))/pi
    #   result, error = quad(CM_Integrand, 0, 100) #left-hand point can be 0

    #   return factor*result


    #by FFT

    N=2**11
    dk=0.025
    dv = 2*pi/(N*dk) # depends on N, dk
    b=0.5*(N-1)*dk # depends on N, dk

    ku = -b + dk*linspace(0,N-1,N) # log strikes from -b, -b+dk, -b+2dk, ...-, b+(N-1)dk=b

    vn = linspace(0,N-1,N)*dv # vn= (n-1)*dv

    a=exp(1j*b*vn)*Heston_psi(vn) # input array for fft
    fft_out = np.fft.fft(a)
    iterm=0.5*(CM_Integrand(0)+CM_Integrand((N-1)*dv)) # terms from trapezoidal rule

    prices = (1/pi)*(exp(-alpha*ku))*(fft_out.real-iterm)*dv

    call_price = interpolate.interp1d(ku, prices) # interpolation

    return call_price(log(K))





# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:51:45 2018

@author: hutsi
"""

import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
import sklearn.linear_model as sklm
import scipy
    
class Psd:
    
    def __init__(self, sigm, u, xmax, xmin, what:str=""):
        self.sigm = sigm
        self.u = u
        self.xmax = xmax
        self.xmin = xmin
        self.what = what
        self.x = None
        self.y = None

        
    def _f(self, x):
        t = (x - self.xmin) / (self.xmax - self.xmin)
        exp = np.exp( -0.5 * (self.u + self.sigm * np.log( t / (1 - t) ) ) ** 2 )
        ret = (exp * self.sigm) / ( np.sqrt(2*np.pi) * (self.xmax - self.xmin) * t * (t - 1) )
        return -ret
    
    def process(self, x: np.array):
        self.y = self._f(x) * 1e-9
        self.x = x
        
    def show(self):
        plt.plot(self.x, self.y)

    def __str__(self):
        return (self.what)        
    

class Simulator:
    
    def __init__(self, lamd, theta, n, t, eta, d:list, what:str=""):
        k = 1.3807e-23
        self.coef = 2*(k * t/(3 * np.pi * eta)) * (4 * np.pi * n / lamd)**2 * np.sin(theta/2)**2
        self.I = None
        self.f = None
        self.d = np.array(d)
    
    def simulate(self): 
        self.f = 2.0*np.arange(1,5001)
        G = self.coef / self.d
        for g in G:
            if self.I == None:
                self.I = g / ((2 * np.pi * self.f)**2 + g**2)
            else:
                self.I += g / ((2 * np.pi * self.f)**2 + g**2)
                
    def show(self):
        plt.plot(self.f, self.I)
    
    
class Wrim:
    def __init__(self, sim):
        self.dmin = 1e-9 
        self.dmax = 1e-6 
        self.y = sim.I
        
        Gmin = sim.coef/self.dmax
        Gmax = sim.coef/self.dmin
        N = 256*8 
        M = sim.f.size 
        p = (Gmax/Gmin)**(1/(N-1))
        G = np.zeros(N)
        
        for j in range(N):
            G[j] = Gmin * (p**j)
        
        self.A = np.zeros((M,N))
        print(1)
        for i in range(M):
            for j in range(N):
                self.A[i,j] = G[j] / ((2 * np.pi * sim.f[i])**2 + G[j]**2)
        
        
        
        
if __name__ == "__main__":
    eta = 0.89e-3
    t = 298
    theta = np.pi / 2
    n = 1.331
    lamd = 623.8e-9
    
    psd = Psd(sigm=2.0, u=0.4, xmin=300e-9, xmax=700e-9)    
    x = np.linspace(301e-9, 699e-9, 100)
    psd.process(x)
    sim = Simulator(lamd, theta, n, t, eta, [477.01e-9])
    sim.simulate()
    #sim.show()
    wrim = Wrim(sim)
    psd.show()
    
    
    
    
    
    
    
        
        
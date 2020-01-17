#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:51:18 2019

@author: georgef
"""

import numpy as np
#%%
## Simulating the escape rate problem for a fixed noise intensity for OU process
## Simulated originally using htcondor 
N_sample = 1000
T = 1000
D = 1
tau = 1
def dU(c,x):  ## The derivative of the cubic potential U =-x^3/3 - cx  
    return  -x**2 - c

def escape (tau, var, N_sample, T):
    x0 = -10         ## starting point
    xf = 10          ## end point
    counter = 0      ## the number of escapes during a time window
    counter_tot = [] ## the total number of escapes for all time windows N_sample
    dt = 10 ** -3    ## the time step
    N = int(T/dt)    ## the number of steps
    for i in range(0, N_sample):
        y0 = np.random.normal(0, np.sqrt(var)) #start point of noise is laplace dist.
        x = x0
        y = y0  #starting point is Normal dist. so the process is stationary
        for j in range(0, N):
            ## the SDE of OU process
            y = y - y / tau * dt +  (np.sqrt(2 * D * dt) / tau) * np.random.normal(0,1)
            force =  - dU(-1 , x)   ## the force acting on the system  
            x = x + dt * (force + y) ## the Langevin equation with colored noise 
            if x > xf :
                counter += 1
                x = x0   ## reset the position of the Langevin equation but not the external noise 
        counter_tot.append(counter)
        counter = 0 
    return  counter_tot
#%%
## The escape rate is the mean of the counter divided by the time window
## This way we would eliminate bias sampling 
## the error is the std of the counter array divided by the sqrt(N_sample -1)
## The simulation must be repeated for different values of D, tau
## The results to be found in Fig. 16 in the thesis 
counter =  escape(tau, D , N_sample, T)
r = np.mean(counter) / (T)
error = np.std(counter) / np.sqrt(N_sample - 1)
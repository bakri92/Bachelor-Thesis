#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:17:47 2019

@author: georgef
"""

import numpy as np 
import matplotlib.pyplot as plt
#%%
## simulating the OU process which has a Normal stationary dist.
## see eq 3.1 in the thesis and chapter 4.1
## plots and comparision between simulation and analytical solution


##Analytical time dep. solution of the FP equation 
def analytical_OU(x, x1, t, b, d ):
    avg_v = x1 * np.exp(-b * t)
    var_v = d / b * (1 - np.exp(- 2 * b * t))
    return 1 / np.sqrt((2 * np.pi * var_v)) * np.exp( - (x - avg_v) **2 / (2 * var_v))

## Stationary distribution of the OU process
def stationary_OU(x,b,d):
    var = d / b
    return 1 / np.sqrt(2 * np.pi * var ) * np.exp(- x**2 / (2 * var))

## simulation steps of the process
def process_OU(x1, b, d, T):
    x = x1
    dt = 10 ** -3 
    N = int(T / dt)
    x_steps = []
    x_steps.append(x)
    for i in range(0, N-1):
      x = x - b * x * dt + np.sqrt(2 * d * dt) * np.random.normal(0,1)
      x_steps.append(x)
    return x_steps

#%%
##simulating 10^4 trajectories x0=1, beta=sqrt(8), d=4, in 5 seconds
## corresponds to eq 4.1 and simulation steps as in eq 2.25
x_all = []
for i in range(0,10000):
    x_all.append(process_OU(1, 1, 2 , 5))
x_all = np.array(x_all)
#%%
##Plotting some trajectories of the process
##PLotting some time instances as dashed lines to take their histog.  
T = np.linspace(0, 5 , 5000 )
plt.figure(figsize = (12,8))
for i in range(0,5):
    plt.plot(T, x_all[i], '-',  linewidth= 1)
plt.xlabel('Time', weight = 'bold', fontsize = 12) 
plt.ylabel('x', weight = 'bold',fontsize = 12)  
plt.axvline(x = T[1000], linestyle = '--' )
plt.axvline(x = T[100], linestyle = '--' )
plt.axvline(x = T[500], linestyle = '--' )
plt.axvline(x = T[4000], linestyle = '--' )
#plt.savefig("OU_x.svg")
#plt.savefig("OU_x.jpg")
plt.show()

#%%
## Ploting the distr. of the process and Comparing at 4 dfferent times with the analytical solution
## the times taken are 0.02, 0.1, 0.5, 5

fig, axes = plt.subplots(nrows=2, ncols = 2, figsize=(9, 7),sharex='col', sharey='row',  gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
axes[0,0].hist(x_all[:, 20], bins = 50 , density = True) ## histogram at t = 0.1
axes[0,0].plot(x_all[:,  20], analytical_OU(x_all[:, 20], 1, T[20], 1, 2),'ok', markersize = 1.5, label = 'Time Dep. Solution')   
#axes[0,0].set_xlabel(r'x', fontsize = 14)
axes[0,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
axes[0,0].text(2.4, 0.6, 't = '+ str(np.round(T[20],2)) ,fontsize = 15)
axes[0,0].set_xlim(-5, 5)
axes[0,0].set_ylim(0,1.5)
axes[0,0].tick_params('y',labelsize=17)
####################################################################
axes[0,1].hist(x_all[:, 120], bins = 50 , density = True)
axes[0,1].text(3, 0.6, 't = '+ str(np.round(T[120],2)),fontsize = 15)
axes[0,0].set_xlim(-5, 5)
axes[0,1].plot(x_all[:,  120], analytical_OU(x_all[:, 120], 1, T[120], 1, 2), 'ok', markersize = 1.5, label = 'Time Dep. Solution')
####################################################################
axes[1,0].hist(x_all[:, 500], bins = 50 , density = True)
axes[1,0].plot(x_all[:,  500], analytical_OU(x_all[:, 500], 1, T[500], 1, 2),'ok', markersize = 1.5 , label = 'Time Dep. Solution')
axes[1,0].plot(x_all[:,  500], stationary_OU(x_all[:, 500], 1, 2), '.r', markersize = 0.5 , label = 'Stationary Solution')
axes[1,0].tick_params('both',labelsize=17)
axes[1,0].set_xlabel(r'x', fontsize = 20 )
axes[1,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
axes[1,0].text(2, 0.6, 't = '+ str(np.round(T[500],2)),fontsize = 15)
########################################################################
axes[1,1].tick_params('x',labelsize=17)
axes[1,1].hist(x_all[:, 4000], bins = 100 , density = True)
axes[1,1].plot(x_all[:,  4000], analytical_OU(x_all[:, 4000], 1, T[4000], 1, 2),'ok', markersize = 1.5,label = 'Time Dep. Distribution')
axes[1,1].plot(x_all[:,  4000], stationary_OU(x_all[:, 4000], 1, 2), 'or', markersize = 1.5 , label = 'Stationary Distribution')
axes[1,1].set_xlabel(r'x', fontsize = 20 )
axes[1,1].set_ylim(0, 1.1)
axes[1,1].text(3, 0.6, 't = 4.0 ', fontsize = 15)
axes[1,1].legend(fontsize= 14, markerscale = 4)
plt.show()
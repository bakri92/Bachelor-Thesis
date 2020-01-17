#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:02:36 2019

@author: georgef
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import special 
#%%
## Simulating the ED Process with exponentailly stationary distr.,
## See equation 2.3 in the thesis and chapter 2
## Plots and comparision between simulation and analytical solution

##The time dependent solution of the corresponding Fokker-PLanck equation
def analytical_ED(x, x1, t, b, d ): 
    p1 = (4 * np.pi * d * t)**(-1/2) * np.exp(- (x - x1 + b * t)**2 / (4 * t *d))
    p2 = (4 * np.pi * d * t)**(-1/2) * np.exp((b * x1 / d)- (x + x1 + b * t)**2 / (4 * t *d))
    p3 = b / (2 * d) * np.exp(-b * x / d) * special.erfc((x + x1 - b * t) / (2 * np.sqrt(d * t)))
    return p1 + p2 + p3 

## the stationary distribution 
def stationary_process(x, b, d):
    return b / d * np.exp(-x * b / d)

## Simulation of the process 
def process_ed1(x0, b, d, T):
    x = x0  ## starting point
    dt = 10 ** -3  ## time step
    N = int(T / dt)
    x_steps = []
    x_steps.append(x)
    for i in range(0, N):
        # the SDE
        x = x - b * dt + np.sqrt(2 * d * dt) * np.random.normal(0,1)
        if x <= 0:  ## reflective boundary condition at the origin
            x = - x
        x_steps.append(x)
    return x_steps

#%%
 ##simulating 10^4 trajectories x0=1, beta=sqrt(8), d=4, in 5 seconds
## corresponds to eq 2.3 and simulation steps as in eq 2.25
x_all = []
for i in range(0,10000):  
    x_all.append(process_ed1(1, np.sqrt(8), 4 , 5))
x_all = np.array(x_all)
    
#%%
T = np.linspace(0, 5 , 5001)

#Plotting some trajectories of the process
plt.figure(figsize = (9,6))
for i in range(8,14):
    plt.plot(T, x_all[i], '-',  linewidth= 0.5)
plt.xlabel('t', fontsize = 20) 
plt.ylabel('x',fontsize = 20)  
## the times taken to compare the histograms 
plt.axvline(x = T[20], linestyle = '--' ) 
plt.axvline(x = T[100], linestyle = '--' )
plt.axvline(x = T[500], linestyle = '--' )
plt.axvline(x = T[4000], linestyle = '--' )
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.savefig("ED_x.svg")
#plt.savefig('ED_x.pdf', format = 'pdf')
plt.show()
#%%
## Ploting the distr. of the process and Comparing at 4 dfferent times with the analytical solution
## the times taken are 0.02, 0.1, 0.5, 5
fig, axes = plt.subplots(nrows=2, ncols = 2, figsize=(9, 7),sharex='col', sharey='row',  gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
axes[0,0].hist(x_all[:, 20], bins = 50 , density = True) ## histogram at t = 0.1
axes[0,0].plot(x_all[:,  20], analytical_ED(x_all[:, 20], 1, T[20], np.sqrt(8), 4),'ok', markersize = 1.5, label = 'Time Dep. Solution')   
#axes[0,0].set_xlabel(r'x', fontsize = 14)
axes[0,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
axes[0,0].text(3, 0.6, 't = '+ str(T[20]) ,fontsize = 15)
axes[0,0].set_xlim(-0.05, 7)
axes[0,0].set_ylim(0,1.1)
axes[0,0].tick_params('y',labelsize=17)
####################################################################
axes[0,1].hist(x_all[:, 120], bins = 50 , density = True)
axes[0,1].text(3, 0.6, 't = '+ str(T[100]),fontsize = 15)
axes[0,1].set_xlim(-0.05, 7)
axes[0,1].plot(x_all[:,  120], analytical_ED(x_all[:, 120], 1, T[120], np.sqrt(8), 4), 'ok', markersize = 1.5, label = 'Time Dep. Solution')
####################################################################
axes[1,0].hist(x_all[:, 500], bins = 50 , density = True)
axes[1,0].plot(x_all[:,  500], analytical_ED(x_all[:, 500], 1, T[500], np.sqrt(8), 4),'ok', markersize = 1.5 , label = 'Time Dep. Solution')
axes[1,0].plot(x_all[:,  500], stationary_process(x_all[:, 500], np.sqrt(8), 4), '.r', markersize = 0.5 , label = 'Stationary Solution')
axes[1,0].tick_params('both',labelsize=17)
axes[1,0].set_xlabel(r'x', fontsize = 20 )
axes[1,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
axes[1,0].text(3, 0.6, 't = '+ str(T[500]),fontsize = 15)
########################################################################
axes[1,1].tick_params('x',labelsize=17)
axes[1,1].hist(x_all[:, 4000], bins = 100 , density = True)
axes[1,1].plot(x_all[:,  4000], analytical_ED(x_all[:, 4000], 1, T[4000], np.sqrt(8), 4),'ok', markersize = 1.5,label = 'Time Dep. Distribution')
axes[1,1].plot(x_all[:,  4000], stationary_process(x_all[:, 4000], np.sqrt(8), 4),  'or', markersize = 1.5 , label = 'Stationary Distribution')
axes[1,1].set_xlabel(r'x', fontsize = 20 )
axes[1,1].set_ylim(0, 1.1)
axes[1,1].text(3, 0.6, 't = 4.0 ', fontsize = 15)
axes[1,1].legend(fontsize= 14, markerscale = 4)
plt.show()
## for the complete figure with the explanation see figure 2 in the thesis 
    

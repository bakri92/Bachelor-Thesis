#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:58:32 2019

@author: georgef
"""

import numpy as np 
import matplotlib.pyplot as plt
#%%
## simulating the LA process which has a Laplace stationary dist.
## see eq 2.7 in the thesis and chapter 2
## plots and comparision between simulation and analytical solution

## The stationary solution which is Laplace distribution 
def stationary_process(x, b, d): 
    return b / (2 * d) * np.exp(- np.abs(x) * b / d)

def process_LA(x1, b, d, T):
    x = x1
    dt = 10 ** -3 
    N = int(T / dt)
    x_steps = []
    x_steps.append(x)
    for i in range(0, N):
    ## the two cases for the absolute value of x
        if x >= 0: 
            x = x - b * dt + np.sqrt(2 * d * dt) * np.random.normal(0,1)
        elif x < 0:
            x = x + b * dt + np.sqrt(2 * d * dt) * np.random.normal(0,1)
        x_steps.append(x)
    return x_steps

#%%
## simulating 10^4 trajectories x0=1, beta=2.5, d=2.5, in 5 seconds
## corresponds to eq 2.7 and eq 2.25
x_all = []
for i in range(0,10000):
    x_all.append(process_LA(1, 5 / 2, 5 / 2, 5))
x_all = np.array(x_all) ##making them an array 

#%%
T = np.linspace(0, 5 , 5001 )
## plotting several trajectories of the process
plt.figure(figsize = (9,6))
for i in range(5,10):
    plt.plot(T, x_all[i],  linewidth= 1)
plt.xlabel('t', fontsize = 20) 
plt.ylabel('x',fontsize = 20)  
## the time instances of the histog. as dashed lines
plt.axvline(x = T[50], linestyle = '--' )
plt.axvline(x = T[100], linestyle = '--' )
plt.axvline(x = T[500], linestyle = '--' )
plt.axvline(x = T[4000], linestyle = '--' )
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.savefig("ED2_x.svg")
#plt.savefig("ED2_x.pdf", format = 'pdf')
plt.show()

#%%
## Ploting the distr. of the process and Comparing at 4 dfferent times with the analytical solution
## the times taken are 0.05, 0.1, 0.5, 5
fig, axes = plt.subplots(nrows=2, ncols = 2, figsize=(9, 7),sharex='col', sharey='row',  gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

axes[0,0].hist(x_all[:, 50], bins = 50 , density = True)
axes[0,0].set_ylim(0,0.9)
axes[0,0].tick_params('y',labelsize=17)
axes[0,0].text(-6, 0.6, 't = '+ str(T[50]),fontsize = 15)
axes[0,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
###############################
axes[0,1].hist(x_all[:, 100], bins = 50 , density = True)
axes[0,1].text(-6, 0.7, 't = '+ str(T[100]),fontsize = 15)
###########################
axes[1,0].hist(x_all[:, 500], bins = 50 , density = True)
axes[1,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
axes[1,0].text(-6, 0.6, 't = '+ str(T[500]),fontsize = 15)
axes[1,0].set_xlim(-7, 7)
axes[1,0].set_ylim(0,0.9)
axes[1,0].set_xlabel(r'x', fontsize = 20 )
axes[1,0].set_ylabel(r'$P(x, t | x \prime )$', fontsize = 20)
axes[1,0].tick_params('both',labelsize=17)
###############################
axes[1,1].hist(x_all[:, 4000], bins = 100 , density = True)
axes[1,1].plot(x_all[:,  4000], stationary_process(x_all[:, 4000],5 / 2 , 5 / 2 ),  'or', markersize = 1.5 , label = 'Stationary Distribution')
axes[1,1].set_xlabel(r'x', fontsize = 20 )
axes[1,1].set_xlim(-7, 7)
axes[1,1].set_ylim(0,0.9)
axes[1,1].legend(fontsize= 15, markerscale = 4)
axes[1,1].text(-6, 0.6, 't = 4.0', fontsize = 15)
axes[1,1].tick_params('x',labelsize=17)
#plt.savefig("E2_T2.svg")
#plt.savefig("E2_T2.pdf")
plt.show()
## for the whole plot see figure 3 in the thesis
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:47:34 2020

@author: georg
"""
import numpy as np
#%%

### Simuating Escape rate counter with white gaussian noise
def w_gauss (D, N_sample, T, dt):
    x0 = -10
    xf = 10
    count2 = 0
    count_tot = []
    x = x0
    N = int(T/dt)
    for i in range(0, N_sample):
        x = x0
        for j in range(0, N):
            x = x + dt * (x ** 2 - 1) + (np.sqrt(2 * D * dt)) * np.random.normal(0,1)
            if x > xf :
                count2 += 1
                x = x0
        count_tot.append(count2)
        count2 = 0 
    return  count_tot

### Simuating Escape rate counter with white lapalce noise
def w_laplace (D, N_sample, T, dt):
    x0 = -10
    xf = 10
    count2 = 0
    count_tot = []
    x = x0
    N = int(T/dt)
    for i in range(0, N_sample):
        x = x0
        for j in range(0, N):
            x = x + dt * (x ** 2 - 1) + (np.sqrt(2 * D * dt)) * np.random.laplace(0,np.sqrt(1/2))
            if x > xf :
                count2 += 1
                x = x0
        count_tot.append(count2)
        count2 = 0 
    return  count_tot


## Laplace noise with noise intensity 
def LA_D(D, tau, T, dt):
    N = int(T / dt)
    var = D / tau
    x = np.zeros(N)
    x[0] = np.random.laplace(0, np.sqrt(var / 2)) 
    D1 = - 5 * dt / (2 * tau) * np.sqrt(D / (2 * tau)) ##first term of the SDE
    D2 = np.sqrt(5 * D * dt / (2 * tau ** 2)) * np.random.normal(0, 1, N) #2nd term with WN
    for i in range(1, N):
        x[i] = x[i-1] + np.sign(x[i-1]) * D1 + D2[i]
    return x   

## Laplace noise with vairance  
def LA_var(var, tau, T, dt):
    N = int(T / dt)
    std = np.sqrt(var)
    x = np.zeros(N)
    x[0] = np.random.laplace(0, np.sqrt(var / 2))
    D1 = - 5 * std * dt / (2 * np.sqrt(2) * tau)  ##first term of the SDE
    D2 = np.sqrt(5 * var * dt / (2 * tau)) * np.random.normal(0, 1, N) #2nd term with WN
    for i in range(1, N):
        x[i] = x[i-1] + np.sign(x[i-1]) * D1 + D2[i]
    return x

## OU noise with variance
def OU_var(var, tau, T, dt):
    N = int(T / dt)
    x = np.zeros(N)
    x[0] = np.random.normal(0, np.sqrt(var))
    D1 = - dt / tau  ##first term of the SDE
    D2 =  (np.sqrt(2 * var * tau * dt) / tau) * np.random.normal(0,1, N) #2nd term with WN
    for i in range(1, N):
        x[i] = x[i-1] + x[i-1] * D1 + D2[i-1]
    return x


## OU noise with noise intenity
def OU_D(D, tau, T, dt):
    N = int(T / dt)
    var = D / tau
    x = np.zeros(N)
    x[0] = np.random.normal(0, np.sqrt(var))
    D1 = - dt / tau       ##first term of the SDE
    D2 = np.sqrt(2 * var * tau * dt) / tau * np.random.normal(0,1, size = N) #2nd term with WN
    for i in range(1, N):                 ##integrating
        x[i] = x[i-1] + x[i - 1] * D1 + D2[i - 1]
    return x



##Correlation Functon in var space
def Corr_LA_var(tau, var, T, dt):
    points = []
    N = int(T / dt)
    process = LA_var(tau, var, N, dt)  ##Trajectory
    ## power specturm:
    psd = np.fft.fft(process) * np.conj(np.fft.fft(process)) / N  
    C = np.fft.ifft(psd).real - np.mean(process) ** 2 ##corr iFFT
    points.append(choose_points(t,C)) 
    return points

##Correlation Functon in Noise intensity 
def Corr_LA_D(tau, D, T, dt):
    points = []
    N = int(T / dt)
    t = np.arange(0, T, dt)
    process = LA_D(tau, D, N, dt)  ##Trajectory
    ## power specturm:
    psd = np.fft.fft(process) * np.conj(np.fft.fft(process)) / N  
    C = np.fft.ifft(psd).real - np.mean(process) ** 2 ##corr iFFT
    points.append(choose_points(t,C)) 
    return points

## escpae rate with Laplace colored noise 
def escD_step (tau, D, N_sample, T, dt):
    x0 = -10
    xf = 10
    count2 = 0
    count_tot = []
    N = int(T/dt)
    x = np.zeros(N)
    x_all = []
    for i in range(0, N_sample):
        ys = LA_D(D, tau, T, dt)
        x[0] = x0
        for j in range(1, N):      
            x[j] = x[j- 1] + dt * ( x[j- 1] ** 2 - 1 + ys[j- 1])
            if x[j] > xf :
                count2 += 1
                x[j] = x0
        x_all.append(x)
        x = np.zeros(N)
        count_tot.append(count2)
        count2 = 0 
    return  count_tot, x_all

## escpae rate with OU colored noise 
def escOU_step (tau, D, N_sample, T, dt):
    x0 = -10
    xf = 10
    count2 = 0
    count_tot = []
    N = int(T/dt)
    x = np.zeros(N)
    x_all = []
    for i in range(0, N_sample):
        ys = OU_D(D, tau, T, dt)
        x[0] = x0
        for j in range(1, N):      
            x[j] = x[j- 1] + dt * ( x[j- 1] ** 2 - 1 + ys[j- 1])
            if x[j] > xf :
                count2 += 1
                x[j] = x0
        x_all.append(x)
        x = np.zeros(N)
        count_tot.append(count2)
        count2 = 0 
    return  count_tot, x_all

## Lplace distribution 
def laplace(x, var, mean):
 #   var = D / tau
    return 1 / (np.sqrt(2 * var)) * np.exp(- np.abs(x - mean) * np.sqrt(2 / var)) 

## normal distribution 
def normal(x, var, mean):
    #var = D / tau
    return 1 / np.sqrt(2 * var * np.pi) * np.exp(- (x - mean) ** 2 / (2 * var)) 


##### Kramers-Moryal coeffiecients #####  
## choose indecies for Kramers Moryal 
def ind(x, x0): 
    a = np.where(np.isclose([ x0 ], x, atol = 0.05) == True)[0]
    return a
### Choose time points
def points_time(x,x0, T):
    a = np.where(np.isclose([ x0 ], x, atol = 0.01) == True)[0]
    return x[a], T[a]
## find delta x
def delta_x(del_t, T0, interp):
    return interp(T0 + del_t) - interp(T0)

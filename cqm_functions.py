# -*- coding: utf-8 -*-
"""
Created on Wednesday June 12 19:26:43 2024

@author: zaxari
"""

"""
This code contains all the functions used by the main code written for Computational Quantum Mechanics course project 
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd


# Symbols
r=sp.symbols("r")
k=sp.symbols("k")

## Functions related to positions (Coordinate Space) ##

# Radial Functions (STOs) for the needed combinations of principal quantum numbers and azimuthial quantum numbers (n,l)

# n=1, l=0 (1s state)
def S_1s(z,r):
    return (2*z**(3/2))*(sp.exp(-z*r)) # z is the orbital exponent

# n=2, l=0, l=1 (2s and 2p states)
def S_2(z,r):
    return (2/sp.sqrt(3))*(z**(5/2)*r)*(sp.exp(-z*r))

# n=3, l=0 (3s state)
def S_3s(z,r):
    return (2**(3/2)/(3*sp.sqrt(5)))*(z**(7/2)*r**2)*(sp.exp(-z*r))

# RHF wavefunctions

def wf_R1s(c_1s, c_2s, c_2p, s_vals, p_vals): # For Helium
    R1s=0
    R2s=0
    R2p=0
    
    for i in range(7):
    
        R2p=R2p+c_2p[i]*S_2(p_vals[i],r)
    
        if i==0:
            R1s=R1s+c_1s[i]*S_1s(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_1s(s_vals[i],r)
        elif i==1:
            R1s=R1s+c_1s[i]*S_3s(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_3s(s_vals[i],r)
        else:
            R1s=R1s+c_1s[i]*S_2(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_2(s_vals[i],r) 
            
    return R1s,R2s,R2p

def wf_R3s(c_1s, c_2s, c_2p, s_vals, p_vals): 
    R1s=0
    R2s=0
    R2p=0
    
    for i in range(7):
    
        R2p=R2p+c_2p[i]*S_2(p_vals[i],r)
    
        if i==0 or i==1:
            R1s=R1s+c_1s[i]*S_1s(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_1s(s_vals[i],r)
        elif i==2:
            R1s=R1s+c_1s[i]*S_3s(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_3s(s_vals[i],r)
        else:
            R1s=R1s+c_1s[i]*S_2(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_2(s_vals[i],r) 
            
    return R1s,R2s,R2p

def wf_R3s_2(c_1s, c_2s, c_2p, s_vals, p_vals): # For Be, B
    R1s=0
    R2s=0
    R2p=0
    
    for i in range(7):
    
        R2p=R2p+c_2p[i]*S_2(p_vals[i],r)
    
        if i==0 or i==1:
            R1s=R1s+c_1s[i]*S_1s(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_1s(s_vals[i],r)
        elif i==2 or i==3:
            R1s=R1s+c_1s[i]*S_3s(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_3s(s_vals[i],r)
        else:
            R1s=R1s+c_1s[i]*S_2(s_vals[i],r)
            R2s=R2s+c_2s[i]*S_2(s_vals[i],r) 
            
    return R1s,R2s,R2p

# Symbolic Integration (r, 0, inf) # Will be used for entropy Sr

def r_sym_int(f):
    num_f=sp.lambdify(r,f,modules=["numpy"]) # Converting symbolic to numerical
    def integrand(r):
        return num_f(r)
    int_r, e_r=quad(integrand, 0, np.inf)
    return int_r

## Functions related to k-Space ##

# k-space STOs
def Sk_1s(z,k):
    return (1/((2*sp.pi)**(3/2)))*(16*sp.pi*z**(5/2))/((z**2)+(k**2))**2

def Sk_2s(z,k):
    return (1/((2*sp.pi)**(3/2)))*(16*sp.pi*z**(5/2))*(3*z**2 -k**2)/(sp.sqrt(3)*(z**2 +k**2)**3)

def Sk_3s(z,k):
    return (1/(2*sp.pi)**(3/2))*(64*sp.sqrt(10)*sp.pi*(z**(9/2)))*((z**2 -k**2)/(5*(z**2 +k**2)**4))

def Sk_2p(z,k):
    return (1/(2*sp.pi)**(3/2))*(64*sp.pi*k*(z**(7/2)))/(sp.sqrt(3)*(z**2 +k**2)**3)

# RHF wavefunctions in k-space

def wf_K1s(c_1s,c_2s,c_2p,s_vals,p_vals):
    K1s=0
    K2s=0
    K2p=0
    
    for i in range(7):
        K2p=K2p+c_2p[i]*Sk_2p(p_vals[i],k)
    
        if i==0:
            K1s=K1s+c_1s[i]*Sk_1s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_1s(s_vals[i],k)
        elif i==1:
            K1s=K1s+c_1s[i]*Sk_3s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_3s(s_vals[i],k)
        else:
            K1s=K1s+c_1s[i]*Sk_2s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_2s(s_vals[i],k)
            
    return K1s,K2s,K2p

def wf_K3s(c_1s,c_2s,c_2p,s_vals,p_vals):
    K1s=0
    K2s=0
    K2p=0
    
    for i in range(7):
        K2p=K2p+c_2p[i]*Sk_2p(p_vals[i],k)
    
        if i==0 or i==1:
            K1s=K1s+c_1s[i]*Sk_1s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_1s(s_vals[i],k)
        elif i==2:
            K1s=K1s+c_1s[i]*Sk_3s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_3s(s_vals[i],k)
        else:
            K1s=K1s+c_1s[i]*Sk_2s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_2s(s_vals[i],k)
            
    return K1s,K2s,K2p

def wf_K3s_2(c_1s,c_2s,c_2p,s_vals,p_vals):
    K1s=0
    K2s=0
    K2p=0
    
    for i in range(7):
        K2p=K2p+c_2p[i]*Sk_2p(p_vals[i],k)
    
        if i==0 or i==1:
            K1s=K1s+c_1s[i]*Sk_1s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_1s(s_vals[i],k)
        elif i==2 or i==3:
            K1s=K1s+c_1s[i]*Sk_3s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_3s(s_vals[i],k)
        else:
            K1s=K1s+c_1s[i]*Sk_2s(s_vals[i],k)
            K2s=K2s+c_2s[i]*Sk_2s(s_vals[i],k)
            
    return K1s,K2s,K2p

# Symbolic Integration (k, 0, inf) # Will be used for entropy Sk

def k_sym_int(f):
    num_f=sp.lambdify(k,f,modules=["numpy"]) # Converting symbolic to numerical
    int_k, e_r=quad(num_f, 0, np.inf)
    return int_k


# Function to plot the DOS in position space and k-space.
def dos_plot(p_sym, n_sym, element_name):
    
    # Convert symbolic expressions to numerical functions
    p_num = sp.lambdify(r, p_sym, modules=['numpy'])
    n_num = sp.lambdify(k, n_sym, modules=['numpy'])
    
    # Determine suitable ranges for r and k
    r_vals = np.linspace(0, 4, 500)
    k_vals = np.linspace(0, 4, 500)

    # Evaluate the numerical functions over the given ranges
    p_vals = p_num(r_vals)
    n_vals = n_num(k_vals)
    
    fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))

    # Plotting ρ(r) against r
    ax1[0].plot(r_vals, p_vals, label='ρ(r) (Position Space)', color='b')
    ax1[0].set_xlabel('r')
    ax1[0].set_ylabel('Density of States')
    ax1[0].set_title(f'{element_name} Density of States in Position Space')
    ax1[0].legend()
    ax1[0].grid(True)

    # Plotting η(k) against k
    ax1[1].plot(k_vals, n_vals, label='η(k) (k-Space)', color='r')
    ax1[1].set_xlabel('k')
    ax1[1].set_ylabel('Density of States in k-Space')
    ax1[1].set_title(f'{element_name} Density of States in k-Space')
    ax1[1].legend()
    ax1[1].grid(True)

    plt.tight_layout()
    plt.show() 

# Function to create excel files for the calculated data of each element    
def create_element_excel(element, data):
 
  df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
  df.index.name = 'Property'
  filename = f'element_{element}.xlsx'
  df.to_excel(filename)

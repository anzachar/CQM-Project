# -*- coding: utf-8 -*-
"""
Created on Thursday June 13 00:07:17 2024

@author: zaxari
"""

"""
This is the main part of the code written for the Computational Quantum Mechanics course project 
"""

import cqm_functions as RHF
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

# Initializing lists to store values

Sr_vals=[] # list to store values of information entropy in position space for each element
Sk_vals=[] # list to store values of information entropy in k-space for each element
Stot_vals=[] # list to store values of total information entropy for each element


r=sp.symbols("r")
k=sp.symbols("k")

#----- He (Z=2) -----#
print("Results for He (Z=2)")
Z=2 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=0.0 # number of electrons on the 2s orbital
e_2p=0.0 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
He_c_1s=[1.347900, -0.001613, -0.100506, -0.270779, 0.0, 0.0, 0.0]
He_c_2s=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
He_c_2p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
He_s_vals=[1.4595, 5.3244, 2.6298, 1.7504, 1, 1, 1]
He_p_vals=[1, 1, 1, 1, 1, 1, 1,]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
He_R1s, He_R2s, He_R2p=RHF.wf_R1s(He_c_1s, He_c_2s, He_c_2p, He_s_vals, He_p_vals)

# Calculating integrals
He_int_R1s=RHF.r_sym_int(He_R1s**2*r**2)
print("I_R1s= ",He_int_R1s)
He_int_R2s=RHF.r_sym_int(He_R2s**2*r**2)
print("I_R2s= ",He_int_R2s)
He_int_R2p=RHF.r_sym_int(He_R2p**2*r**2)
print("I_R2p= ",He_int_R2p)

# Density of states
He_p=(1/(4*np.pi*Z))*(e_1s*He_R1s**2 +e_2s*He_R2s**2 +e_2p*He_R2p**2)

# Confirming normalization (integral should be equal to 1)
He_Ip=(RHF.r_sym_int(4*sp.pi*He_p*r**2))
print("Ip= ", He_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
He_Sr=(RHF.r_sym_int(-4*sp.pi*He_p*sp.log(He_p+eps)*r**2))
Sr_vals.append(He_Sr)
print("Sr entropy= ", He_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
He_K1s, He_K2s, He_K2p=RHF.wf_K1s(He_c_1s, He_c_2s, He_c_2p, He_s_vals, He_p_vals)

# Calculating integrals
He_int_K1s=RHF.k_sym_int(He_K1s**2*k**2)
print("I_K1s= ",He_int_K1s)
He_int_K2s=RHF.k_sym_int(He_K2s**2*k**2)
print("I_K2s= ",He_int_K2s)
He_int_K2p=RHF.k_sym_int(He_K2p**2*k**2)
print("I_K2p= ",He_int_K2p)

# Density of states
He_n=(1/(4*np.pi*Z))*(e_1s*He_K1s**2 +e_2s*He_K2s**2 +e_2p*He_K2p**2)

# Confirming normalization (integral should be equal to 1)
He_In=(RHF.k_sym_int(4*sp.pi*He_n*k**2))
print("In= ", He_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
He_Sk=(RHF.k_sym_int(-4*sp.pi*He_n*sp.log(He_n+eps)*k**2))
Sk_vals.append(He_Sk)
print("Sk entropy= ", He_Sk)

# Total information entropy
He_S=He_Sr+He_Sk
Stot_vals.append(He_S)
print("S= ", He_S)


# Density of States Plots
RHF.dos_plot(He_p, He_n, 'He')
print("\n")


data = {
    'I_R1s': He_int_R1s, 'I_R2s': He_int_R2s, 'I_R2p': He_int_R2p, 'Ip': He_Ip, 'Sr': He_Sr, 'I_K1s': He_int_K1s, 'I_K2s': He_int_K2s, 'I_K2p': He_int_K2p, 'In': He_In, 'Sk': He_Sk, 'S': He_S
    
}

RHF.create_element_excel('He', data)

#----- Li (Z=3) -----#
print("Results for Li (Z=3)")
Z=3 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=1 # number of electrons on the 2s orbital
e_2p=0.0 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
Li_c_1s=[0.141279, 0.874231, -0.005201 ,-0.002307 ,0.006985 ,-0.000305 ,0.000760]
Li_c_2s=[-0.022416,-0.135791,0.000389,-0.000068,-0.076544,0.340542,0.715708]
Li_c_2p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Li_s_vals=[4.3069 ,2.4573 ,6.7850 ,7.4527 ,1.8504,0.7667,0.6364]
Li_p_vals=[1, 1, 1, 1, 1, 1, 1,]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
Li_R1s, Li_R2s, Li_R2p=RHF.wf_R3s(Li_c_1s, Li_c_2s, Li_c_2p, Li_s_vals, Li_p_vals)

# Calculating integrals
Li_int_R1s=RHF.r_sym_int(Li_R1s**2*r**2)
print("I_R1s= ",Li_int_R1s)
Li_int_R2s=RHF.r_sym_int(Li_R2s**2*r**2)
print("I_R2s= ",Li_int_R2s)
Li_int_R2p=RHF.r_sym_int(Li_R2p**2*r**2)
print("I_R2p= ",Li_int_R2p)

# Density of states
Li_p=(1/(4*np.pi*Z))*(e_1s*Li_R1s**2 +e_2s*Li_R2s**2 +e_2p*Li_R2p**2)

# Confirming normalization (integral should be equal to 1)
Li_Ip=(RHF.r_sym_int(4*sp.pi*Li_p*r**2))
print("Ip= ", Li_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
Li_Sr=(RHF.r_sym_int(-4*sp.pi*Li_p*sp.log(Li_p+eps)*r**2))
Sr_vals.append(Li_Sr)
print("Sr entropy= ", Li_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
Li_K1s, Li_K2s, Li_K2p=RHF.wf_K3s(Li_c_1s, Li_c_2s, Li_c_2p, Li_s_vals, Li_p_vals)

# Calculating integrals
Li_int_K1s=RHF.k_sym_int(Li_K1s**2*k**2)
print("I_K1s= ",Li_int_K1s)
Li_int_K2s=RHF.k_sym_int(Li_K2s**2*k**2)
print("I_K2s= ",Li_int_K2s)
Li_int_K2p=RHF.k_sym_int(Li_K2p**2*k**2)
print("I_K2p= ",Li_int_K2p)

# Density of states
Li_n=(1/(4*np.pi*Z))*(e_1s*Li_K1s**2 +e_2s*Li_K2s**2 +e_2p*Li_K2p**2)

# Confirming normalization (integral should be equal to 1)
Li_In=(RHF.k_sym_int(4*sp.pi*Li_n*k**2))
print("In= ", Li_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
Li_Sk=(RHF.k_sym_int(-4*sp.pi*Li_n*sp.log(Li_n+eps)*k**2))
Sk_vals.append(Li_Sk)
print("Sk entropy= ", Li_Sk)

# Total information entropy
Li_S=Li_Sr+Li_Sk
Stot_vals.append(Li_S)
print("S= ", Li_S)


# Density of States Plots
RHF.dos_plot(Li_p, Li_n, 'Li')
print("\n")

data = {
    'I_R1s': Li_int_R1s, 'I_R2s': Li_int_R2s, 'I_R2p': Li_int_R2p, 'Ip': Li_Ip, 'Sr': Li_Sr, 'I_K1s': Li_int_K1s, 'I_K2s': Li_int_K2s, 'I_K2p': Li_int_K2p, 'In': Li_In, 'Sk': Li_Sk, 'S': Li_S
    
}

RHF.create_element_excel('Li', data)

#----- Be (Z=4) -----#
print("Results for Be (Z=4)")
Z=4 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=0.0 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
Be_c_1s=[0.285107 ,0.474813 ,-0.001620 ,0.052852,0.243499 ,0.000106 ,-0.000032]
Be_c_2s=[-0.016378,-0.155066,0.000426,-0.059234,-0.031925,0.387968,0.685674]
Be_c_2p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Be_s_vals=[5.7531 ,3.7156 ,9.9670 ,3.7128 ,4.4661,1.2919,0.8555]
Be_p_vals=[1, 1, 1, 1, 1, 1, 1,]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
Be_R1s, Be_R2s, Be_R2p=RHF.wf_R3s_2(Be_c_1s, Be_c_2s, Be_c_2p, Be_s_vals, Be_p_vals)

# Calculating integrals
Be_int_R1s=RHF.r_sym_int(Be_R1s**2*r**2)
print("I_R1s= ",Be_int_R1s)
Be_int_R2s=RHF.r_sym_int(Be_R2s**2*r**2)
print("I_R2s= ",Be_int_R2s)
Be_int_R2p=RHF.r_sym_int(Be_R2p**2*r**2)
print("I_R2p= ",Be_int_R2p)
# Density of states
Be_p=(1/(4*np.pi*Z))*(e_1s*Be_R1s**2 +e_2s*Be_R2s**2 +e_2p*Be_R2p**2)

# Confirming normalization (integral should be equal to 1)
Be_Ip=(RHF.r_sym_int(4*sp.pi*Be_p*r**2))
print("Ip= ", Be_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
Be_Sr=(RHF.r_sym_int(-4*sp.pi*Be_p*sp.log(Be_p+eps)*r**2))
Sr_vals.append(Be_Sr)
print("Sr entropy= ", Be_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
Be_K1s, Be_K2s, Be_K2p=RHF.wf_K3s_2(Be_c_1s, Be_c_2s, Be_c_2p, Be_s_vals, Be_p_vals)

# Calculating integrals
Be_int_K1s=RHF.k_sym_int(Be_K1s**2*k**2)
print("I_K1s= ",Be_int_K1s)
Be_int_K2s=RHF.k_sym_int(Be_K2s**2*k**2)
print("I_K2s= ",Be_int_K2s)
Be_int_K2p=RHF.k_sym_int(Be_K2p**2*k**2)
print("I_K2p= ",Be_int_K2p)

# Density of states
Be_n=(1/(4*np.pi*Z))*(e_1s*Be_K1s**2 +e_2s*Be_K2s**2 +e_2p*Be_K2p**2)

# Confirming normalization (integral should be equal to 1)
Be_In=(RHF.k_sym_int(4*sp.pi*Be_n*k**2))
print("In= ", Be_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
Be_Sk=(RHF.k_sym_int(-4*sp.pi*Be_n*sp.log(Be_n+eps)*k**2))
Sk_vals.append(Be_Sk)
print("Sk entropy= ", Be_Sk)

# Total information entropy
Be_S=Be_Sr+Be_Sk
Stot_vals.append(Be_S)
print("S= ", Be_S)

# Density of States Plots
RHF.dos_plot(Be_p, Be_n, 'Be')
print("\n")

data = {
    'I_R1s': Be_int_R1s, 'I_R2s': Be_int_R2s, 'I_R2p': Be_int_R2p, 'Ip': Be_Ip, 'Sr': Be_Sr, 'I_K1s': Be_int_K1s, 'I_K2s': Be_int_K2s, 'I_K2p': Be_int_K2p, 'In': Be_In, 'Sk': Be_Sk, 'S': Be_S
    
}

RHF.create_element_excel('Be', data)

#----- B (Z=5) -----#
print("Results for B (Z=5)")
Z=5 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=1 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
B_c_1s=[0.381607 ,0.423958 ,-0.001316 ,-0.000822,0.237016 ,0.001062 ,-0.000137]
B_c_2s=[-0.022549,0.321716,-0.000452,-0.072032,-0.050313,-0.484281,-0.518986]
B_c_2p=[0.007600 ,0.045137 ,0.184206 ,0.394754 ,0.432795 ,0.0 ,0.0]
B_s_vals=[7.0178 ,3.9468 ,12.7297 ,2.7646 ,5.7420,1.5436,1.0802]
B_p_vals=[5.7416,2.6341,1.8340,1.1919,0.8494,1,1]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
B_R1s, B_R2s, B_R2p=RHF.wf_R3s_2(B_c_1s, B_c_2s, B_c_2p, B_s_vals, B_p_vals)

# Calculating integrals
B_int_R1s=RHF.r_sym_int(B_R1s**2*r**2)
print("I_R1s= ",B_int_R1s)
B_int_R2s=RHF.r_sym_int(B_R2s**2*r**2)
print("I_R2s= ",B_int_R2s)
B_int_R2p=RHF.r_sym_int(B_R2p**2*r**2)
print("I_R2p= ",B_int_R2p)

# Density of states
B_p=(1/(4*np.pi*Z))*(e_1s*B_R1s**2 +e_2s*B_R2s**2 +e_2p*B_R2p**2)

# Confirming normalization (integral should be equal to 1)
B_Ip=(RHF.r_sym_int(4*sp.pi*B_p*r**2))
print("Ip= ", B_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
B_Sr=(RHF.r_sym_int(-4*sp.pi*B_p*sp.log(B_p+eps)*r**2))
Sr_vals.append(B_Sr)
print("Sr entropy= ", B_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
B_K1s, B_K2s, B_K2p=RHF.wf_K3s_2(B_c_1s, B_c_2s, B_c_2p, B_s_vals, B_p_vals)

# Calculating integrals
B_int_K1s=RHF.k_sym_int(B_K1s**2*k**2)
print("I_K1s= ",B_int_K1s)
B_int_K2s=RHF.k_sym_int(B_K2s**2*k**2)
print("I_K2s= ",B_int_K2s)
B_int_K2p=RHF.k_sym_int(B_K2p**2*k**2)
print("I_K2p= ",B_int_K2p)

# Density of states
B_n=(1/(4*np.pi*Z))*(e_1s*B_K1s**2 +e_2s*B_K2s**2 +e_2p*B_K2p**2)

# Confirming normalization (integral should be equal to 1)
B_In=(RHF.k_sym_int(4*sp.pi*B_n*k**2))
print("In= ", B_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
B_Sk=(RHF.k_sym_int(-4*sp.pi*B_n*sp.log(B_n+eps)*k**2))
Sk_vals.append(B_Sk)
print("Sk entropy= ", B_Sk)

# Total information entropy
B_S=B_Sr+B_Sk
Stot_vals.append(B_S)
print("S= ", B_S)

# Density of States Plots
RHF.dos_plot(B_p, B_n, 'B')
print("\n")

data = {
    'I_R1s': B_int_R1s, 'I_R2s': B_int_R2s, 'I_R2p': B_int_R2p, 'Ip': B_Ip, 'Sr': B_Sr, 'I_K1s': B_int_K1s, 'I_K2s': B_int_K2s, 'I_K2p': B_int_K2p, 'In': B_In, 'Sk': B_Sk, 'S': B_S
    
}

RHF.create_element_excel('B', data)

#----- C (Z=6) -----#
print("Results for C (Z=6)")
Z=6 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=2 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
C_c_1s=[0.352872 ,0.473621 ,-0.001199 ,0.210887,0.000886 ,0.000465 ,-0.000119]
C_c_2s=[-0.071727,0.438307,-0.000383,-0.091194,-0.393105,-0.579121,-0.126067]
C_c_2p=[0.006977 ,0.070877 ,0.230802 ,0.411931 ,0.350701 ,0.0 ,0.0]
C_s_vals=[8.4936 ,4.8788 ,15.4660 ,7.0500 ,2.2640,1.4747,1.1639]
C_p_vals=[7.0500,3.2275,2.1908,1.4413,1.0242,1,1]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
C_R1s, C_R2s, C_R2p=RHF.wf_R3s(C_c_1s, C_c_2s, C_c_2p, C_s_vals, C_p_vals)

# Calculating integrals
C_int_R1s=RHF.r_sym_int(C_R1s**2*r**2)
print("I_R1s= ",C_int_R1s)
C_int_R2s=RHF.r_sym_int(C_R2s**2*r**2)
print("I_R2s= ",C_int_R2s)
C_int_R2p=RHF.r_sym_int(C_R2p**2*r**2)
print("I_R2p= ",C_int_R2p)

# Density of states
C_p=(1/(4*np.pi*Z))*(e_1s*C_R1s**2 +e_2s*C_R2s**2 +e_2p*C_R2p**2)

# Confirming normalization (integral should be equal to 1)
C_Ip=(RHF.r_sym_int(4*sp.pi*C_p*r**2))
print("Ip= ", C_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
C_Sr=(RHF.r_sym_int(-4*sp.pi*C_p*sp.log(C_p+eps)*r**2))
Sr_vals.append(C_Sr)
print("Sr entropy= ", C_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
C_K1s, C_K2s, C_K2p=RHF.wf_K3s(C_c_1s, C_c_2s, C_c_2p, C_s_vals, C_p_vals)

# Calculating integrals
C_int_K1s=RHF.k_sym_int(C_K1s**2*k**2)
print("I_K1s= ",C_int_K1s)
C_int_K2s=RHF.k_sym_int(C_K2s**2*k**2)
print("I_K2s= ",C_int_K2s)
C_int_K2p=RHF.k_sym_int(C_K2p**2*k**2)
print("I_K2p= ",C_int_K2p)

# Density of states
C_n=(1/(4*np.pi*Z))*(e_1s*C_K1s**2 +e_2s*C_K2s**2 +e_2p*C_K2p**2)

# Confirming normalization (integral should be equal to 1)
C_In=(RHF.k_sym_int(4*sp.pi*C_n*k**2))
print("In= ", C_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
C_Sk=(RHF.k_sym_int(-4*sp.pi*C_n*sp.log(C_n+eps)*k**2))
Sk_vals.append(C_Sk)
print("Sk entropy= ", C_Sk)

# Total information entropy
C_S=C_Sr+C_Sk
Stot_vals.append(C_S)
print("S= ", C_S)


# Density of States Plots
RHF.dos_plot(C_p, C_n, 'C')
print("\n")

data = {
    'I_R1s': C_int_R1s, 'I_R2s': C_int_R2s, 'I_R2p': C_int_R2p, 'Ip': C_Ip, 'Sr': C_Sr, 'I_K1s': C_int_K1s, 'I_K2s': C_int_K2s, 'I_K2p': C_int_K2p, 'In': C_In, 'Sk': C_Sk, 'S': C_S
    
}

RHF.create_element_excel('C', data)

#----- N (Z=7) -----#
print("Results for N (Z=7)")
Z=7 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=3 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
N_c_1s=[0.354839 ,0.472579 ,-0.001038 ,0.208492,0.001687 ,0.000206 ,0.000064]
N_c_2s=[-0.067498,0.434142,-0.000315,-0.080331,-0.374128,-0.522775,-0.207735]
N_c_2p=[0.006323 ,0.082938 ,0.260147 ,0.418361 ,0.308272 ,0.0 ,0.0]
N_s_vals=[9.9051 ,5.7429 ,17.9816 ,8.3087 ,2.7611,1.8223,1.4191]
N_p_vals=[8.3490,3.8827,2.5920,1.6946,1.1914,1,1]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
N_R1s, N_R2s, N_R2p=RHF.wf_R3s(N_c_1s, N_c_2s, N_c_2p, N_s_vals, N_p_vals)

# Calculating integrals
N_int_R1s=RHF.r_sym_int(N_R1s**2*r**2)
print("I_R1s= ",N_int_R1s)
N_int_R2s=RHF.r_sym_int(N_R2s**2*r**2)
print("I_R2s= ",N_int_R2s)
N_int_R2p=RHF.r_sym_int(N_R2p**2*r**2)
print("I_R2p= ",N_int_R2p)
# Density of states
N_p=(1/(4*np.pi*Z))*(e_1s*N_R1s**2 +e_2s*N_R2s**2 +e_2p*N_R2p**2)

# Confirming normalization (integral should be equal to 1)
N_Ip=(RHF.r_sym_int(4*sp.pi*N_p*r**2))
print("Ip= ", N_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
N_Sr=(RHF.r_sym_int(-4*sp.pi*N_p*sp.log(N_p+eps)*r**2))
Sr_vals.append(N_Sr)
print("Sr entropy= ", N_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
N_K1s, N_K2s, N_K2p=RHF.wf_K3s(N_c_1s, N_c_2s, N_c_2p, N_s_vals, N_p_vals)

# Calculating integrals
N_int_K1s=RHF.k_sym_int(N_K1s**2*k**2)
print("I_K1s= ",N_int_K1s)
N_int_K2s=RHF.k_sym_int(N_K2s**2*k**2)
print("I_K2s= ",N_int_K2s)
N_int_K2p=RHF.k_sym_int(N_K2p**2*k**2)
print("I_K2p= ",N_int_K2p)

# Density of states
N_n=(1/(4*np.pi*Z))*(e_1s*N_K1s**2 +e_2s*N_K2s**2 +e_2p*N_K2p**2)

# Confirming normalization (integral should be equal to 1)
N_In=(RHF.k_sym_int(4*sp.pi*N_n*k**2))
print("In= ", N_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
N_Sk=(RHF.k_sym_int(-4*sp.pi*N_n*sp.log(N_n+eps)*k**2))
Sk_vals.append(N_Sk)
print("Sk entropy= ", N_Sk)

# Total information entropy
N_S=N_Sr+N_Sk
Stot_vals.append(N_S)
print("S= ", N_S)


# Density of States Plots
RHF.dos_plot(N_p, N_n, 'N')
print("\n")

data = {
    'I_R1s': N_int_R1s, 'I_R2s': N_int_R2s, 'I_R2p': N_int_R2p, 'Ip': N_Ip, 'Sr': N_Sr, 'I_K1s': N_int_K1s, 'I_K2s': N_int_K2s, 'I_K2p': N_int_K2p, 'In': N_In, 'Sk': N_Sk, 'S': N_S
    
}

RHF.create_element_excel('N', data)

#----- O (Z=8) -----#
print("Results for O (Z=8)")
Z=8 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=4 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
O_c_1s=[0.360063 ,0.466625 ,-0.000918 ,0.208441,0.002018 ,0.000216 ,0.000133]
O_c_2s=[-0.064363,0.433186,-0.000275,-0.072497,-0.369900,-0.512627,-0.227421]
O_c_2p=[0.005626 ,0.126618 ,0.328966 ,0.395422 ,0.231788 ,0.0 ,0.0]
O_s_vals=[11.2970 ,6.5966 ,20.5019 ,9.5546 ,3.2482,2.1608,1.6411]
O_p_vals=[9.6471,4.3323,2.7502,1.7525,1.2473,1,1]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
O_R1s, O_R2s, O_R2p=RHF.wf_R3s(O_c_1s, O_c_2s, O_c_2p, O_s_vals, O_p_vals)

# Calculating integrals
O_int_R1s=RHF.r_sym_int(O_R1s**2*r**2)
print("I_R1s= ",O_int_R1s)
O_int_R2s=RHF.r_sym_int(O_R2s**2*r**2)
print("I_R2s= ",O_int_R2s)
O_int_R2p=RHF.r_sym_int(O_R2p**2*r**2)
print("I_R2p= ",O_int_R2p)
# Density of states
O_p=(1/(4*np.pi*Z))*(e_1s*O_R1s**2 +e_2s*O_R2s**2 +e_2p*O_R2p**2)

# Confirming normalization (integral should be equal to 1)
O_Ip=(RHF.r_sym_int(4*sp.pi*O_p*r**2))
print("Ip= ", O_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
O_Sr=(RHF.r_sym_int(-4*sp.pi*O_p*sp.log(O_p+eps)*r**2))
Sr_vals.append(O_Sr)
print("Sr entropy= ", O_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
O_K1s, O_K2s, O_K2p=RHF.wf_K3s(O_c_1s, O_c_2s, O_c_2p, O_s_vals, O_p_vals)

# Calculating integrals
O_int_K1s=RHF.k_sym_int(O_K1s**2*k**2)
print("I_K1s= ",O_int_K1s)
O_int_K2s=RHF.k_sym_int(O_K2s**2*k**2)
print("I_K2s= ",O_int_K2s)
O_int_K2p=RHF.k_sym_int(O_K2p**2*k**2)
print("I_K2p= ",O_int_K2p)
# Density of states
O_n=(1/(4*np.pi*Z))*(e_1s*O_K1s**2 +e_2s*O_K2s**2 +e_2p*O_K2p**2)

# Confirming normalization (integral should be equal to 1)
O_In=(RHF.k_sym_int(4*sp.pi*O_n*k**2))
print("In= ", O_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
O_Sk=(RHF.k_sym_int(-4*sp.pi*O_n*sp.log(O_n+eps)*k**2))
Sk_vals.append(O_Sk)
print("Sk entropy= ", O_Sk)

# Total information entropy
O_S=O_Sr+O_Sk
Stot_vals.append(O_S)
print("S= ", O_S)

# Density of States Plots
RHF.dos_plot(O_p, O_n, 'O')
print("\n")

data = {
    'I_R1s': O_int_R1s, 'I_R2s': O_int_R2s, 'I_R2p': O_int_R2p, 'Ip': O_Ip, 'Sr': O_Sr, 'I_K1s': O_int_K1s, 'I_K2s': O_int_K2s, 'I_K2p': O_int_K2p, 'In': O_In, 'Sk': O_Sk, 'S': O_S
    
}

RHF.create_element_excel('O', data)

#----- F (Z=9) -----#
print("Results for F (Z=9)")
Z=9 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=5 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
F_c_1s=[0.377498 ,0.443947 ,-0.000797 ,0.213846,0.002183 ,0.000335 ,0.000147]
F_c_2s=[-0.058489,0.426450,-0.000274,-0.063457,-0.358939,-0.516660,-0.239143]
F_c_2p=[0.004879 ,0.130794 ,.337876 ,0.396122 ,0.225374 ,0.0 ,0.0]
F_s_vals=[12.6074 ,7.4101 ,23.2475 ,10.7416 ,3.7543,2.5009,1.8577]
F_p_vals=[11.0134,4.9962,3.1540,1.9722,1.3632,1,1]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
F_R1s, F_R2s, F_R2p=RHF.wf_R3s(F_c_1s, F_c_2s, F_c_2p, F_s_vals, F_p_vals)

# Calculating integrals
F_int_R1s=RHF.r_sym_int(F_R1s**2*r**2)
print("I_R1s= ",F_int_R1s)
F_int_R2s=RHF.r_sym_int(F_R2s**2*r**2)
print("I_R2s= ",F_int_R2s)
F_int_R2p=RHF.r_sym_int(F_R2p**2*r**2)
print("I_R2p= ",F_int_R2p)

# Density of states
F_p=(1/(4*np.pi*Z))*(e_1s*F_R1s**2 +e_2s*F_R2s**2 +e_2p*F_R2p**2)

# Confirming normalization (integral should be equal to 1)
F_Ip=(RHF.r_sym_int(4*sp.pi*F_p*r**2))
print("Ip= ", F_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
F_Sr=(RHF.r_sym_int(-4*sp.pi*F_p*sp.log(F_p+eps)*r**2))
Sr_vals.append(F_Sr)
print("Sr entropy= ", F_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
F_K1s, F_K2s, F_K2p=RHF.wf_K3s(F_c_1s, F_c_2s, F_c_2p, F_s_vals, F_p_vals)

# Calculating integrals
F_int_K1s=RHF.k_sym_int(F_K1s**2*k**2)
print("I_K1s= ",F_int_K1s)
F_int_K2s=RHF.k_sym_int(F_K2s**2*k**2)
print("I_K2s= ",F_int_K2s)
F_int_K2p=RHF.k_sym_int(F_K2p**2*k**2)
print("I_K2p= ",F_int_K2p)
# Density of states
F_n=(1/(4*np.pi*Z))*(e_1s*F_K1s**2 +e_2s*F_K2s**2 +e_2p*F_K2p**2)

# Confirming normalization (integral should be equal to 1)
F_In=(RHF.k_sym_int(4*sp.pi*F_n*k**2))
print("In= ", F_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
F_Sk=(RHF.k_sym_int(-4*sp.pi*F_n*sp.log(F_n+eps)*k**2))
Sk_vals.append(F_Sk)
print("Sk entropy= ", F_Sk)

# Total information entropy
F_S=F_Sr+F_Sk
Stot_vals.append(F_S)
print("S= ", F_S)

# Density of States Plots
RHF.dos_plot(F_p, F_n, 'F')
print("\n")

data = {
    'I_R1s': F_int_R1s, 'I_R2s': F_int_R2s, 'I_R2p': F_int_R2p, 'Ip': F_Ip, 'Sr': F_Sr, 'I_K1s': F_int_K1s, 'I_K2s': F_int_K2s, 'I_K2p': F_int_K2p, 'In': F_In, 'Sk': F_Sk, 'S': F_S
    
}

RHF.create_element_excel('F', data)

#----- Ne (Z=10) -----#
print("Results for Ne (Z=10)")
Z=10 # (atomic number = number of electrons)
e_1s=2 # number of electrons on the 1s orbital
e_2s=2 # number of electrons on the 2s orbital
e_2p=6 # number of orbitals on the 2p orbital
# Coefficients as given by Bunge
Ne_c_1s=[0.392290,0.425817,-0.000702,0.217206,+0.002300,0.000463,0.000147]
Ne_c_2s=[-0.053023,0.419502,-0.000263,-0.055723,-0.349457,-0.523070,-0.246038]
Ne_c_2p=[0.004391,0.133955,0.342978,0.395742,0.221831,0.0,0.0]
Ne_s_vals=[13.9074,8.2187,26.0325,11.9249,4.2635,2.8357,2.0715]
Ne_p_vals=[12.3239,5.6525,3.5570,2.2056,1.4948,1,1]

# Calculations in position space

# Calling RHF function to calculate wavefunctions
Ne_R1s, Ne_R2s, Ne_R2p=RHF.wf_R3s(Ne_c_1s, Ne_c_2s, Ne_c_2p, Ne_s_vals, Ne_p_vals)

# Calculating integrals
Ne_int_R1s=RHF.r_sym_int(Ne_R1s**2*r**2)
print("I_R1s= ",Ne_int_R1s)
Ne_int_R2s=RHF.r_sym_int(Ne_R2s**2*r**2)
print("I_R2s= ",Ne_int_R2s)
Ne_int_R2p=RHF.r_sym_int(Ne_R2p**2*r**2)
print("I_R2p= ",Ne_int_R2p)

# Density of states
Ne_p=(1/(4*np.pi*Z))*(e_1s*Ne_R1s**2 +e_2s*Ne_R2s**2 +e_2p*Ne_R2p**2)

# Confirming normalization (integral should be equal to 1)
Ne_Ip=(RHF.r_sym_int(4*sp.pi*Ne_p*r**2))
print("Ip= ", Ne_Ip)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
Ne_Sr=(RHF.r_sym_int(-4*sp.pi*Ne_p*sp.log(Ne_p+eps)*r**2))
Sr_vals.append(Ne_Sr)
print("Sr entropy= ", Ne_Sr)

# Calculations in k-space

# Calling RHF function to calculate wavefunctions
Ne_K1s, Ne_K2s, Ne_K2p=RHF.wf_K3s(Ne_c_1s, Ne_c_2s, Ne_c_2p, Ne_s_vals, Ne_p_vals)

# Calculating integrals
Ne_int_K1s=RHF.k_sym_int(Ne_K1s**2*k**2)
print("I_K1s= ",Ne_int_K1s)
Ne_int_K2s=RHF.k_sym_int(Ne_K2s**2*k**2)
print("I_K2s= ",Ne_int_K2s)
Ne_int_K2p=RHF.k_sym_int(Ne_K2p**2*k**2)
print("I_K2p= ",Ne_int_K2p)
# Density of states
Ne_n=(1/(4*np.pi*Z))*(e_1s*Ne_K1s**2 +e_2s*Ne_K2s**2 +e_2p*Ne_K2p**2)

# Confirming normalization (integral should be equal to 1)
Ne_In=(RHF.k_sym_int(4*sp.pi*Ne_n*k**2))
print("In= ", Ne_In)

# Shanon information entropy in position space
eps=np.finfo(float).eps # machine epsilon used to avoid having 0 in log
Ne_Sk=(RHF.k_sym_int(-4*sp.pi*Ne_n*sp.log(Ne_n+eps)*k**2))
Sk_vals.append(Ne_Sk)
print("Sk entropy= ", Ne_Sk)

# Total information entropy
Ne_S=Ne_Sr+Ne_Sk
Stot_vals.append(Ne_S)
print("S= ", Ne_S)


# Density of States Plots
RHF.dos_plot(Ne_p, Ne_n, 'Ne')
print("\n")

data = {
    'I_R1s': Ne_int_R1s, 'I_R2s': Ne_int_R2s, 'I_R2p': Ne_int_R2p, 'Ip': Ne_Ip, 'Sr': Ne_Sr, 'I_K1s': Ne_int_K1s, 'I_K2s': Ne_int_K2s, 'I_K2p': Ne_int_K2p, 'In': Ne_In, 'Sk': Ne_Sk, 'S': Ne_S
    
}

RHF.create_element_excel('Ne', data)

# ----Entropy Results----

# List of information entropy values in position space (Sr)
print("Information entropy Sr values for Z=2 to Z=9: \n ")
for i in range(0,9):
    print("Sr(",i+2,")= ",Sr_vals[i])
print("\n")
    
# List of information entropy values in k-space (Sk)
print("Information entropy Sk values for Z=2 to Z=9: \n ")
for i in range(0,9):
    print("Sk(",i+2,")= ",Sk_vals[i])
print("\n")

# List of total information entropy values (S)
print("Total information entropy S values for Z=2 to Z=9: \n ")
for i in range(0,9):
    print("S(",i+2,")= ",Stot_vals[i])

Z_vals = list(range(2, 11))

# Excel table for S values 
data={'Z':Z_vals, 'S_r': Sr_vals, 'S_k': Sk_vals, 'S':Stot_vals}
df = pd.DataFrame(data)
excel_filename = 'entropy.xlsx'
df.to_excel(excel_filename, index=False)

# Plotting Sr, Sk, and S vs Z 
plt.figure(figsize=(12, 8))

plt.plot(Z_vals, Sr_vals, marker='o', linestyle='-', color='blue', label='Sr (Position Space)')
plt.plot(Z_vals, Sk_vals, marker='o', linestyle='-', color='red', label='Sk (k-Space)')
plt.plot(Z_vals, Stot_vals, marker='o', linestyle='-', color='purple', label='S (Total Entropy)')

plt.title('Information Entropy Sr, Sk, and S vs Z')
plt.xlabel('Atomic Number (Z)')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)

# Show plot
plt.show()





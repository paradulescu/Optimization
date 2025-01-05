## Run the Xfoil file with the specified string name
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

def Xfoil_wrapper(airfoil_seed,alpha_i=0,alpha_f=10,alpha_step=1,Re=10**6,n_iters=100):
    ##XFOIL input file writer 
    #runs XFOIL with the specified values
    #inputs:
    #Airfoil Seed: the dat or NACA string to load
    #alpha_i,alpha_f,alpha_step: The initial alpha, final alpha, and the step size for the AOA sweep
    #Re: Reynolds number
    #n_iters: Number of iterations for XFOIL to run to converge on solns higher better

    #output:
    #polar_data: Drag polar array set with colns that corrispond with
    #alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr

    #writes the latest drag polar to a txt file (removed at start of run)
    if os.path.exists("polar_file.txt"):
        os.remove("polar_file.txt")

    input_file = open("input_file.in", 'w')
    input_file.write("LOAD {0}\n".format(airfoil_seed))
    input_file.write(airfoil_seed + '\n')
    input_file.write("PANE\n")
    input_file.write("OPER\n")
    input_file.write("Visc {0}\n".format(Re))
    input_file.write("PACC\n")
    input_file.write("polar_file.txt\n\n")
    input_file.write("ITER {0}\n".format(n_iters))
    input_file.write("ASeq {0} {1} {2}\n".format(alpha_i, alpha_f,alpha_step))
    input_file.write("\n\n")
    input_file.write("quit\n")
    input_file.close()

    subprocess.call("xfoil.exe < input_file.in", shell=True) 

    polar_data = np.loadtxt("polar_file.txt", skiprows=12)

    return polar_data

def Xfoil_wrapper_Cl(airfoil_seed,Cl_target,Re=10**6,n_iters=500):
    ##XFOIL input file writer 
    #runs XFOIL with the specified values
    #inputs:
    #Airfoil Seed: the dat or NACA string to load
    #alpha_i,alpha_f,alpha_step: The initial alpha, final alpha, and the step size for the AOA sweep
    #Re: Reynolds number
    #n_iters: Number of iterations for XFOIL to run to converge on solns higher better

    #output:
    #polar_data: Drag polar array set with colns that corrispond with
    #alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr

    #writes the latest drag polar to a txt file (removed at start of run)
    if os.path.exists("polar_file.txt"):
        os.remove("polar_file.txt")

    input_file = open("input_file.in", 'w')
    input_file.write("LOAD {0}\n".format(airfoil_seed))
    input_file.write(airfoil_seed + '\n')
    input_file.write("PANE\n")
    input_file.write("OPER\n")
    input_file.write("Visc {0}\n".format(Re))
    input_file.write("PACC\n")
    input_file.write("polar_file.txt\n\n")
    input_file.write("ITER {0}\n".format(n_iters))
    input_file.write("C {0}\n".format(Cl_target))
    input_file.write("\n\n")
    input_file.write("quit\n")
    input_file.close()

    subprocess.call("xfoil.exe < input_file.in", shell=True) 

    polar_data = np.loadtxt("polar_file.txt", skiprows=12)

    return polar_data


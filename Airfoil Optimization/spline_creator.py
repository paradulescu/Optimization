##Function that creates a DAT files based off bspline definition
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep,splev
from scipy.interpolate import BSpline

def spline_to_DAT(knots,coeffs,degree,points=160):
    #input
    #knots: The knots of the spline as numpy arrays
    #coeffs: The coefficents of the spline knots
    #degrees: the degree if the Bspline
    #points: The number of points to be created, defult is 100

    #output:
    #creates a DAT file of the x y cordinates of the spline
    #spline: returns the spline function
    #x,y: returns the x and y arrays

    if os.path.exists("spline.dat"):
        os.remove("spline.dat")

    # Create the B-spline
    spline = BSpline(knots, coeffs, degree)

    # Generate x values
    x = np.linspace(knots[degree], knots[-degree-1], points)  # Generate 100 points within the valid range

    # Evaluate the B-spline for these x values
    y = spline(x)

    # Write x and y to a .dat file
    with open('spline.dat', 'w') as file:
        for xi, yi in zip(x, y):
            file.write(f"{xi}\t{yi}\n")
            
    print("Data has been written to spline.dat")

    return spline,x,y

def spline_controlPoints_based(control_points,num_pnts):

    # Parameters for the B-spline
    t = np.linspace(0, 1, len(control_points))

    # Fit a B-spline using the control points
    tck, u = splprep([control_points[:, 0], control_points[:, 1]], s=0, k=3)

    # Evaluate the B-spline at a set of points
    u_fine = np.linspace(0, 1, num_pnts)
    x_fine, y_fine = splev(u_fine, tck)

    return x_fine,y_fine

def create_bspline(x_coords, y_coords, num_points=300):
    
    control_points = np.column_stack((x_coords, y_coords))
    tck, u = splprep([control_points[:, 0], control_points[:, 1]], s=0, k=3)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)

    return x_fine, y_fine
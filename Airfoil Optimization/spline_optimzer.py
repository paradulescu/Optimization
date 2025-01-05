##Import functions libs
import os
import subprocess
from Xfoil_runner import Xfoil_wrapper, Xfoil_wrapper_Cl
from NACA_to_DAT import naca4
from spline_creator import spline_to_DAT,spline_controlPoints_based, create_bspline
from fit_bspline import bSpline_fit,bSpline_fit_cos

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import Bounds

# control_upper,control_lower=bSpline_fit("naca5512.dat",10)

# plot the airfoil!
# x1,y1=spline_controlPoints_based(control_upper,100)
# x2,y2=spline_controlPoints_based(control_lower,100)

# xs_temp=np.append(x1,x2)
# ys_temp=np.append(y1,y2)

##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##
#constants
control_points_numbers=8 #number of control points used to edit the airfoil
Cl=0.6 #Cl that is optimized for
y_max=0.2 #this is the max abs value that y can be
y_min=0.01 #this is the min abs value that y can be
tau=10**-5 #tol for scipy optimizer
naca_number="2412"
step = 5*10**-3
Re = 1*10**5

i = control_points_numbers
    # lowerBound = np.ones(i)*y_min # upper surface lb
    # lowerBound = np.concatenate((lowerBound, np.ones(i)*-y_max) ) # lower surfcae lower bound
t_high = 1.15
t_low = 0.85

#Create a bspline baseline from the optiml NACA airfoil with n number of control points
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##

#from the NACA_optimzer import what NACA airfoil to start with
x_coords, y_coords=naca4(naca_number)
#normalize
x_coords=x_coords/(max(x_coords))
y_coords=y_coords/(max(x_coords))

#Write to .dat file
if os.path.exists('naca.dat'):
        os.remove('naca.dat')

np.savetxt('naca.dat', np.column_stack((x_coords,y_coords)), delimiter=' ')
print(f"Data has been written to naca{naca_number}.dat")

##-------------------------------------------------------------------------##
#cosine spacing
##-------------------------------------------------------------------------##
# max you can do is 16 points for cos space
# control points for bspline some reason inveresed
# control_upper,control_lower=bSpline_fit("naca.dat",control_points_numbers)
control_upper,control_lower=bSpline_fit_cos("naca.dat",control_points_numbers)
control_points_seed=np.row_stack((control_lower,control_upper))
print("control points seed",control_points_seed)
# plot the airfoil!
# x1,y1=spline_controlPoints_based(control_upper,100)
# x2,y2=spline_controlPoints_based(control_lower,100)
xs,ys=create_bspline(control_points_seed[:,0], control_points_seed[:,1])

plt.plot(xs, ys)
plt.scatter(control_points_seed[:,0], control_points_seed[:,1])
plt.axis("equal")
plt.show()

# control points for bspline need to be inveresed

# xs=np.append(x2,x1)
# ys=np.append(y2,y1)

# control_upper,control_lower=bSpline_fit("naca.dat",control_points_numbers)
# x1,y1=spline_controlPoints_based(control_upper,100)
# x2,y2=spline_controlPoints_based(control_lower,100)

# control_points_seed=np.row_stack((control_upper,control_lower))

# xs=np.append(x1,x2)
# ys=np.append(y1,y2)


if os.path.exists("bspline_airfoil.dat"):
        os.remove("bspline_airfoil.dat")

# Save the B-spline coordinates to a .dat file
np.savetxt('bspline_airfoil.dat', np.column_stack((xs,ys)), delimiter=' ')
print("B-spline coordinates have been written to bspline_airfoil.dat")

# plt.scatter(xs, ys)
# plt.show()


xs_seed=xs
ys_seed=ys

def airfoil_optimzer(Cl,y_max,y_min,control_points_start):

    # optimization function
    control_points_x=control_points_start[:,0] #this is the horizontal spacing of the control points
    control_points_y0=control_points_start[:,1]

    def objective(control_points_y):

        x_fine, y_fine = create_bspline(control_points_x, control_points_y)

        if os.path.exists("bspline_airfoil.dat"):
            os.remove("bspline_airfoil.dat")

        np.savetxt('bspline_airfoil.dat', np.column_stack((x_fine,y_fine)), delimiter=' ')

        # Save Cd
        try:
            output = Xfoil_wrapper_Cl("bspline_airfoil.dat",Cl, Re,)
            # Cl_index=np.argmax(dragPolar[:,2] > Cl) #find C
            drag=output[2]

        except IndexError:
            print("XFOIL didn't converge")
            return 1
        
        print(f"Drag Coefficient: {drag}")

        if drag<=(10**-6):
             return 1

        return drag
    
    drag_0=objective(control_points_y0)
    
    # def constraint_y_min(control_points_y):
    #     #define that the absolute value of y for the non trailing and leading edges needs to be at least y_min
    #     return abs(control_points_y[1:len(control_points_y)-2])-y_min
    
    # def constraint_y_max(control_points_y):
    #     #define that the absolute value of y for the non trailing and leading edges needs to be at least y_min
    #     return abs(control_points_y[:])+y_max
    
    # con1 = lambda control_points_y: abs(control_points_y[1:len(control_points_y)-2])
    # con2 = lambda control_points_y: abs(control_points_y[:])
    #i = 16
    # lowerBound = np.ones(i)*y_min # upper surface lb
    # lowerBound = np.concatenate((lowerBound, np.ones(i)*-y_max) ) # lower surfcae lower bound
    #t_ratio = 0.9
    lowerBound = control_points_y0[0: i]*t_high # lower surface lb
    lowerBound = np.concatenate((lowerBound, control_points_y0[i:]*t_low)) # upper surfcae lower bound

    
    UpperBound = control_points_y0[0: i]*t_low # lower surface ub
    UpperBound = np.concatenate((UpperBound, control_points_y0[i:]*t_high)) #upper surface ub

    # b = [lowerBound, UpperBound]

    # bounds = Bounds(lowerBound, UpperBound, keep_feasible=True)

    bounds = Bounds(lowerBound, UpperBound, keep_feasible=True)

    # con1 = lambda control_points_y: control_points_y[0:control_points_numbers] # selects top syrface control points
    # con2 = lambda control_points_y: control_points_y[control_points_numbers:-1] # selects leading edge and bottom control points
    # con3 = lambda control_points_y: control_points_y[0] # selects the top trailing edge control points
    # con3 = lambda control_points_y: control_points_y[-1] # bottom top trailing edge control points

    # Define the constraint as a NonlinearConstraint
    # constraint1 = scipy.optimize.NonlinearConstraint(con1, y_min, y_max)
    # constraint2 = scipy.optimize.NonlinearConstraint(con2, -y_min, -y_max)
    # constraint3 = scipy.optimize.NonlinearConstraint(con3, y_min, .002)
    # constraint4 = scipy.optimize.NonlinearConstraint(con2, -y_min, -.002)

    # # constraint1 = scipy.optimize.NonlinearConstraint(con1, y_min, y_max)
    # # constraint2 = scipy.optimize.NonlinearConstraint(con2, 0, y_max)
    # # constraint2 = scipy.optimize.NonlinearConstraint(con3, 0, y_max)
    # constraint3 = scipy.optimize.NonlinearConstraint(con3, y_min, .002)
    # constraint3 = scipy.optimize.NonlinearConstraint(con2, -y_min, -.002)

    # constraint=[constraint1,constraint2, constraint3, constraint4]

    options={'maxiter': 10, 'eps': step}
    # methods='SLSQP'
    methods='None'

    # result = scipy.optimize.minimize(objective,control_points_y0,method='SLSQP',constraints=constraint, options=options)
    # result = scipy.optimize.minimize(objective,control_points_y0,method='L-BFGS-B',bounds=bounds, options=options)
    result = scipy.optimize.minimize(objective,control_points_y0,method='L-BFGS-B', tol = tau, bounds=bounds, options=options)

    return result,drag_0

# output=Xfoil_wrapper_Cl("bspline_airfoil.dat",Cl)

result,drag_0=airfoil_optimzer(Cl,y_max,y_min,control_points_seed)

print("initial drag",drag_0,"\n")
print("scipy result",result)

xs, ys = create_bspline(control_points_seed[:,0],result.x)

#run Xfoil sweep
# output=Xfoil_wrapper_Cl("bspline_airfoil.dat",Cl,Re=10**6,n_iters=100)
#find Cd where the CL constraint is met
#find where the Cl=0.4
# drag=output[2]
# print("The drag of the airfoil is: ", drag)

#plot the airfoil
fig, axs = plt.subplots()
#plt.plot(x_coords, y_coords, '--', label='Naca Airfoil')
plt.plot(xs_seed, ys_seed, '--', label='starting airfoil')
plt.plot(xs,ys,label='optimized airfoil')
plt.scatter(control_points_seed[:,0],result.x)
plt.title("B-Spline Fitting")
axs.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print("control points intial y",control_points_seed[:,1])
print("control points optimal y",result.x)
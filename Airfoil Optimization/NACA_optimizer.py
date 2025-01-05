"""
An example to show how to combine an airfoil_generator and XFOIL.
"""

# from naca4series import NACA4
import os
import subprocess
from Xfoil_runner import Xfoil_wrapper, Xfoil_wrapper_Cl

import matplotlib.pyplot as plt
import numpy as np

# Operating point
Re = 10**6
Cl = .4
t=12

ms=[0,1,2,3,4,5]
ps=[0,1,2,3,4,5]

drags = np.zeros((len(ms), len(ps)))

# m is camber in percent, p is position of max. camber in tenths
for m in ms:
    for p in ps:

        #Define the NACA Airfoil to use
        airfoil="NACA{0}{1}{2}".format(m,p,t)

        # Let XFOIL do its thing
        polar = Xfoil_wrapper_Cl(airfoil,Cl)

        # #find where the Cl=0.4
        # dra=np.argmax(polar[:,2] > Cl) #find C

        # Save Cd
        try:
            drags[m-ms[0]][p-ps[0]] = polar[3]
        except IndexError:
            raise Warning("XFOIL didn't converge on NACA{}{}12 at Cl={}."
                          .format(m,p,Cl))

#best airfoil finder
flat_index = np.argmin(drags)
row, col = np.unravel_index(flat_index, drags.shape)
print("NACA{0}{1}{2}".format(ms[row],ps[col],t),"is the lowest drag airfoil for starting design")

# Plot drag values in color
plt.pcolor(drags, cmap=plt.cm.coolwarm)

# Make plot pretty
plt.title(r"$C_d$ of $NACAmp15$ at $C_l={}$ and $Re={:g}$".format(Cl, Re))
plt.xlabel("Location of max. camber $p$")
plt.ylabel("Max. camber $m$")
cbar = plt.colorbar()
cbar.ax.set_ylabel("Drag coefficient $C_d$")
# plt.tight_layout()

# Show our artwork
# plt.show()

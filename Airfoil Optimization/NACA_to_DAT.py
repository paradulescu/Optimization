import numpy as np
import matplotlib.pyplot as plt

def naca4(number, n_points=250):
    #input
    #number: String of the NACA number

    #output:
    #creates a DAT file of the x y cordinates of the NACA file
    #returns the x and y cordinates as arrays
    
    m = int(number[0]) / 100.0  # Maximum camber
    p = int(number[1]) / 10.0   # Location of maximum camber
    t = int(number[2:4]) / 100.0  # Maximum thickness

    def thickness(x):
        return (t / 0.2) * (
            0.2969 * np.sqrt(x) -
            0.1260 * x -
            0.3516 * x**2 +
            0.2843 * x**3 -
            0.1015 * x**4
        )

    def camber_line(x):
        if x < p:
            return (m / (p**2)) * (2 * p * x - x**2)
        else:
            return (m / ((1 - p)**2)) * ((1 - 2 * p) + 2 * p * x - x**2)

    x = np.linspace(0, 1, n_points)
    yt = thickness(x)
    yc = np.array([camber_line(xi) for xi in x])

    dyc_dx = np.array([(m / (p**2)) * (2 * p - 2 * xi) if xi < p else
                       (m / ((1 - p)**2)) * (2 * p - 2 * xi) for xi in x])
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    return x_coords, y_coords

# naca_number = "2412"
# x_coords, y_coords = naca4(naca_number)

# # Write to .dat file
# with open(f"naca{naca_number}.dat", "w") as file:
#     for x, y in zip(x_coords, y_coords):
#         file.write(f"{x:.6f}\t{y:.6f}\n")

# print(f"Data has been written to naca{naca_number}.dat")

# # Optional: Plot the airfoil to visualize it
# plt.plot(x_coords, y_coords, '-o', label=f'NACA {naca_number}')
# plt.title(f"NACA {naca_number} Airfoil")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('equal')
# plt.legend()
# plt.show()
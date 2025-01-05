import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

def bSpline_fit(data,num_points):
    #input:
    #data: dat file that has the bspline be fit to
    #num_pts: The number of b spline control points

    #output:
    #give the control points at a specified spacing of a DAT file

    # Read the data from the .dat file
    data = np.loadtxt(data)

    # Extract x and y coordinates
    x_data = data[:, 0]

    index_middle=np.where(x_data == 0)[0]
    index_middle=index_middle[0]

    x_upper=data[:index_middle+1,0]
    y_upper=data[:index_middle+1,1]

    x_lower=data[index_middle+1:,0]
    y_lower=data[index_middle+1:,1]
    
    #upper first
    dx=len(x_upper)/num_points
    control_upper=np.zeros((num_points+1,2))
    for i in range(num_points):
        control_upper[i,:]=np.array([x_upper[int(i*dx)],y_upper[int(i*dx)]])

    dx=len(x_lower)/num_points
    control_lower=np.zeros((num_points,2))
    for i in range(num_points):
        control_lower[i,:]=np.array([x_lower[int(i*dx)],y_lower[int(i*dx)]])

    control_lower[num_points-1,:]=np.array([x_lower[len(x_upper)-2],y_lower[len(x_upper)-2]])

    return control_upper, control_lower

def bSpline_fit_cos(data,num_points):
    #cosine space the control points to be more clustured at the leading and trailing edge
    #input:
    #data: dat file that has the bspline be fit to
    #num_pts: The number of b spline control points

    #output:
    #give the control points at a specified spacing of a DAT file

    # Read the data from the .dat file
    data = np.loadtxt(data)

    # Extract x and y coordinates
    x_data = data[:, 0]

    index_middle=np.where(x_data == 0)[0]
    index_middle=index_middle[0]

    x_upper=data[:index_middle+1,0]
    y_upper=data[:index_middle+1,1]

    x_lower=data[index_middle+1:,0]
    y_lower=data[index_middle+1:,1]

    def cosine_spaced_indices(n, length):

        # Generate n points in the range [0, π]
        theta = np.linspace(0, np.pi, n)
        
        # Use cosine to map [0, π] to [1, -1]
        cos_values = np.cos(theta)
        
        # Normalize to the desired range [0, 1]
        cos_spaced = (1 + cos_values) / 2
        
        # Map to indices of the array (ranging from 0 to length-1)
        indices = (cos_spaced * (length - 1)).astype(int)
    
        return indices
      
    #upper first
    index=cosine_spaced_indices(num_points,len(x_upper))
    control_upper=np.zeros((num_points,2))

    k=0
    for i in index:
        
        control_upper[k,:]=np.array([x_upper[int(i)],y_upper[int(i)]])
        k+=1
    # control_upper[num_points-1,:]=np.array([x_upper[0],y_upper[0]])
    #control_upper[num_points-1,:]=np.array([x_upper[len(x_upper)-1],y_upper[len(x_upper)-1]])

    index=cosine_spaced_indices(num_points,len(x_lower))
    control_lower=np.zeros((num_points,2)) 
    k=0
    for i in index:
        control_lower[k,:]=np.array([x_lower[int(i)],y_lower[int(i)]])
        k+=1

    control_lower[0,:]=np.array([x_lower[len(x_upper)-2],y_lower[len(x_upper)-2]])

    # dx=len(x_lower)/num_points
    # control_lower=np.zeros((num_points+1,2))
    # for i in range(num_points):
    #     control_lower[i,:]=np.array([x_lower[int(i*dx)],y_lower[int(i*dx)]])

    # control_lower[num_points,:]=np.array([x_lower[len(x_upper)-1],y_lower[len(x_upper)-1]])

    return control_upper,control_lower
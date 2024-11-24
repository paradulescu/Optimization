import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# # Ensure Jax runs smoothly
# import jax
# jax.config.update("jax_enable_x64",True)

def SimplexCreate(x0,l):
    #Create a simplex
    #Input:
    #x0,l: starting x0 and the simplex length size

    #Output: 
    #x: Simples vector

    n=x0.shape[0] #number of dimensions
    x=x0

    for j in range(1,n+1):
    #construct s vector the same size as x
        sj=np.zeros((n,1))

        for i in range(n):
            if j==i:
                sj[i,0]=l/(n*np.sqrt(2))*(np.sqrt(n+1)-1)+l/np.sqrt(2)
            if j!=i:
                sj[i,0]=l/(n*np.sqrt(2))*(np.sqrt(n+1)-1)

        x=np.column_stack((x,x0+sj))

    return x

def SimplexSize(x,n):
    #find the size of the simplex (EX 7.6)
    delx=0
    for i in range(0,n-1):
        delx+=np.linalg.norm((x[:,i]-x[:,n]),ord=1)
    return delx

def std_Func(x,n,func):
    #Find the standard devation of the function value (EX 7.7)
    fbar=0
    for i in range(n):
        fbar=func(x[:,i])

    fbar=fbar/(n+1)

    innerthing=0
    for i in range(n):
        innerthing+=(func(x[:,i])-fbar)**2

    delf=np.sqrt(innerthing/(n+1))

    return delf

def sort_func_pts(x,n,func):
    #Sort the array of x to be in lowest->highest vector of function vals
    #Create list of function evals
    fs=np.zeros(n+1)

    for j in range(n+1):
        fs[j]=func(x[:,j])

    for i in range(n+1):
        # Track if any swap was made in this pass
        swapped = False
        for j in range(0, n - i):

            if fs[j] > fs[j + 1]:
                # Swap if elements are in the wrong order
                fj=np.copy(fs[j])
                fs[j]= fs[j + 1]
                fs[j + 1] = fj

                x_valj=np.copy(x[:,j])
                x[:,j]= x[:,j+1]
                x[:,j+1]=x_valj

                swapped = True
        # If no swaps were made, the list is already sorted
        if not swapped:
            break

    return x,fs

def NelderMead(x_start,func,tolx=10**-6,tolf=10**-6,tol_funs=10**-6,maxIters=10000,l=1):
    #Input:
    #x0: Starting point
    #tolx: Simplex size tolerance
    #tolf: Function value standard deviation tolerances
    #l: Length of the simplex defult of 1

    #Output:
    #x*: Optimal Point

    #i is the index of the dimensions
    #j is the 

    n=x_start.size #dimensions of the problem
    #create starting Simplex
    x=SimplexCreate(x_start,l)

    x_history=[x_start]

    delx=SimplexSize(x,n)
    delf=std_Func(x,n,func)

    k=0
    func_calls=0

    while delx>tolx or delf>tolf:
        
        x,f=sort_func_pts(x,n,func) #get function evals of the simplex points and sort
        func_calls+=(n+1)

        if abs(f[0]-f[n-1])<tol_funs:
            # print("iteration convereged",k)
            return x0, x_history, func_calls

        xc=np.zeros((n,1)) #Centroid excludding the worst point
        for i in range (n-1):
            xi=np.array(x[:,i])
            xi=xi.reshape((n,1))
            xc=np.add(xc,xi)
        xc=1/n*xc

        xn=np.array(x[:,n])
        xn=xn.reshape((n,1))

        xr=xc+(xc-xn) #reflect #alpha=1
        xr=xr.reshape((n,1))

        if func(xr)<func(x[:,0]):
            xe=xc+2*(xc-xn) #expansion
            xe=xe.reshape((n,1))
            if func(xe)<func(x[:,0]):
                x[:,n]=np.copy(xe[:,0]) #accept expansion and replace worst point
            else:
                x[:,n]=np.copy(xr[:,0]) #accept just reflection
        elif func(xr)<=func(x[:,n-1]): #reflection point better than 2nd worse?
            x[:,n]=xr[:,0] #accept reflection
        else:
            if func(xr)>func(x[:,n]): #reflection worse than worse point?

                xic=xc-0.5*(xc-xn) #inside contraction
                xic=xic.reshape((n,1))

                if func(xic)<func(x[:,n]): #inside contraction better than worst?
                    x[:,n]=np.copy(xic[:,0]) #accept inside contraction
                else:
                    for j in range(1,n):
                        x[:,j]=x[:,0]+0.5*(x[:,j]-x[:,0]) #shrink gamma =0.5
            else:
                xoc=xc+0.5*(xc-xn) #outside contraction alpha
                xoc=xoc.reshape((n,1))
                if func(xoc)<func(xr): #outside contraction better than reflection?
                    x[:,n]=np.copy(xoc[:,0]) #accept outside contraction 
                else:
                    for j in range(1,n):
                        x[:,j]=x[:,0]+0.5*(x[:,j]-x[:,0]) #shrink with gamma=0.5

        x0=np.copy(x[:,0])
        x0=x0.reshape((n,1))
        x_history+=[x0]
        k+=1

        delx=SimplexSize(x,n)
        delf=std_Func(x,n,func)

        # if k%100==0:
        #     print("iteration",k)
        #     print("x0",x0)
        #     print("convergence",abs(f[0]-f[n-1]))

        if k>maxIters:
            # print("x",x)
            # print("final f",f)
            # print("iteration max",k)
            return x0,x_history, func_calls
    
    return x0,x_history,func_calls


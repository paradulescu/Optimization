import matplotlib.pyplot as plt
import jax
import scipy.optimize
import sympy as sp
import numpy as np
# Ensure Jax runs smoothly
jax.config.update("jax_enable_x64",True)

# import optimizer
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#Simple Backtrack Linesearch
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

def Backtrack_linesearch(func,x,p,a0,mu1,rho,mu):
    # input
    # a0>0: Initial step length
    # 0<mu1<1: sufficent decrease factor
    # 0<rho<1 Backtracking factor
    # x: Current x value
    # p: search direction vector
    # f: function that is being linesearched
    # f1: The derivative of the function
    #mu: used for penalty

    #outputs:
    #a: step size satisfying sufficent decrease condition
    #history: The history of the optimizer

    a=a0
    psi_a,psi1_a=psi_fnc(func,x,a,p,mu)
    psi0,psi1_0=psi_fnc(func,x,0,p,mu)
    fnc_iters=2
    #check to ensure slope is negative
    if (psi1_0>0):
        print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1_0)
        return
    
    history=[]
    while (psi_a> psi0+mu1*a*psi1_0):
           a=rho*a
           psi_a,dump=psi_fnc(func,x,a,p,mu)
           fnc_iters+=1
           history=history+[a]
    return a, history, fnc_iters
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#Strong Wolfe Pinpoint and Bracket Linesearch
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def Pinpoint(func,mu,x,p,alow,ahigh,psi0,psi1_0,psi_low,psi_high,psi1_low,psi1_high,mu1,mu2):
    # input
    # alow: interval endpoint with lower func value
    # ahigh: interval endpoint with higher func value
    # psi0,psi1_0,psi_low,psi_high: computed in outer routine
    # psi1_low,psi1_high: derivative of psi
    # 0<mu1<1: sufficent Decrease factor
    # mu1<mu1<1: Sufficent Curvature Factor

    #Outputs
    #a_star: Acceptable step size
    #history: hsitory of iterations

    ##THIS OPTIMIZER USES QUADRATIC INTERPOLATION

    k=0
    history=[]
    flag=True
    function_iters=0
    while (flag==True):

        ap=interpolation_quadratic(alow,ahigh,psi_low,psi_high,psi1_low)
        
        psi_p,dump=psi_fnc(func,x,ap,p,mu)
        function_iters+=1
        if (psi_p > psi0+mu1*ap*psi1_0) or (psi_p>psi_low):
            #update the uper limit to the current interpolated value
            ahigh=ap
            psi_high,psi1_high=psi_fnc(func,x,ahigh,p,mu)
            function_iters+=1
            
            history+=[ap]
        
        else:
            dump,psi1_p=psi_fnc(func,x,ap,p,mu)
            function_iters+=1
            if (abs(psi1_p)<=(-1*mu2*psi1_0)):
                a_star=ap
                history+=[ap]
                return a_star,history, function_iters
            elif(psi1_p*(ahigh-alow)>=0):
                ahigh=alow

            if abs(alow-ahigh)<=10**-9:
                #the system converged
                a_star=ap
                history+=[ap]
                return a_star,history, function_iters
            alow=ap
            psi_high,psi1_high=psi_fnc(func,x,ahigh,p,mu)
            psi_low,psi1_low=psi_fnc(func,x,alow,p,mu)
            history+=[ap]
            function_iters+=2
        k=k+1

def Bracketing(func,mu,x,p,a0,mu1,mu2,sigma):
    # input
    # a0>0: Initial step size
    # psi0,psi1_0: computed in outer routine passed to save function call
    # mu2<mu1<1: Sufficent Curvature Factor
    # sigma>1: step size increase factor
    #mu: used in penalty

    #Outputs
    #a_star: Acceptable step size
    #history: History of a*

    a1=0
    a2=a0
    psi0,psi1_0=psi_fnc(func,x,0,p,mu)
    psi1,psi1_1=psi_fnc(func,x,a1,p,mu)
    function_iters=2
    first=True
    flag=True
    Bracket_history=[]

    while (flag is True):
        psi2,psi1_2=psi_fnc(func,x,a2,p,mu)
        function_iters+=1
        Bracket_history+=[a1,a2]

        if (psi2>psi0+mu1*a2*psi1_0) or ((first is not True) and (psi2>psi1)):
            #1=low,2=high

            a_star,history_pin,fnc_iters_pnpnt=Pinpoint(func,mu,x,p,a1,a2,psi0,psi1_0,psi1,psi2,psi1_1,psi1_2,mu1,mu2)
            function_iters+=fnc_iters_pnpnt
            Bracket_history+=[a1,a2]
            return a_star,history_pin,Bracket_history, function_iters
        
        elif (abs(psi1_2)<=-1*mu2*psi1_0): #step acceptable exit line search
            a_star=a2
            return a_star,[a2],Bracket_history, function_iters

        elif(psi1_2>=0):
            #2=low,1=high
            a_star,history_pin,fnc_iters_pnpnt=Pinpoint(func,mu,x,p,a2,a1,psi0,psi1_0,psi2,psi1,psi1_2,psi1_1,mu1,mu2)
            function_iters+=fnc_iters_pnpnt
            Bracket_history+=[a2,a1]
            return a_star,history_pin,Bracket_history,function_iters
        
        else:
            #increase the step
            a1=a2
            a2=sigma*a2
            Bracket_history+=[a1,a2]
        
        #set the first loophistory_pin to be false
        first=False

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#Interpolation Functions
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def interpolation_cubic(a1,a2,psi_1,psi_2,psi1_1,psi1_2):
    # input
    # a1: The lower step
    # a2: The higher step
    # psi_1: Function derivative at a1
    # psi_1: Function derivative at a1
    # psi1_1: Function derivative at a1
    # psi1_2: Function derivative at a2

    #outputs:
    # a_star: Interpolated step value

    b1=psi1_1+psi1_2-3*((psi_1-psi_2)/(a1-a2))
    b2=np.sign(a2-a1)*np.sqrt(b1**2-psi1_1*psi1_2)
    a_star=a2-(a2-a1)*( (psi1_2+b2-b1)/(psi1_2-psi1_1+2*b2) )
    return a_star

def interpolation_quadratic(a1,a2,psi1,psi2,psi1_1):
    # input
    # a1: The lower step
    # a2: The higher step
    # psi_1: Function derivative at a1
    # psi_1: Function derivative at a1
    # psi1_1: Function derivative at a1
    # psi1_2: Function derivative at a2

    #outputs:
    # a_star: Interpolated step value
    
    upper=2*a1*(psi2-psi1)+psi1_1*(a1**2-a2**2)
    lower=2*(psi2-psi1+psi1_1*(a1-a2))

    return upper/lower
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#Psi functions
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#used to compute psi from the actual function f
def psi_fnc(f,xk,a,pk,mu):
    # input
    # a: Step Size
    # xk: Current x value
    # pk: Current search direction vector
    # f: function that is being linesearched
    #mu: used in penelty

    #outputs:
    # psi(a): The psi value at the current step
    # psi_prime(a): psi derivative at current step
    psi=f(xk+a*pk,mu)[0]
    psi1=np.array(f(xk+a*pk,mu)[1:]).dot(pk)[0]

    return psi,psi1

def Steepest_Descent(x0,tol,func,mu,mu1=10**-5,mu2=0.7,sigma=2):
    #Simple function that choices line search based on steepest descent of gradient
    #input:
    #x0: Starting Point
    #tol: Tolerance
    #f: The function
    #f1: The function gradient
    #mu1, mu2, sigma: Used In linesearch set to defult values unless specified
    #flag_method: checks if it is interior or exterior penalty method

    #Output:
    #x_star: Min Point
    #f_star: Min Function eval
    #Historu: History of Optimizer
    History={}
    k=0
    x_history=[x0]
    a0=0.1
    x=x0
    x_prev=x0
    fx,f1x=func(x,mu)
    f1x=np.array(f1x); fx=np.array(fx)
    fx_old,f1x_old=func(x_prev,mu)
    f1x_old=np.array(f1x_old); fx=np.array(fx_old)
    pk_old=-1*(f1x/np.linalg.norm(f1x))
    pk_old=np.array(pk_old)
    ak_old=a0
    function_iters=2
    convg_history=[np.max(np.abs(f1x))]

    while (np.max(np.abs(f1x))>=tol):
        
        #steepest descent direction
        pk=-1*f1x/np.linalg.norm(f1x)
        pk=np.array(pk)
        # estimate intial step size
        anit=ak_old*np.dot(f1x_old,pk_old)/(np.dot(f1x,pk))
        #linesearch algorithm 
        psi,psi1=psi_fnc(func,x,anit,pk,mu)
        function_iters+=1
        if (psi1>0):
            # print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1)
            while(psi1>0):
                anit=anit*.9
                psi,psi1=psi_fnc(func,x,anit,pk,mu)
                function_iters+=1
        #ak,history_pin,Bracket_history,fnc_iters=Bracketing(func,mu,x,pk,anit,mu1,mu2,sigma)
        ak, history, fnc_iters=Backtrack_linesearch(func,x,pk,a0,mu1,rho,mu)
        function_iters+=fnc_iters
        #update all the variables
        x_prev=x
        x=x+ak*pk
        pk_old=pk
        x_history+=[x]
        f1x_old=f1x 
        fx_old=fx
        fx,f1x=func(x,mu)
        f1x=np.array(f1x); fx=np.array(fx)
        ak_old=ak
        k+=1
        function_iters+=1
        convg_history+=[np.max(np.abs(f1x))]
        # print("iterations",k)
        # print("x",x)
        # print("f1",f1x)
        # print(np.max(np.abs(f1x)))
        
        if k>=100:
            History['x']=x_history; History['convergence']=convg_history; History['Function Calls']=function_iters
            History['iterations']=k
            return x, fx, History


    History['x']=x_history; History['convergence']=convg_history; History['Function Calls']=function_iters
    History['iterations']=k
    return x, fx, History

def NonLinear_Conjugate(x0,tol,func,mu,mu1=10**-5,mu2=0.7,sigma=2):
    #Simple function that choices line search based on steepest descent of gradient
    #input:
    #x0: Starting Point
    #tol: Tolerance
    #f: The function
    #f1: The function gradient
    #mu1, mu2, sigma: Used In linesearch set to defult values unless specified
    #mu:Used for penelty method

    #Output:
    #x_star: Min Point
    #f_star: Min Function eval
    #History: History of optimizer

    History = {}
    k=0
    x_history=[x0]
    a0=0.1
    x=x0
    x_prev=x0
    fx,f1x=func(x,mu)
    
    f1x=np.array(f1x); fx=np.array(fx)
    fx_old,f1x_old=func(x_prev,mu)
    f1x_old=np.array(f1x_old); fx=np.array(fx_old)
    function_iters=2
    # print("f1x",f1x)
    pk_old=-1*(f1x/np.linalg.norm(f1x))
    
    pk_old=np.array(pk_old)
    ak_old=a0

    convg_history=[np.max(np.abs(f1x))]
    # betak=1
    reset=False
    while (np.max(np.abs(f1x))>=tol):
        # print("f1",f1x)
        #calculate betak and see if it is zero
        betak=f1x.dot(f1x-f1x_old)/(f1x_old.dot(f1x_old))
        betak=max(0,betak)
        #reset if betak=0 or if 5 iterations have passed
        if betak==0 or k%5==0:
            reset=True
        else:
            reset=False
        #steepest descent direction if need to reset direction
        if reset==True or k==0:
            pk=-1*f1x/np.linalg.norm(f1x)
            pk=np.array(pk)

        #use non linear conjugate 
        if reset==False:
            pk=pk=-1*f1x/np.linalg.norm(f1x)+betak*pk_old
            
        # estimate intial step size
        anit=ak_old*np.dot(f1x_old,pk_old)/(np.dot(f1x,pk))

        #check if psi>0
        psi,psi1=psi_fnc(func,x,anit,pk,mu)
        function_iters+=1
        if (psi1>0):
            # print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1)
            while(psi1>0):
                anit=anit*.9
                reset=True
                pk=-1*f1x/np.linalg.norm(f1x)
                psi,psi1=psi_fnc(func,x,anit,pk,mu)
                function_iters+=1
        #linesearch algorithm 
        ak,history_pin,Bracket_history,fnc_iters_linesearch=Bracketing(func,mu,x,pk,anit,mu1,mu2,sigma)
        #ak, history, fnc_iters_linesearch=Backtrack_linesearch(func,x,p,a0,mu1,rho,mu)
        function_iters+=fnc_iters_linesearch
        #update all the variables
        x_prev=x
        x=x+ak*pk
        pk_old=pk
        x_history+=[x]
        f1x_old=f1x 
        fx_old=fx
        fx,f1x=func(x,mu)
        fx=np.array(fx);f1x=np.array(f1x)
        ak_old=ak
        k+=1
        function_iters+=1
        convg_history+=[np.max(np.abs(f1x))]
        #print("iterations optimzer",k)
        #print("xk",x)
    History['x']=x_history; History['convergence']=convg_history; History['Function Calls']=function_iters
    History['iterations']=k
    return x, fx, History

def Quasi_Newton(x0,tol,func,mu1=10**-5,mu2=0.7,sigma=2,threshold=5):
    #Simple function that choices line search based on steepest descent of gradient
    #input:
    #x0: Starting Point
    #tol: Tolerance
    #f: The function
    #f1: The function gradient
    #mu1, mu2, sigma: Used In linesearch set to defult values unless specified

    #Output:
    #x_star: Min Point
    #f_star: Min Function eval
    #f1_star: function gradient at eval
    #xss: History of iterations
    #function iters: Number of Function calls total

    History={}
    k=0
    x_history=[x0]
    #initial step size
    a0=1
    x=x0
    x_prev=x0
    fx,f1x=func(x)
    f1x=np.array(f1x); fx=np.array(fx)
    fx_old,f1x_old=func(x_prev)
    f1x_old=np.array(f1x_old); fx=np.array(fx_old)
    function_iters=2

    convg_history=[np.max(np.abs(f1x))]

    reset=False
    I=np.identity(len(f1x))

    while (np.max(np.abs(f1x))>=tol):
        #resets the Hessian back to zero
        if k==0 or reset==True:
            Vk=1/np.linalg.norm(f1x)*I
        else:
            s=x-x_prev
            y=f1x-f1x_old
            #THIS SIGMA IS DIFFERENT THAN THE INPUT FOR LINE SEARCH
            sigma_Newton=(s@np.transpose(y))

            s=np.array(s); sigma_Newton=np.array(sigma_Newton); y=np.array(y)
            Vk=(I-sigma_Newton*np.transpose(s)@y)*Vk_old*(I-sigma_Newton*np.transpose(y)@s)+sigma_Newton*np.transpose(s)@(s)
            
        #anit=ak_old*np.dot(f1x_old,pk_old)/(np.dot(f1x,pk))
        p=-Vk@f1x
        reset=False
        if abs(f1x@p)>threshold or (k%5==0 and k!=0):
            reset=True
            Vk=1/np.linalg.norm(f1x)*I
            p=-Vk@f1x
            a0=1
            # if k%10!=0:
            #     print("f1x@p",f1x@p,"reset",reset)
        #check if psi>0
        anit=a0
        psi,psi1=psi_fnc(func,x,anit,p)
        function_iters+=1
        if (psi1>0):
            # print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1)
            while(psi1>0):
                anit=anit*.9
                reset=True
                pk=-1*f1x/np.linalg.norm(f1x)
                psi,psi1=psi_fnc(func,x,anit,p)
                function_iters+=1
        #linesearch algorithm 
        ak,history_pin,Bracket_history,fnc_iters_bracket=Bracketing(func,x,p,anit,mu1,mu2,sigma)
        #backtrack not as good
        #ak,history_backtrack,fnc_iters_bracket=Backtrack_linesearch(func,x,p,anit,mu1,rho=.7)
        function_iters+=fnc_iters_bracket

        #update all the variables
        x_prev=x
        x=x+ak*p
        x_history+=[x]
        f1x_old=f1x 
        fx_old=fx
        fx,f1x=func(x)
        Vk_old=Vk
        k+=1
        function_iters+=1
        convg_history+=[np.max(np.abs(f1x))]

    History['x']=x_history; History['convergence']=convg_history; History['Function Calls']=function_iters
    History['iterations']=k
    return x, fx, History

def uncon_optimizer(func, x0, epsilon_g,mu, options=None):
    """An algorithm for unconstrained optimization.

    Parameters
    ----------
    func : function handle
        Function handle to a function of the form: f, g = func(x)
        where f is the function value and g is a numpy array containing
        the gradient. x are design variables only.
    x0 : ndarray
        Starting point
    epsilon_g : float
        Convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
    mu: float
        Penalty method 
    options : dict
        A dictionary containing options.  You can use this to try out different
        algorithm choices.  I will not pass anything in on autograder,
        so if the input is None you should setup some defaults.

    Returns
    -------
    xopt : ndarray
        The optimal solution
    fopt : float
        The corresponding optimal function value
    output : dictionary
        Other miscelaneous outputs that you might want, for example an array
        containing a convergence metric at each iteration.

        `output` must includes the alias, which will be used for mini-competition for extra credit.
        Do not use your real name or uniqname as an alias.
        This alias will be used to show the top-performing optimizers *anonymously*.
    """

    # TODO: set your alias for mini-competition here
    output = {}
    output['alias'] = 'Optimizer Never Heard of Her'

    if options is None:
        # TODO: set default options here.
        # You can pass any options from your subproblem runscripts, but the autograder will not pass any options.
        # Therefore, you should sse the  defaults here for how you want me to run it on the autograder.
        #
        #Set the auto to be one of the optimization formulas
        optimizer_function=Steepest_Descent

    if options == "NonLinear_Conjugate":
        optimizer_function=NonLinear_Conjugate
    if options == "Steepest_Descent":
        optimizer_function=Steepest_Descent
    if options == "Quasi_Newton":
        optimizer_function=Quasi_Newton

    x0=np.array(x0)
    xopt,fopt,history=optimizer_function(x0,epsilon_g,func,mu)
    output['history']=history

    # TODO: 

    return xopt, fopt, output

def Exterior_Penelty(x0,mu0,rho,func,epsilon_g):
    #Inputs
    #x0: Starting guess
    #mu0>0: Initial penalty guess
    #rho>0, backtrackiing penetly increase factor
    #1.2 conservative, 10 aggressive
    #func: function to be minimized
    #epsilon_g: tolerence for optimization

    #Outputs:
    #x_star: Optimal Point
    #f_star: Function value of optimal point

    k=0
    converge=False
    mu=mu0
    xk=x0
    history=[x0]
    first=True
    xk_old=x0
    while(converge==False):
        xk,fopt,output=uncon_optimizer(func, xk, epsilon_g,mu,'Steepest_Descent')
        mu=rho*mu
        k+=1
        history+=[xk]
        if (mu>=10**3 and first!=True):
            converge=True
            return xk,fopt, history,k
        # print("xk",xk)
        # print("Iteration",k)
        # print("mu",mu)
        xk_old=xk
        first=False
    return xk,fopt, history,k

def Interior_Penelty(x0,mu0,rho,func,epsilon_g):
    #Inputs
    #x0: Starting guess
    #mu0>0: Initial penalty guess
    #rho<0, backtrackiing penalty decrease factor
    #1.2 conservative, 10 aggressive
    #func: function to be minimized
    #epsilon_g: tolerence for optimization

    #Outputs:
    #x_star: Optimal Point
    #f_star: Function value of optimal point

    k=0
    converge=False
    mu=mu0
    xk=x0
    history=[x0]
    first=True
    xk_old=x0
    while(converge==False):
        # print("xk",xk)
        # print("Iteration",k)
        # print("mu",mu)
        xk,fopt,output=uncon_optimizer(func, xk, epsilon_g,mu,'Steepest_Descent')
        mu=rho*mu
        k+=1
        history+=[xk]
        if (mu<=0.1 and first!=True):
            converge=True
            return xk,fopt, history,k
        xk_old=xk
        first=False
    return xk,fopt, history,k

def Ex54_constraint(x):
    h=(1/4*x[0]**2+x[1]**2-1)
    return h

def def_function_Ex54(x):
    f=x[0]+2*x[1]
    return f

def Ex54_exterior(x,mu):
    h_fun=[0,(1/4*x[0]**2+x[1]**2-1)**2]
    f=x[0]+2*x[1]+mu/2*np.max(h_fun)
    
    return f

def Ex54_interior(x,mu):
    h_fun=-(1/4*x[0]**2+x[1]**2-1)
    f=x[0]+2*x[1]-mu*np.log(h_fun)
    
    return f

def interior_penalty_Ex54(x,mu):
    #exterior penetly function with hard coded mu value
    f=Ex54_interior(x,mu)
    f11=1
    f12=2
    h1_f1=-2*x[0]*mu/(x[0]**2+4*(x[1]**2-1))
    h2_f1=-8*x[1]*mu/(4*x[1]**2+x[0]**2-4)

    f1=[f11+h1_f1,f12+h2_f1]
    
    return f,f1

def exterior_penalty_quad_Ex54(x,mu):
    #exterior penetly function with hard coded mu value
    f=Ex54_exterior(x,mu)
    f11=1
    f12=2
    h1_f1=x[0]*(x[0]**2+4*(x[1]**2-1))/8*mu
    h2_f1=x[1]*(4*x[1]**2+x[0]**2-4)/2*mu
    f1=[f11+h1_f1,f12+h2_f1]
    
    return f,f1

def Optimzizer_Plotter(x0s,lvls,mu0,rho,func,func_graph,func_const,optimzer_func,tol=10**-2):
    #Function to run the optimizer and plot the output
    #Input:
    #x0s: Starting points
    #func: Funcetion that is being optimized
    #func_graph: Function to be graphed
    #tol: Tolerance of the optimizer 10^-6 defult
    #Optimizer alg: The optimer algorithm
    #filename: File name that the plot should be named

    #Output:
    #iterations: Number of iterations for the optimizer
    #Saves a file of the progression of the optimizer
    tests=len(x0s)
    x1s=np.linspace(-3,9)
    x2s=np.linspace(-5,5)
    X1,X2=np.meshgrid(x1s,x2s)
    Z=func_graph([X1,X2])
    plt.contour(X1,X2,Z,levels=lvls)
    #plt.contour(X1, X2, Z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
    # plt.axis('equal')
    plt.colorbar()
    plt.xlabel("x1")
    plt.ylabel("x2")
    colors=["b","g","c","y",'m']
    for i in range(tests):
        # print("iteration",i)
        x0=x0s[i]
        x_opt,f_opt,history,k=optimzer_func(x0,mu0,rho,func,tol)

        #used to plot history
        history_x1s=np.zeros(len(history))
        history_x2s=np.zeros(len(history))

        for j in range(len(history)):
            history_x1s[j]=history[j][0]; history_x2s[j]=history[j][1]

        plt.plot(history_x1s,history_x2s,marker='o',color=colors[i],label=str(i))
        plt.plot(x0[0],x0[1],color='k',marker='x')
        plt.plot(x_opt[0],x_opt[1],color='red',marker='o')
        
    # print(history[0])
    plt.plot(x0[0],x0[1],color='k',marker='x',label="Start")
    plt.plot(x_opt[0],x_opt[1],color='red',marker='o',label="Optimal")
    Z=func_const([X1,X2])
    plt.contourf(X1,X2,Z,colors='red',levels=[0,np.max(Z)],alpha=.3)
    plt.legend()
    
    # plt.savefig(filename)

#x0s=[[4.93006254, 1.011207 ],[ 0.92519986, -0.21909331],[ 9.07002026, -3.35493422],[0.56052395, 2.75820612],[2.13037829, 3.44901001]]

# mu0=0.5
# rho=1.2
# x0=[0,0]
# x_exterior,f_exterior,dump,dump=Exterior_Penelty(x0,mu0,rho,exterior_penalty_quad_Ex54,10**-2)

# mu0=3
# rho=0.8
# x0=[0,0]
# x_interior,f_interior,dump,dump=Interior_Penelty(x0,mu0,rho,interior_penalty_Ex54,10**-2)

# print("points Exterior",x_exterior,"and Interior",x_interior)
# print("Constraint values for Exterior",Ex54_constraint(x_exterior),"Interior",Ex54_constraint(x_interior))
# ##Plotting
# #Optimzizer_Plotter(x0s,25,mu0,rho,exterior_penalty_quad_Ex54,def_function_Ex54,Ex54_constraint,Exterior_Penelty,tol=10**-2)
# # x0s=[[0,0],[.25,.25],[.5,.5]]
# # # Interior_Penelty(x0,mu0,rho,interior_penalty_Ex54,10**-2)
# # Optimzizer_Plotter(x0s,25,mu0,rho,interior_penalty_Ex54,def_function_Ex54,Ex54_constraint,Interior_Penelty,tol=10**-2)
# # plt.axis([-3,3,-3,3])

# # plt.show()

#Prob 3b-----------------------------------------------------------------
#Constants
b=0.125
h=0.25
P=100.*1000
l=1.
sigma_yield=200.*10**6
tau=116.*10**6

def beam_jaxfun_exterior(x,mu):
    tb=x[0];tw=x[1]
    #Constants
    b=0.125
    h=0.25
    P=100.*1000
    l=1.
    sigma_yield=200.*10**6
    tau=116.*10**6
    I=(h**3)/12*tw+b/6*tb**3+(h**2)*b/2*tb
    g1=(P*l*h/2-sigma_yield*I)**2
    g2=(1.5*P/(h*tw)-tau)**2
    h=g1+g2
    f=2*b*tb+h*tw+mu/2*h

    return f

def beam_fun_exterior(x,mu):
    tb=x[0];tw=x[1]
    #Constants
    b=0.125
    h=0.25
    P=100.*1000
    l=1.
    sigma_yield=200.*10**6
    tau=116.*10**6
    I=(h**3)/12*tw+b/6*tb**3+(h**2)*b/2*tb
    g1=(P*l*h/2-sigma_yield*I)**2
    g2=(1.5*P/(h*tw)-tau)**2
    h=g1+g2
    f=2*b*tb+h*tw+np.max([0,h])
    f1=jax.grad(beam_jaxfun_exterior)

    return f, f1(x,mu)

def beam_jaxfun_interior(x,mu):
    tb=x[0];tw=x[1]
    #Constants
    b=0.125
    h=0.25
    P=100.*1000
    l=1.
    sigma_yield=200.*10**6
    tau=116.*10**6
    I=(h**3)/12*tw+b/6*tb**3+(h**2)*b/2*tb

    g1=(P*l*h/2-sigma_yield*I)
    g2=(1.5*P/(h*tw)-tau)
    f=2*b*tb+h*tw-mu*jax.numpy.log(-(g1+g2))

    return f

def beam_fun(x):
    tb=x[0];tw=x[1]
    #Constants
    #Constants
    b=0.125
    h=0.25
    P=100.*1000
    l=1.
    sigma_yield=200.*10**6
    tau=116.*10**6
    I=(h**3)/12*tw+b/6*tb**3+(h**2)*b/2*tb
    f=2*b*tb+h*tw

    return f

def beam_fun_interior(x,mu):
    tb=x[0];tw=x[1]
    #Constants
    #Constants
    b=0.125
    h=0.25
    P=100.*1000
    l=1.
    sigma_yield=200.*10**6
    tau=116.*10**6
    I=(h**3)/12*tw+b/6*tb**3+(h**2)*b/2*tb

    g1=(P*l*h/2-sigma_yield*I)
    g2=(1.5*P/(h*tw)-tau)
    # tb_max=10**6
    # tw_max=.5*tb_max
    # I_max=(h**3)/12*tw_max+b/6*tb_max**3+(h**2)*b/2*tb_max
    g1_max=1
    g2_max=g1_max
    f=2*b*tb+h*tw-mu*np.log(-(g1/g1_max+g2/g2_max))
    # f=beam_jaxfun_interior(x,mu)
    # f1=jax.grad(beam_jaxfun_interior)

    f1=[-116260416.666667*mu/(4166666.66666667*tb**3 + 781250.0*tb + 116260416.666667*tw - 612500.0) + 0.25,-mu*(12500000.0*tb**2 + 781250.0)/(4166666.66666667*tb**3 + 781250.0*tb + 116260416.666667*tw - 612500.0) + 0.25]

    return f, f1

mu0=0.5
rho=1.2
x0=[0.,0.]
x_exterior,f_exterior,history,k=Exterior_Penelty(x0,mu0,rho,beam_fun_exterior,10**-2)
print("Done with exterior")
print("Exterior value f(x*)=",beam_fun(x_exterior))

mu0=4
rho=0.6
x0=[.08,.08]
x_interior,f_interior,history,k=Interior_Penelty(x0,mu0,rho,beam_fun_interior,10**-2)
print("Interior value f(x*)=",beam_fun(x_interior))


tws=np.linspace(-.1,.1)
tbs=np.linspace(-.1,.1)
[TWS,TBS]=np.meshgrid(tws,tbs)
F=2*b*TBS+h*TWS
I_contour=(h**3)/12*TWS+b/6*TBS**3+(h**2)*b/2*TBS
G1=(P*l*h/2-sigma_yield*I_contour)
G2=(1.5*P/(h)-tau*TWS)
#constriant

plt.figure(0)
f_plt=plt.contour(TWS,TBS,F,levels=25,colors='k',alpha=0.5)
plt.clabel(f_plt,f_plt.levels[::15],inline=True,fmt='Function')
g1_plt=plt.contourf(TWS,TBS,G1,levels=[0,np.max(G1)],colors='m',alpha=.3)
g2_plt=plt.contourf(TWS,TBS,G2,levels=[0,np.max(G2)],colors='blue',alpha=.3)
plt.plot(x_interior[0],x_interior[1],'o',color='green')
plt.text(x_interior[0]-0.1,x_interior[1]-0.01, 'Interior Method Optimium, f(x*)='+str(0.0400),color='green',fontsize=10)

plt.plot(x_exterior[0],x_exterior[1],'o',color='black')
plt.text(x_exterior[0]-.05,x_exterior[1]-.01, 'Exterior Method Optimium, f(x*)='+str(0.0),color='black',fontsize=10)

plt.text(-.05, .07, 'g2',color='blue',fontsize=15)
plt.text(.05, -.05, 'g1',color='m',fontsize=15)

plt.ylabel('tb (m)')
plt.xlabel('tw (m)')

plt.show()
# plt.savefig('problem2.svg')
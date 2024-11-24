"""
This is a template for Assignment 3: unconstrained optimization

You can (and should) call other functions or import functions from other files,
but make sure you do not change the function signature (i.e., function name `uncon_optimizer`, inputs, and outputs) in this file.
The autograder will import `uncon_optimizer` from this file. If you change the function signature, the autograder will fail.
"""
import numpy as np
import matplotlib.pyplot as plt
import jax
import numpy as np
import scipy.optimize

# Ensure Jax runs smoothly
jax.config.update("jax_enable_x64",True)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#Simple Backtrack Linesearch
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def Backtrack_linesearch(func,x,p,a0,mu1=10**-5,rho=0.5):
    # input
    # a0>0: Initial step length
    # 0<mu1<1: sufficent decrease factor
    # 0<rho<1 Backtracking factor
    # x: Current x value
    # p: search direction vector
    # f: function that is being linesearched
    # f1: The derivative of the function

    #outputs:
    #a: step size satisfying sufficent decrease condition
    #history: The history of the optimizer

    a=a0
    psi_a,psi1_a=psi_fnc(func,x,a,p)
    psi0,psi1_0=psi_fnc(func,x,0,p)
    fnc_iters=2
    #check to ensure slope is negative
    if (psi1_0>0):
        print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1_0)
        return
    
    history=[]
    while (psi_a> psi0+mu1*a*psi1_0):
           a=rho*a
           psi_a,dump=psi_fnc(func,x,a,p)
           fnc_iters+=1
           history=history+[a]
    return a, history, fnc_iters
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#Strong Wolfe Pinpoint and Bracket Linesearch
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def Pinpoint(func,x,p,alow,ahigh,psi0,psi1_0,psi_low,psi_high,psi1_low,psi1_high,mu1,mu2):
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

        ap=interpolation_cubic(alow,ahigh,psi_low,psi_high,psi1_low,psi1_high)
        
        psi_p,dump=psi_fnc(func,x,ap,p)
        function_iters+=1
        if (psi_p > psi0+mu1*ap*psi1_0) or (psi_p>psi_low):
            #update the uper limit to the current interpolated value
            ahigh=ap
            psi_high,psi1_high=psi_fnc(func,x,ahigh,p)
            function_iters+=1
            # psi_high,psi1_high=psi_fnc(func,x,ap,p)
            history+=[ap]
        
        else:
            dump,psi1_p=psi_fnc(func,x,ap,p)
            function_iters+=1
            if (abs(psi1_p)<=(-1*mu2*psi1_0)):
                a_star=ap
                history+=[ap]
                return a_star,history, function_iters
            elif(psi1_p*(ahigh-alow)>=0):
                ahigh=alow
            if abs(alow-ahigh)<=10**-5:
                #the system converged
                a_star=ap
                history+=[ap]
                return a_star,history, function_iters
            alow=ap
            psi_high,psi1_high=psi_fnc(func,x,ahigh,p)
            psi_low,psi1_low=psi_fnc(func,x,alow,p)
            history+=[ap]
            function_iters+=2
        k=k+1

def Bracketing(func,x,p,a0,mu1,mu2,sigma):
    # input
    # a0>0: Initial step size
    # psi0,psi1_0: computed in outer routine passed to save function call
    # mu2<mu1<1: Sufficent Curvature Factor
    # sigma>1: step size increase factor

    #Outputs
    #a_star: Acceptable step size
    #history: History of a*

    a1=0
    a2=a0
    psi0,psi1_0=psi_fnc(func,x,0,p)
    psi1,psi1_1=psi_fnc(func,x,a1,p)
    function_iters=2
    first=True
    flag=True
    Bracket_history=[]

    while (flag is True):
        psi2,psi1_2=psi_fnc(func,x,a2,p)
        function_iters+=1
        Bracket_history+=[a1,a2]

        if (psi2>psi0+mu1*a2*psi1_0) or ((first is not True) and (psi2>psi1)):
            #1=low,2=high

            a_star,history_pin,fnc_iters_pnpnt=Pinpoint(func,x,p,a1,a2,psi0,psi1_0,psi1,psi2,psi1_1,psi1_2,mu1,mu2)
            function_iters+=fnc_iters_pnpnt
            Bracket_history+=[a1,a2]
            return a_star,history_pin,Bracket_history, function_iters
        
        elif (abs(psi1_2)<=-1*mu2*psi1_0): #step acceptable exit line search
            a_star=a2
            return a_star,[a2],Bracket_history, function_iters

        elif(psi1_2>=0):
            #2=low,1=high
            a_star,history_pin,fnc_iters_pnpnt=Pinpoint(func,x,p,a2,a1,psi0,psi1_0,psi2,psi1,psi1_2,psi1_1,mu1,mu2)
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
def psi_fnc(f,xk,a,pk):
    # input
    # a: Step Size
    # xk: Current x value
    # pk: Current search direction vector
    # f: function that is being linesearched

    #outputs:
    # psi(a): The psi value at the current step
    # psi_prime(a): psi derivative at current step
    psi=f(xk+a*pk)[0]
    psi1=np.array(f(xk+a*pk)[1:]).dot(pk)[0]

    return psi,psi1

def Steepest_Descent(x0,tol,func,mu1=10**-5,mu2=0.7,sigma=2):
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
    #Historu: History of Optimizer
    
    History={}
    k=0
    x_history=[x0]
    a0=0.1
    x=x0
    x_prev=x0
    fx,f1x=func(x)
    f1x=np.array(f1x); fx=np.array(fx)
    fx_old,f1x_old=func(x_prev)
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
        psi,psi1=psi_fnc(func,x,anit,pk)
        function_iters+=1
        if (psi1>0):
            # print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1)
            while(psi1>0):
                anit=anit*.9
                psi,psi1=psi_fnc(func,x,anit,pk)
                function_iters+=1
        ak,history_pin,fnc_iters_bracket,=Backtrack_linesearch(func,x,pk,anit)
        function_iters+=fnc_iters_bracket
        #update all the variables
        x_prev=x
        x=x+ak*pk
        pk_old=pk
        x_history+=[x]
        f1x_old=f1x 
        fx_old=fx
        fx,f1x=func(x)
        ak_old=ak
        k+=1
        function_iters+=1
        convg_history+=[np.max(np.abs(f1x))]

    History['x']=x_history; History['convergence']=convg_history; History['Function Calls']=function_iters
    History['iterations']=k
    return x, fx, History

def NonLinear_Conjugate(x0,tol,func,mu1=10**-5,mu2=0.7,sigma=2):
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
    #History: History of optimizer

    History = {}
    k=0
    x_history=[x0]
    a0=0.1
    x=x0
    x_prev=x0
    fx,f1x=func(x)
    f1x=np.array(f1x); fx=np.array(fx)
    fx_old,f1x_old=func(x_prev)
    f1x_old=np.array(f1x_old); fx=np.array(fx_old)

    function_iters=2
    pk_old=-1*(f1x/np.linalg.norm(f1x))
    pk_old=np.array(pk_old)
    ak_old=a0

    convg_history=[np.max(np.abs(f1x))]
    # betak=1
    reset=False
    while (np.max(np.abs(f1x))>=tol):
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
        psi,psi1=psi_fnc(func,x,anit,pk)
        function_iters+=1
        if (psi1>0):
            # print("ERROR, psi'(0)>0 the slope is not negative, psi'(0)=",psi1)
            while(psi1>0):
                anit=anit*.9
                reset=True
                pk=-1*f1x/np.linalg.norm(f1x)
                psi,psi1=psi_fnc(func,x,anit,pk)
                function_iters+=1
        #linesearch algorithm 
        ak,history_pin,Bracket_history,fnc_iters_bracket=Bracketing(func,x,pk,anit,mu1,mu2,sigma)
        function_iters+=fnc_iters_bracket
        #update all the variables
        x_prev=x
        x=x+ak*pk
        pk_old=pk
        x_history+=[x]
        f1x_old=f1x 
        fx_old=fx
        fx,f1x=func(x)
        ak_old=ak
        k+=1
        function_iters+=1
        convg_history+=[np.max(np.abs(f1x))]
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

def uncon_optimizer(func, x0, epsilon_g, options=None):
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
        optimizer_function=NonLinear_Conjugate

    if options == "NonLinear_Conjugate":
        optimizer_function=NonLinear_Conjugate
    if options == "Steepest_Descent":
        optimizer_function=Steepest_Descent
    if options == "Quasi_Newton":
        optimizer_function=Quasi_Newton
        
    x0=np.array(x0)
    xopt,fopt,history=optimizer_function(x0,epsilon_g,func)
    output['history']=history

    # TODO: 

    return xopt, fopt, output
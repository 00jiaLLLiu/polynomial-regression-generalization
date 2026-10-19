import random
from numpy import *
import pandas as pd

def getMSE(X,Y,a):
    
    diff = dot(X , a.T )- Y 
    #shape(N,1)
    #print("diff=",diff)
    MSE = dot(diff.T,diff)/ X.shape[0] 
    #print("MSE.shape",MSE.shape)
    
    return MSE[0]


def getData(N,d,sigma):
    
    X = zeros((N,1))
    x = random.rand(N,1)
    
    for i in range(d+1):
        X= concatenate([X,x**i],axis = 1)
    X=delete(X,0,axis=1)   
    
    Z = random.normal(loc = 0.0,scale = sigma**2)
    if d>0:
        Y = cos(2*pi*X[:,1:2]) + Z 
    else:
        Y=  cos(2*pi*X[:,:1]) + Z
    
    #shape X,Y (N,d+1)
    
    return X,Y


"""calculate gradient"""

def gradient(X,Y,a):  

    diff = dot(X , a.T)- Y
    #print("diff=",diff)
    
    grad = dot( X.T, diff)/ X.shape[0]   
    #print("gradient=",grad)   #shape(1,d+1)
    return grad.T


def fitData(X,Y,a,sigma,weight_decay):
    
    "gradient descent" 
    
    ln_r=0.01   #learning rate
    grad=gradient(X,Y,a)
    for epochs in range(1000):#while not all( abs(grad)<= 1e-5 ):
        a = a *(1- weight_decay *ln_r)   - ln_r  * grad
        grad=gradient(X,Y,a)   #shape(1,d+1)
    #print("gradient=",grad)
    #print ("a=",a)

    ein=getMSE(X,Y,grad) 
    #print("ein:",ein)
    
    Xout,Yout=getData(1000,d,sigma)  # 1000 or 2000 points
    eout=getMSE(Xout,Yout,grad) 
    #print("eout:",eout)
    
    return  ein,eout,a


def experiment(N,d,sigma,weight_decay):
    
    A = zeros((1,d+1))
    Ein,Eout=[],[] 
    
    """M trails=50"""
    for i in range(50):
        X,Y=getData(N,d,sigma)  #shape X,Y (N,d+1)
        #print("X.shape  ", X.shape)
        
        a=random.rand(1,X.shape[1])  
        #print("a=",a)
        ein,eout,a=fitData(X,Y,a,sigma,weight_decay)
        Ein=append(Ein,ein)
        Eout=append(Eout,eout) 
        A = concatenate([A,a])
    
    
    """computed averaged Ein and Eout respectively"""
    A = delete(A,0,0)
    A_avg= mean(A,axis =0).reshape(1,X.shape[1])  
    #A_avg  before reshape: [a1,a2,a3,a4]  after :[[a1,a2,a3,a4]] 
    #otherwise:Ebias wrong format
    #print("A_Avg=",A_avg)
    
    Ein_avg = mean(Ein)
    Eout_avg = mean(Eout)
    
    """generates another large dataset"""
    X,Y = getData(1000,d,sigma)    # 1000/2000
    E_bias = getMSE(X,Y,A_avg)
    #print("E_bias",E_bias[0])
    
    return Ein_avg,Eout_avg,E_bias[0]

#N,d = X.shape

#weight_decay=[0,0.1]
for weight_decay in [0,0.1]:
    
    data_n,data_d,data_sigma,data_Ein,data_Eout,data_Ebias= [],[],[],[],[],[]
    for N in [2 , 5, 10, 20, 50, 100, 200]:
        for d in range(0,21):
            for sigma in [0.01,0.1,1]:
                print("\n")
                print("N = "+str(N)+"; d = "+str(d)+"; var = "+str(sigma))

                Ein_avg,Eout_avg,E_bias=experiment(N,d,sigma,weight_decay)
                #Ein_avg,Eout_avg,E_bias=experiment(N,d,sigma,weight_decay)
                
                print("Ein = "+str(Ein_avg)+"; Eout = "+str(Eout_avg)+"; E_bias = "+str(E_bias))
                data_n.append(N)
                data_d.append(d)
                data_sigma.append(sigma)
                data_Ein.append(Ein_avg)
                data_Eout.append(Eout_avg)
                data_Ebias.append(E_bias)
                
    result = pd.DataFrame(list(zip(data_n,data_d,data_sigma,data_Ein,data_Eout,data_Ebias)), columns=['N','d','Variance','Ein','Eout', 'E_bias'])

    # Saving results
    if weight_decay==0:
        result.to_csv("results.csv")
    else:
        result.to_csv("results_reg.csv")

import numpy as np
from numpy import random
from numpy import linalg
import copy
from scipy.optimize import fmin
import math
import scipy.linalg
import sys
from scipy.linalg import hadamard



def gen_J_offset(J,ipt,trgt,Ntmp,Ltmp):
    Jdiag=np.diag(J)
    Doff=np.zeros((Ntmp,Ntmp))
    e,X=np.zeros(2*Ltmp),np.zeros((2*Ltmp,Ntmp-1))
    for i in range(Ntmp):
        e[:Ltmp]=J[i,i]*trgt[i,:]
        e[Ltmp:]=J[i,i]*ipt[i,:]
        for j in range(Ltmp):
            X[j,:i]=trgt[:i,j]
            X[j,i:]=trgt[i+1:,j]
            X[j+Ltmp,:i]=ipt[:i,j]
            X[j+Ltmp,i:]=ipt[i+1:,j]
        tmp=np.dot(np.linalg.pinv(X),e)
        Doff[i,:i]=tmp[:i]
        Doff[i,i+1:]=tmp[i:]
    return Doff
    
    
def gen_J(iseed,is_Hadamard,Ntmp,Ltmp):
    random.seed(seed=iseed)
    
    if is_Hadamard:
        tmp=hadamard(Ntmp)
        ipt=tmp[:,:Ltmp]
        trgt=tmp[:,Ltmp:2*Ltmp]
    else:
        ipt,trgt=np.zeros((Ntmp,Ltmp)),np.zeros((Ntmp,Ltmp))
        for i in range(Ltmp):
            ipt[:,i:i+1] =np.where(random.uniform(-1,1,(Ntmp,1))>0,1,-1)
            trgt[:,i:i+1]=np.where(random.uniform(-1,1,(Ntmp,1))>0,1,-1)
    tmp=np.zeros((Ntmp,2*Ltmp))
    tmp[:,:Ltmp],tmp[:,Ltmp:]=trgt,ipt
    A=np.linalg.pinv(tmp)
    tmp[:,:Ltmp],tmp[:,Ltmp:]=trgt-ipt,trgt-ipt
    J=np.dot(tmp,A)
    Joff=gen_J_offset(J,ipt,trgt,Ntmp,Ltmp)
    #print(np.mean(np.fabs(np.diag(J))))
    J=J-np.diag(np.diag(J))+Joff
    np.savetxt("J_N%dL%d_%d.dat" %(Ntmp,Ltmp,iseed),J)
    tmpmap=np.zeros((2*Ltmp,Ntmp))
    for i in range(Ltmp):
        tmpmap[i*2,:]=ipt[:,i]
        tmpmap[i*2+1,:]=trgt[:,i]
    np.savetxt("map_N%dL%d_%d.dat"%(Ntmp,Ltmp,iseed),tmpmap,fmt="%d")
    return J,ipt,trgt


def gen_eigs(gamma,J,BETA_tmp,ipt,trgt,id_apl,Ntmp):
    aa=np.tanh(BETA_tmp*gamma)
    bb=np.tanh(BETA_tmp*(2*aa-gamma))
    a,b=(aa+bb)/2.0,(aa-bb)/2.0
    
    x=a*trgt[:,id_apl:id_apl+1]+b*ipt[:,id_apl:id_apl+1]
    u=calc_u(J,x,gamma,ipt[:,id_apl:id_apl+1],BETA_tmp)
    deno=np.zeros((Ntmp,Ntmp))
    #print(u.shape,J.shape,x.shape)
    
    for i in range(Ntmp):
        #deno[i]=np.cosh(u[:,0])
        deno[:,i]=np.cosh(u[:,0])
    deno=deno*deno
    DJ=BETA_tmp*J/deno - np.eye(Ntmp)
    eigs=np.linalg.eig(DJ)
    #print(np.dot(J,x))
    print(np.sum(np.fabs(np.tanh(u)-x)))
    return eigs,a,b



############## dyn ##############
def calc_u(J,x,gamma,ipt,BETA_tmp):
    return BETA_tmp*(np.dot(J,x)+gamma*ipt)


def func(J,x,gamma,ipt,BETA_tmp):
    u=calc_u(J,x,gamma,ipt,BETA_tmp)
    return np.tanh(u)-x

#def func1(J,x,gamma,ipt,BETA_tmp):
#    return np.dot(J,np.tanh(BETA_tmp*x))+gamma*ipt-x

def calc_next_noise(J,x,gamma,ipt,dt,BETA_tmp):
    df=func(J,x,gamma,ipt,BETA_tmp)*dt+np.sqrt(dt)*np.random.normal(0,0.01,(x.shape))  # for noisy dynamics
    x+=df
    return


    
def calc_next(J,x,gamma,ipt,dt,BETA_tmp):
    df=func(J,x,gamma,ipt,BETA_tmp)*dt
    x+=df
    return


def func_lrn_hebb(J,x,trgt):
    N=len(J)
    return np.dot(x,x.T)/float(N)

def func_lrn(J,x,trgt):
    N=len(J)
    return np.dot(trgt-x,x.T)/float(N)

def calc_next_lrn(J,x,gamma,ipt,trgt,dt,BETA_tmp,is_Hebb):
    df=func(J,x,gamma,ipt,BETA_tmp)*dt
    if is_Hebb:
        dJ=func_lrn_hebb(J,x,trgt)*0.01*dt
    else:
        dJ=func_lrn(J,x,trgt)*0.01*dt
    x+=df
    J+=dJ
    return

def calc_next_lrn_noise(J,x,gamma,ipt,trgt,dt,BETA_tmp,is_Hebb):
    df=func(J,x,gamma,ipt,BETA_tmp)*dt+np.sqrt(dt)*np.random.normal(0,0.0002,(x.shape))  # for noisy dynamics
    if is_Hebb:
        dJ=func_lrn_hebb(J,x,trgt)*0.01*dt
    else:
        dJ=func_lrn(J,x,trgt)*0.01*dt
    x+=df
    J+=dJ
    return

def calc_next_lrn1(J,x,gamma,ipt,trgt,dt,BETA_tmp,is_Hebb):
    df=func(J,x,gamma,ipt,BETA_tmp)*dt
    if is_Hebb:
        dJ=func_lrn_hebb(J,x,trgt)*0.03*dt
    else:
        dJ=func_lrn(J,x,trgt)*0.03*dt
    x+=df
    J+=dJ
    return


def gen_Df(x,_J,_gamma,_beta,_ipt):  # ipt.shape=(N,1)
    _N=len(x)
    u=calc_u(_J,x,_gamma,_ipt,_beta)
    deno=np.zeros((_N,_N))
    
    for i in range(_N):
        deno[:,i]=np.cosh(u[:,0])
    deno=deno**2
    return _beta*_J/deno - np.eye(_N)


def calc_next_lyp(J,x,U,L,gamma,ipt,dt,BETA_tmp):
    df=func(J,x,gamma,ipt,BETA_tmp)*dt
    
    Df = gen_Df(x,J,gamma,BETA_tmp,ipt)
    A = U.T.dot(Df.dot(U))
    dL = np.diag(A).copy()*dt
    
    A=np.triu(A.T,k=1).T-np.triu(A.T,k=1)
    dU = U.dot(A)*dt

    x+=df
    U+=dU
    L+=dL
    
    return
    

def gen_LEC(J,gamma,ipt,BETA_tmp,var,_N):
    x=var[:_N].reshape(_N,1)
    U=var[_N:_N+_N*_N].reshape(_N,_N)
    #L=var[-_N:]
    
    f=func(J,x,gamma,ipt,BETA_tmp)
    Df =gen_Df(x,J,gamma,BETA_tmp,ipt)
    A = U.T.dot(Df.dot(U))
    dL=np.diag(A).copy()
    
    A=np.triu(A.T,k=1).T-np.triu(A.T,k=1)
    dU=U.dot(A)
    return np.concatenate([f.flatten(),dU.flatten(),dL])

    

def calc_lypexp(J,x0,gamma,ipt_trgt,T,nitr_rcd,BETA_tmp):
    dt=0.001
    N=len(J)
    x=np.copy(x0)
    dyn=[]
    ipt=ipt_trgt[0]
    trgt=ipt_trgt[1]

    dyn.append(np.copy(x))
    U=np.identity(N)
    L=np.zeros(N)
    dyn_L=[np.copy(L)]
    
    dtinv=int(1/dt)
    if dtinv==0:
        print("invalid dt")
        return 


    for it in np.arange(0,int(T/dt)):
        calc_next_lyp(J,x,U,L,gamma,ipt,dt,BETA_tmp)
                    
        if it%nitr_rcd==0:
            dyn.append(np.copy(x))
            dyn_L.append(np.copy(L))
            
    dyn=np.array(dyn)
    dyn_L=np.array(dyn_L)

    return x,dyn,dyn_L



def calc_dyn1(J,x0,gamma,ipt_trgt,T,nitr_rcd,is_rcl,BETA_tmp,*,is_noisy_dyn=False,is_Hebb=False,is_chkfp=False):
    
    dt=0.1
    N=len(J)
    x=np.copy(x0)
    dyn=[]
    ipt=ipt_trgt[0]
    trgt=ipt_trgt[1]
    
    dyn.append(np.copy(x))
    #dyn.append(np.vstack(([0],np.copy(x) )) )
    
    dtinv=int(1/dt)
    if dtinv==0:
        print("invalid dt")
        return 

    chk_fp,chk_fp1,chk_fp2=-10,-20,0
    cnt_fp=dtinv
    
    for it in np.arange(0,int(T/dt)):
        if is_rcl:
            if not is_noisy_dyn:
                calc_next(J,x,gamma,ipt,dt,BETA_tmp)
            else:
                calc_next_noise(J,x,gamma,ipt,dt,BETA_tmp)
            
            #"""
            if is_chkfp and cnt_fp<it:
                chk_fp2=np.mean(x[:,0]*trgt[:,0])
                if np.abs(chk_fp-chk_fp1)<0.00001/N and np.abs(chk_fp-chk_fp2)<0.00001/N:
                    break
                else:
                    chk_fp=chk_fp1
                    chk_fp1=chk_fp2
                    cnt_fp+=dtinv
             #"""
        else:
            calc_next_lrn1(J,x,gamma,ipt,trgt,dt,BETA_tmp,is_Hebb)
            if np.mean(x[:,0]*trgt[:,0])>0.75:
                break
            
        if it%nitr_rcd==0:
            #dyn.append(  np.vstack(([it*dt],np.copy(x) )) )
            dyn.append(np.copy(x))
            
    dyn.append(np.copy(x))
    #dyn.append( np.vstack(([it*dt],np.copy(x) )) )
    dyn=np.array(dyn)
    if is_rcl:
        return x,dyn
    else:
        return x,dyn,J

    
def calc_dyn(J,x0,gamma,ipt_trgt,T,nitr_rcd,is_rcl,BETA_tmp,*,is_noisy_dyn=False,is_Hebb=False,is_chkfp=True):
    
    dt=0.05
    N=len(J)
    x=np.copy(x0)
    dyn=[]
    ipt=ipt_trgt[0]
    trgt=ipt_trgt[1]
    

    dyn.append(np.copy(x))
    
    dtinv=int(1/dt)
    if dtinv==0:
        print("invalid dt")
        return 

    chk_fp,chk_fp1,chk_fp2=-10,-20,0
    cnt_fp=dtinv
    
    for it in np.arange(0,int(T/dt)):
        if is_rcl:
            if not is_noisy_dyn:
                calc_next(J,x,gamma,ipt,dt,BETA_tmp)
            else:
                calc_next_noise(J,x,gamma,ipt,dt,BETA_tmp)
            
            #"""
            if is_chkfp and cnt_fp<it:
                chk_fp2=np.mean(x[:,0]*trgt[:,0])
                if np.abs(chk_fp-chk_fp1)<0.00001/N and np.abs(chk_fp-chk_fp2)<0.00001/N:
                    break
                else:
                    chk_fp=chk_fp1
                    chk_fp1=chk_fp2
                    cnt_fp+=dtinv
             #"""
        else:
            if not is_noisy_dyn:
                calc_next_lrn(J,x,gamma,ipt,trgt,dt,BETA_tmp,is_Hebb)
            else:
                calc_next_lrn_noise(J,x,gamma,ipt,trgt,dt,BETA_tmp,is_Hebb)
            if np.mean((x-trgt)*(x-trgt))<0.05:
                break
            
        if it%nitr_rcd==0:
            dyn.append(np.copy(x))
    dyn.append(np.copy(x))
    dyn=np.array(dyn)
    if is_rcl:
        return x,dyn
    else:
        return x,dyn,J


####################################################

    
##############  dyn  with ipt   ####################

def calc_u_with_ipt(J,x,ipt_tmp,BETA_tmp):
    return BETA_tmp*(np.dot(J,x)+ipt_tmp)

def func_with_ipt(J,x,ipt_tmp,BETA_tmp):
    u=calc_u_with_ipt(J,x,ipt_tmp,BETA_tmp)
    return np.tanh(u)-x

def calc_next_with_ipt(J,x,ipt_tmp,dt,BETA_tmp):
    #df=func(J,x,gamma,ipt,BETA_tmp)*dt+np.sqrt(dt)*np.random.normal(0,0.01,(x.shape))
    df=func_with_ipt(J,x,ipt_tmp,BETA_tmp)*dt
    x+=df
    return    
    
def calc_dyn_with_ipt(J,x0,ipt_tmp,ipt_trgt,T,nitr_rcd,is_rcl,BETA_tmp):
    dt=0.05
    x=np.copy(x0)
    dyn=[]
    ipt=ipt_trgt[0]
    if not is_rcl:
        trgt=ipt_trgt[1]
    
    for it in np.arange(0,int(T/dt)):
        if is_rcl:
            calc_next_with_ipt(J,x,ipt_tmp,dt,BETA_tmp)
            
        if it%nitr_rcd==0:
            dyn.append(np.copy(x))
    dyn=np.array(dyn)
    if is_rcl:
        return x,dyn
    else:
        return x,dyn,J

########################################################


    
def calc_dyn_interval(J,x0,gamma,ipt_trgt,T,T1,T2,nitr_rcd,is_rcl,BETA_tmp):
    dt=0.05
    x=np.copy(x0)
    dyn=[]
    ipt=ipt_trgt[0]
    if not is_rcl:
        trgt=ipt_trgt[1]
    
    for it in np.arange(0,int(T/dt)):
        if is_rcl:
            if it*dt>T1 and it*dt<T2:
                calc_next(J,x,gamma,ipt,dt,BETA_tmp)
            else:
                calc_next(J,x,0,ipt,dt,BETA_tmp)
        else:
            calc_next_lrn(J,x,gamma,ipt,trgt,dt,BETA_tmp)
            if np.mean((x-trgt)*(x-trgt))<0.05:
                break
            
        if it%nitr_rcd==0:
            dyn.append(np.copy(x))
    dyn=np.array(dyn)
    if is_rcl:
        return x,dyn
    else:
        return x,dyn,J
    
def calc_dyn_increasing_gamma(J,x0,gamma,ipt_trgt,T,nitr_rcd,is_rcl,BETA_tmp):
    dt=0.05
    x=np.copy(x0)
    dyn=[]
    ipt=ipt_trgt[0]
    trgt=ipt_trgt[1]
    
    for it in np.arange(0,int(T/dt)):
        if is_rcl:
            calc_next(J,x,gamma*it*dt/T,ipt,dt,BETA_tmp)
        else:
            calc_next_lrn(J,x,gamma,ipt,trgt,dt,BETA_tmp)
            if np.mean((x-trgt)*(x-trgt))<0.05:
                break
            
        if it%nitr_rcd==0:
            #if np.mean(x*trgt)>0.9:
            #    break
            dyn.append(np.copy(x))
    dyn=np.array(dyn)
    
    if is_rcl:
        return x,dyn
    else:
        return x,dyn,J


def gen_dyn_list(list_gamma,J, _BETA, _ipt,_trgt,is_trgt,Ninit,N):
    dyn_gamma=[]
    if type(list_gamma)==list:
        list_gamma=np.array(list_gamma)
    
    for iinit in range(Ninit):
        if is_trgt=="True":
            over,xfp=gen_exact_overlap(list_gamma,_BETA,_ipt,_trgt,len(_trgt))
        elif is_trgt=="False":
            x=np.random.uniform(-1,1,(N,1))

        for igamma, gamma in enumerate(list_gamma):
            if is_trgt=="True":
                x=xfp[:,igamma:igamma+1]+np.random.normal(0,0.000001,(N,1))
            # if is_trgt==False, xinit is the final state in the previous trial
            xinit=x
            x,dyn=calc_dyn(J,xinit,gamma,[_ipt,_trgt],100,10,True,_BETA)    
            dyn_gamma.append(dyn)
    return dyn_gamma





def gen_exact_sol(gamma_list,beta_tmp):
    ab=np.tanh(beta_tmp*gamma_list)
    a_b=np.tanh(beta_tmp*(2*ab-gamma_list))
    a=(ab+a_b)/2.0
    b=(ab-a_b)/2.0
    return a,b

def gen_exact_overlap(gamma_list,beta_tmp,ipt,trgt,N):
    a,b=gen_exact_sol(gamma_list,beta_tmp)
    a,b=a.reshape(1,-1),b.reshape(1,-1)
    #xfp= np.dot(trgt,a)+np.dot(ipt,b)/beta_tmp 
    xfp= np.dot(trgt,a)+np.dot(ipt,b)
    over=np.dot(xfp.T,trgt)/N
    return over,xfp

def gen_exact_overlap1(gamma_list,beta_tmp,id_ipt):
    a,b=gen_exact_sol(gamma_list,beta_tmp)
    a,b=a.reshape(1,-1),b.reshape(1,-1)
    xfp=np.arctanh(np.dot(trgt[:,id_ipt:id_ipt+1],a)+np.dot(ipt[:,id_ipt:id_ipt+1],b))/beta_tmp
    over=np.dot(xfp.T,trgt[:,id_ipt:id_ipt+1])/N
    return over,xfp
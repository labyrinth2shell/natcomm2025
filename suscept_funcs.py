import numpy as np
import pylab as pl
import matplotlib.ticker as ptick
from numpy import random
import copy
import os
from itertools import product

N=512*1
sigma=0.01
f_d_ratio=sigma**2/2
ADR="/Users/tomoki/Dropbox/research/programs/Bif_Anlys/16th_data/"


def gen_lrnspd_alltypes(type_trgt_ipt,type_J,inet,**kwargs):

    if type_J=="random_sym":
        tmp=np.load(ADR+"lrndyn_rndJ_"+type_trgt_ipt+"ipt_N512_inet%d_highgamma.npz"%inet, allow_pickle=True)
    elif type_J=="random_asym":
        tmp=np.load(ADR+"lrndyn_asymJ_"+type_trgt_ipt+"ipt_N512_inet%d.npz"%inet, allow_pickle=True)
    elif type_J=="preemb":
        alpha=kwargs["alpha"]
        tmp=np.load(ADR+"lrndyn_embJ_"+type_trgt_ipt+"ipt3_a%g_N512_inet%d.npz"%(alpha,inet), allow_pickle=True)
        Jipt,Jtrgt=tmp["Jipt"],tmp["Jtrgt"]
    elif type_J=="hopf":
        alpha=kwargs["alpha"]
        tmp=np.load(ADR+"lrndyn_hopfJ_"+type_trgt_ipt+"ipt2_a%g_N512_inet%d.npz"%(alpha,inet), allow_pickle=True)
        Jtrgt=tmp["Jtrgt"]
    else:
        print("invalid type_J="+type_J)

    #tmp=np.load("dyn_rndipt_lrn_N512_inet%d.npz"%inet, allow_pickle=True)
    list_beta=tmp["list_beta"]
    #list_beta=[0.6]
    gamma=tmp["gamma"]
    [ipt,trgt]=tmp["ipt_trgt"]
    list_id=tmp["list_id"]
    Nmap=len(list_id)
    dyn_lrn=np.array([tmp["dyn"+str(i)] for i in range(Nmap*len(list_beta))])
    J=tmp["J"]
    eigs=np.linalg.eig(J)
    dt,eps=tmp["dt"],tmp["eps"]


    Tbin=10

    if type_J=="hopf":
        Nresp=1003
        Ninterval=10
    else:
        Nresp=200
        Ninterval=20
    

    
    Tinterval=Ninterval*(Tbin*dt)

    lrnspd=[]
    for ib,i in product(range(len(list_beta)),range(Nmap)):
        itmp=ib*Nmap+i
        diff_vec=(dyn_lrn[itmp][Nresp+Ninterval]-dyn_lrn[itmp][Nresp])/Tinterval
        lrnspd.append( np.linalg.norm(diff_vec) )
    lrnspd=np.array(lrnspd).reshape(len(list_beta),Nmap)

    lrnspd_all=dict()
    lrnspd_all["lrnspd_emp"]=lrnspd

    
    
    if type_J=="preemb":
        # calc lrn spd by spn with trgt, spn with ipt
        lrnspd_spnspn=calc_lrnspd_all("spn_spn",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                      Nresp=Nresp,inet=inet,type_J=type_J,alpha=alpha,eps=eps)
    
        # calc lrn spd by spn with trgt, resp
        lrnspd_respspn=calc_lrnspd_all("resp_spn",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                       Nresp=Nresp,inet=inet,type_J=type_J,alpha=alpha,eps=eps)

        lrnspd_all["spnspn"]=lrnspd_spnspn
        lrnspd_all["respspn"]=lrnspd_respspn
        
        avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,list_beta,type_J,alpha=alpha)
        avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,list_beta,type_J,alpha=alpha)
        
    elif type_J=="hopf":
        # calc lrn spd by spn with trgt, spn with ip
        lrnspd_all["spnspn_hebb"]=[]
        avevar_spn_ipt=[]
        lrnspd_spnspn,avevar_spn_resp,avevar_spn_ipt=calc_lrnspd_all("spn_spn_hebb",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                      Nresp=Nresp,inet=inet,type_J=type_J,alpha=alpha,eps=eps,fp=dyn_lrn[:,0,:].T)
        lrnspd_all["spnspn_hebb"]=lrnspd_spnspn

        
        # calc lrn spd by spn with trgt, resp
        #lrnspd_all["respspn_hebb"]=[]
        lrnspd_respspn,avevar_spn_resp=calc_lrnspd_all("resp_spn_hebb",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                       Nresp=Nresp,inet=inet,type_J=type_J,alpha=alpha,eps=eps)
        lrnspd_all["respspn_hebb"]=lrnspd_respspn
                
    elif type_J=="random_sym" or type_J=="random_asym":

        if type_trgt_ipt=="random":
            # calc lrn spd  by spn with trgt, spn with ipt in radom case
            lrnspd_spnspn=calc_lrnspd_all("spn_spn",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                          Nresp=Nresp,inet=inet,eps=eps,type_J=type_J,)

            # calc lrn spd  by spn with trgt, resp in radom case
            lrnspd_respspn=calc_lrnspd_all("resp_spn",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                           Nresp=Nresp,inet=inet,eps=eps,type_J=type_J,)

        else:

            # calc lrn spd by spn with trgt, spn with ipt
            lrnspd_spnspn=calc_lrnspd_all("spn_spn",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                          Nresp=Nresp,inet=inet,type_J=type_J,eps=eps)
            # calc lrn spd by spn with trgt, resp
            lrnspd_respspn=calc_lrnspd_all("resp_spn",list_beta,ipt,trgt,gamma=gamma,dyn=dyn_lrn,\
                                           Nresp=Nresp,inet=inet,type_J=type_J,eps=eps)

        #  calc  lrn spd by spn with eig, resp
        #lrnspd_respspneig=calc_lrnspd_all("resp_spn_eig",list_beta,ipt,trgt,eigs=eigs,gamma=gamma,dyn=dyn_lrn,\
        #                                  Nresp=Nresp,inet=inet)

        #  calc semi-analytical lrn spd   (determined only by eigs, resp)
        #lrnspd_semianal=calc_lrnspd_all("semi_analytical",list_beta,ipt,trgt,eigs=eigs,gamma=gamma,dyn=dyn_lrn,Nresp=Nresp)

        # calc analytical learning speed   (determined only by eigs)
        lrnspd_anal=calc_lrnspd_all("analytical",list_beta,ipt,trgt,eigs=eigs,gamma=gamma,eps=eps,type_J=type_J)

        
        avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,list_beta,type_J)
        avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,list_beta,type_J)
        lrnspd_all["anal"]=lrnspd_anal
        lrnspd_all["spnspn"]=lrnspd_spnspn
        lrnspd_all["respspn"]=lrnspd_respspn


    if type_J=="hopf":
        resp=np.array([i[Nresp] for i in dyn_lrn]).T  
        spn_resp={"spn_resp":avevar_spn_resp,"spn_ipt":avevar_spn_ipt, "resp":resp}
    else:
        resp                      =[dyn_lrn[i][Nresp]  for i in range(len(dyn_lrn))]
        spn_resp={"spn_trgt":avevar_spn_trgt,"spn_ipt":avevar_spn_ipt,"resp":resp}

      
    return lrnspd_all,list_beta,spn_resp
    #lrnspd_alpha_trgt.append(lrnspd[0][...])
    #lrnspd_respspn_alpha_trgt.append(lrnspd_respspn[0][...])
    #lrnspd_spnspn_alpha_trgt.append(lrnspd_spnspn[0][...])
    
def calc_lrnspd_all(type_lrnspd,_list_beta,ipt,trgt,**kwargs):
    """
    type_lrnspd= analytical:  eigs
    type_lrnspd= semi_analytical:  eigs, resp   
    """
    lrnspd=[]
    
    
    eps=kwargs["eps"]
    gamma=kwargs["gamma"]
    inet=kwargs.get("inet")  # used for calc of spont fluc.
    eigs=kwargs.get("eigs")
    dyn  = kwargs.get("dyn")
    Nresp = kwargs.get("Nresp")  # num of steps the neural dynamics converge into the response state
    type_J=kwargs.get("type_J")
    
    Nmap=len(ipt[0])
    
    if type_lrnspd=="semi_analytical": 
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],eigs=eigs,beta=beta,gamma=gamma,eps=eps)
                lrnspd[-1].append(tmp)
                
    elif type_lrnspd=="analytical":
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],eigs=eigs,beta=beta,gamma=gamma,eps=eps)
                lrnspd[-1].append(tmp)
            
    elif type_lrnspd=="resp_spn_eig":
        avevar_spn_eig=np.load(ADR+"avevar_spn_eig_all_inet%d.npy"%inet)
        
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],\
                                eigs=eigs,beta=beta,gamma=gamma,resp=dyn[ib*Nmap+imap][Nresp],\
                                spn_eig=avevar_spn_eig[ib],eps=eps)
                lrnspd[-1].append(tmp)
                
    elif type_lrnspd=="resp_spn":
        
        if type_J=="random_sym"or type_J=="random_asym":
            avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,_list_beta,type_J)
            avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,_list_beta,type_J)
        elif type_J=="preemb":
            alpha=kwargs["alpha"]
            avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,_list_beta,type_J,alpha=alpha)
            avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,_list_beta,type_J,alpha=alpha)
        else:
            print("invalid type_J"+type_J+" and type_lrnspd="+type_lrnspd+"in calc_lrnspd_all")
       
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],\
                                beta=beta,gamma=gamma,resp=dyn[ib*Nmap+imap][Nresp],\
                                spn=avevar_spn_trgt[ib,imap],eps=eps)
                lrnspd[-1].append(tmp)    
                
    elif type_lrnspd=="spn_spn":
        
        if type_J=="random_sym"or type_J=="random_asym":
            avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,_list_beta,type_J)
            avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,_list_beta,type_J)
        elif type_J=="preemb":
            alpha=kwargs["alpha"]
            avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,_list_beta,type_J,alpha=alpha)
            avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,_list_beta,type_J,alpha=alpha)
        else:
            print("invalid type_J"+type_J+" and type_lrnspd="+type_lrnspd+"in calc_lrnspd_all")
           
     
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],\
                                beta=beta,gamma=gamma,spn_trgt=avevar_spn_trgt[ib,imap],\
                                spn_ipt=avevar_spn_ipt[ib,imap],eps=eps)
                lrnspd[-1].append(tmp)
                
     
    
    ###############################################################    
    ##################   Pure Hebb learning for Hopfield connectivity     #########     
    elif type_lrnspd=="resp_spn_hebb":
        if type_J=="hopf":
            alpha=kwargs["alpha"]
            resp=np.array([i[Nresp] for i in dyn]).T 

            avevar_spn_resp=[]
            #"""
            for ib,beta in enumerate(_list_beta):
                resptmp=resp[:,ib*Nmap:(ib+1)*Nmap]
                vec=beta*resptmp/np.cosh(np.arctanh(resptmp))**2
                tmp_spn=calc_avevar_spn_vec(inet,vec,[beta],type_J,alpha=alpha)
                avevar_spn_resp.append(tmp_spn[0])
            avevar_spn_resp=np.array(avevar_spn_resp)
            #"""
            
            """
            tmp=np.load(ADR+"spn_resp_hopfJ_a%g_N512_inet%d.npz"%(alpha,inet),allow_pickle=True)
            dyn_noise_all=tmp["dyn"]
            
            for i in range(len(dyn_noise_all)):
                avevar_spn_resp.append(np.cov(dyn_noise_all[i][-9000:,:].T))
            """
            
        else:
            print("invalid type_J"+type_J+" and type_lrnspd="+type_lrnspd+"in calc_lrnspd_all")

        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],\
                                beta=beta,gamma=gamma,resp=dyn[ib*Nmap+imap][Nresp],\
                                #spn_resp=avevar_spn_resp[ib*25+imap],eps=eps)
                                spn_resp=avevar_spn_resp[ib,imap],eps=eps)
                lrnspd[-1].append(tmp)    
                
        
    elif type_lrnspd=="spn_spn_hebb":
        if type_J=="hopf":
            fp=kwargs["fp"]
            alpha=kwargs["alpha"]
            resp=np.array([i[Nresp] for i in dyn]).T 
            
            avevar_spn_resp=[]
            for ib,beta in enumerate(_list_beta):
                resptmp=resp[:,ib*Nmap:(ib+1)*Nmap]
                vec=beta*resptmp/np.cosh(np.arctanh(resptmp))**2
                tmp_spn=calc_avevar_spn_vec(inet,vec,[beta],type_J,alpha=alpha)
                avevar_spn_resp.append(tmp_spn[0])
            avevar_spn_resp=np.array(avevar_spn_resp)
            
            avevar_spn_ipt=[]
            for ib,beta in enumerate(_list_beta):
                vec=( beta/np.cosh(np.arctanh(fp[:,ib*Nmap:(ib+1)*Nmap]))**2 ) * ipt
                tmp_spn=calc_avevar_spn_vec(inet,vec,[beta],type_J,alpha=alpha)
                avevar_spn_ipt.append(tmp_spn[0])
            avevar_spn_ipt=np.array(avevar_spn_ipt)
            """    
            # in this case, fp=Jtrgt[:,0]
            vec=gamma*ipt+np.tanh(beta*fp)
            avevar_spn_ipt  =calc_avevar_spn_vec(inet,vec,_list_beta,type_J,alpha=alpha)
            """
        else:
            print("invalid type_J"+type_J+" and type_lrnspd="+type_lrnspd+"in calc_lrnspd_all")

        fp=kwargs["fp"]
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],\
                                beta=beta,gamma=gamma,resp=dyn[ib*Nmap+imap][Nresp],\
                                spn_resp=avevar_spn_resp[ib,imap],spn_ipt=avevar_spn_ipt[ib,imap],eps=eps,fp=fp[:,ib*Nmap+imap])
                #fp=fp
                lrnspd[-1].append(tmp)    
    ###############################################################
    ###############################################################
        
        
        
    elif type_lrnspd=="rnd_resp_spn":
        avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,_list_beta,type_J)
        ave_var  = calc_spn_all(inet,_list_beta)
            
        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],\
                                beta=beta,gamma=gamma,resp=dyn[ib*Nmap+imap][Nresp],\
                                spn=avevar_spn_trgt[ib,imap],spn_all=ave_var,eps=eps)
                lrnspd[-1].append(tmp) 
                
    elif type_lrnspd=="rnd_spn_spn":
        avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,_list_beta,type_J)
        avevar_spn_ipt  =calc_avevar_spn_vec(inet,ipt,_list_beta,type_J)
        ave_var  = calc_spn_all(inet,_list_beta)

        for ib,beta in enumerate(_list_beta):
            lrnspd.append([])
            
            for imap in range(len(ipt[0])):
                tmp=calc_lrnspd(type_lrnspd,ipt[:,imap:imap+1],trgt[:,imap:imap+1],beta=beta,gamma=gamma,\
                                spn_trgt=avevar_spn_trgt[ib,imap],spn_ipt=avevar_spn_ipt[ib,imap],spn_all=ave_var,eps=eps)
                lrnspd[-1].append(tmp)    

        
    else:
        print("invalid type_lrnspd="+type_lrnspd)

        
    if type_lrnspd=="resp_spn_hebb":
        return np.array(lrnspd),avevar_spn_resp
    elif type_lrnspd=="spn_spn_hebb":
        return np.array(lrnspd),avevar_spn_resp,avevar_spn_ipt
    else:
        return np.array(lrnspd)



def calc_lrnspd(type_lrnspd,_ipt,_trgt,**kwargs):
    beta=kwargs["beta"]
    gamma=kwargs["gamma"]
    eps=kwargs["eps"]
    
    if type_lrnspd=="analytical":
        _eigs=kwargs["eigs"]
        resptmp = calc_resp(beta,gamma,_ipt,_eigs,"analytical")
        tmp         = calc_wgt_prj(_eigs,beta,_trgt)
        lrnspd     = beta*eps*np.sum(resptmp**2)*tmp/N
    elif type_lrnspd=="semi_analytical":
        _eigs=kwargs["eigs"]
        resptmp = kwargs["resp"]
        tmp         = calc_wgt_prj(_eigs,beta,_trgt)
        lrnspd     = beta*eps*np.sum(resptmp**2)*tmp/N
    elif type_lrnspd=="resp_spn_eig":
        _eigs=kwargs["eigs"]
        spn_eig=kwargs["spn_eig"]
        resptmp = kwargs["resp"]
        tmp        = calc_wgt_spn(_eigs,spn_eig,_trgt)
        lrnspd     = beta*eps*np.sum(resptmp**2)*tmp/N
    elif type_lrnspd=="resp_spn":
        spn=kwargs["spn"]
        resptmp = kwargs["resp"]
        lrnspd     = beta*eps*np.sum(resptmp**2)*(spn/f_d_ratio)*_trgt/N
        
    elif type_lrnspd=="spn_spn":
        spn_ipt=kwargs["spn_ipt"]
        spn_trgt=kwargs["spn_trgt"]
        resptmp=beta*gamma*spn_ipt/f_d_ratio*_ipt
        lrnspd     = beta*eps*np.sum(resptmp**2)*(spn_trgt/f_d_ratio)*_trgt/N
        
        
        
        
    elif type_lrnspd=="resp_spn_hebb":
        spn_resp=kwargs["spn_resp"]
        resptmp = kwargs["resp"]
        beta_tmp=beta/np.cosh(np.arctanh(resptmp))**2
        lrnspd     = eps*np.sum(resptmp**2)*(spn_resp/f_d_ratio)*(beta_tmp*resptmp).reshape(-1,1)/N
        #lrnspd     = eps*np.sum(resptmp**2)* ( (spn_resp/f_d_ratio)@(beta_tmp*resptmp).reshape(-1,1))/N
        
    elif type_lrnspd=="spn_spn_hebb":
        spn_resp=kwargs["spn_resp"]
        spn_ipt  =kwargs["spn_ipt"]
        resptmp = kwargs["resp"]
        fp        =kwargs["fp"]
        
        """
        beta_tmp=beta/(np.cosh(beta*fp[:,0])**2)
        resptmp2  = ((spn_ipt/f_d_ratio)*(np.tanh(beta*fp[:,0])+beta_tmp*(gamma*_ipt[:,0]-fp[:,0])))**2
        beta_tmp=beta/np.cosh(np.arctanh(resptmp))**2
        """
        
        beta_tmp=beta/np.cosh(np.arctanh(fp))**2
        resptmptmp=(gamma*(spn_ipt/f_d_ratio)*beta_tmp*_ipt[:,0]+fp)
        beta_tmp=beta/np.cosh(np.arctanh(resptmp))**2
        
        lrnspd     =eps*np.sum(resptmptmp**2)*(spn_resp/f_d_ratio)*(beta_tmp*resptmptmp).reshape(-1,1)/N
        
    elif type_lrnspd=="rnd_resp_spn":
        spn=kwargs["spn"]
        spn_all=kwargs["spn_all"]/N
        spn_tmp=np.sqrt(spn*spn_all)
        resptmp = kwargs["resp"]
        lrnspd     = beta*eps*np.sum(resptmp**2)*(spn_tmp/f_d_ratio)*_trgt/N
    elif type_lrnspd=="rnd_spn_spn":
        spn_ipt=kwargs["spn_ipt"]
        spn_trgt=kwargs["spn_trgt"]
        spn_all=kwargs["spn_all"]/N
        
        spn_tmp=np.sqrt(spn_trgt*spn_all)
        resptmp=beta*gamma*np.sqrt(spn_ipt*spn_all)/f_d_ratio*_ipt
        lrnspd     = beta*eps*np.sum(resptmp**2)*(spn_tmp/f_d_ratio)*_trgt/N
  

    else:
        print("invalid type_lrnspd="+type_lrnspd)    
 
    return lrnspd

def calc_avevar_spn_vec(inet,vec,_list_beta,_type_J,**kwargs):
    spn_trgt,dyn_label=calc_spn_vec(inet,vec,_type_J,**kwargs)
    #return spn_trgt,dyn_label

    avevar_spn_trgt=[]
    for ib,beta in enumerate(_list_beta):
        tmplist=np.where(np.round(dyn_label,10)==np.round(beta,10))[0]
        avevar_spn_trgt.append(np.mean(spn_trgt[tmplist],axis=0))
    return np.array(avevar_spn_trgt)



def gen_J(_N,_type_J,_inet,**kwargs):
    random.seed(seed=_inet)
        
    if _type_J=="random_sym":
        J=random.randn(_N,_N)/np.sqrt(_N)
        J-=np.diag(np.diag(J))
        J=(J+J.T)/2
        return J
    elif _type_J=="random_asym":
        J=random.randn(_N,_N)/np.sqrt(_N)
        J-=np.diag(np.diag(J))
        return J
    elif   _type_J=="preemb": 
        alpha=kwargs["alpha"]
        L=int(alpha*_N)
        ipt,trgt=np.zeros((_N,L)),np.zeros((_N,L))

        for i in range(L):
            ipt[:,i:i+1] =np.where(random.uniform(-1,1,(_N,1))>0,1,-1)
            trgt[:,i:i+1]=np.where(random.uniform(-1,1,(_N,1))>0,1,-1)
    
        J=(trgt-ipt) @ (trgt+ipt).T/_N
        J-=np.diag(np.diag(J))
        return J,trgt,ipt
    elif _type_J=="hopf":
        alpha=kwargs["alpha"]
        L=int(alpha*_N)
        trgt=np.zeros((_N,L))

        for i in range(L):
            trgt[:,i:i+1]=np.where(random.uniform(-1,1,(_N,1))>0,1,-1)
    
        J=trgt @ trgt.T/_N
        J-=np.diag(np.diag(J))
        return J,trgt

    else:
        print("in gen_J, invalid type_J="+_type_J)
    
    return
 
def gen_trgt_ipt(_N,_Ninp,_type,**kwargs):
    
    ipt,trgt=np.zeros((_N,_Ninp)),np.zeros((_N,_Ninp))
    _Ninp2=int(np.sqrt(_Ninp))
    
    if _type=="eig":
        J=kwargs["J"]
        eigs=np.linalg.eig(J)
        
        """
        tmp=-np.linspace(1,_N-3,_Ninp2).astype("int")
        ids_eig=np.argsort(eigs[0])[tmp]
        tmp=-np.linspace(3,_N-1,_Ninp2).astype("int")
        ids_eig1=np.argsort(eigs[0])[tmp]
    
        tmp=np.where(eigs[1][:,ids_eig]>0,1,-1)
        tmp1=np.where(eigs[1][:,ids_eig1]>0,1,-1)
        for i in range(_Ninp2):
            ipt[:,i*_Ninp2:(i+1)*_Ninp2]=np.tile(tmp[:,i:i+1],(1,_Ninp2))
            trgt[:,i::_Ninp2]=np.tile(tmp1[:,i:i+1],(1,_Ninp2))
       """
        
        _Ninp2=int(_Ninp/2)
        tmp=-np.linspace(1,_N-3,_Ninp2+1).astype("int")
        ids_eig=np.argsort(eigs[0])[tmp]
        tmp=-np.linspace(3,_N-1,_Ninp2+1).astype("int")
        ids_eig1=np.argsort(eigs[0])[tmp]

        tmp=np.where(eigs[1][:,ids_eig]>0,1,-1)
        tmp1=np.where(eigs[1][:,ids_eig1]>0,1,-1)
  
        for i in range(_Ninp2):
            ipt[:,:_Ninp2]  =tmp[:,:_Ninp2]
            trgt[:,:_Ninp2]=np.tile( tmp1[:,0:1], (1,_Ninp2) )
            
            ipt[:,_Ninp2:]  =np.tile( tmp[:,_Ninp2:_Ninp2+1],   (1,_Ninp2) )
            trgt[:,_Ninp2:]=tmp1[:,1:_Ninp2+1]

        
    elif _type=="random":
        """
        tmp=np.where(random.uniform(-1,1,(_N,_Ninp2))>0,1,-1)
        tmp1=np.where(random.uniform(-1,1,(_N,_Ninp2))>0,1,-1)

        for i in range(_Ninp2):
            ipt[:,i*_Ninp2:(i+1)*_Ninp2]=np.tile(tmp[:,i:i+1],(1,_Ninp2))
            trgt[:,i::_Ninp2]=np.tile(tmp1[:,i:i+1],(1,_Ninp2))
        """
        
        _Ninp2=int(_Ninp/2)
        tmp=np.where(random.uniform(-1,1,(_N,_Ninp2+1))>0,1,-1)
        tmp1=np.where(random.uniform(-1,1,(_N,_Ninp2+1))>0,1,-1)

        for i in range(_Ninp2):
            ipt[:,:_Ninp2]  =tmp[:,:_Ninp2]
            trgt[:,:_Ninp2]=np.tile( tmp1[:,0:1], (1,_Ninp2) )
            
            ipt[:,_Ninp2:]  =np.tile( tmp[:,_Ninp2:_Ninp2+1],   (1,_Ninp2) )
            trgt[:,_Ninp2:]=tmp1[:,1:_Ninp2+1]

            
    elif _type=="random_preemb":
        Jipt=kwargs["Jipt"]
        Jtrgt=kwargs["Jtrgt"]
        tmp0=gen_random_pat(Jtrgt,Jipt,_Ninp*2)
        #ipt[:,:]=np.where(tmp0[:,:_Ninp]>0,1,-1)
        #trgt[:,:]=np.where(tmp0[:,_Ninp:]>0,1,-1)
        
        #for *2.npy
        """
        tmp=np.where(tmp0[:,:_Ninp2]>0,1,-1)
        tmp1=np.where(tmp0[:,_Ninp2:_Ninp2*2]>0,1,-1)

        for i in range(_Ninp2):
            ipt[:,i*_Ninp2:(i+1)*_Ninp2]=np.tile(tmp[:,i:i+1],(1,_Ninp2))
            trgt[:,i::_Ninp2]=np.tile(tmp1[:,i:i+1],(1,_Ninp2))
        """
        
        # for *3.npy
        Ninp2=(int)(_Ninp/2)
        bNinp2=_Ninp-Ninp2
        ipt[:,:Ninp2]=np.where(tmp0[:,:Ninp2]>0,1,-1)
        tmp=np.where(tmp0[:,Ninp2:Ninp2+1]>0,1,-1)
        ipt[:,Ninp2:]=np.tile(  tmp,(1,bNinp2)   )
        
        trgt[:,Ninp2:]=np.where(tmp0[:,_Ninp+Ninp2:]>0,1,-1)
        tmp=np.where(tmp0[:,_Ninp:_Ninp+1]>0,1,-1)
        trgt[:,:Ninp2]=np.tile(  tmp,(1, Ninp2)   )

    elif _type=="trgt_preemb":
        Jipt=kwargs["Jipt"]
        Jtrgt=kwargs["Jtrgt"]
        if len(Jipt[0])<_Ninp:
            print("invalid: Nmap=%d>alphaN=%d"%(_Ninp,len(Jipt[0])))
            
        #ipt[:,:-1]=Jipt[:,1:_Ninp]
        #ipt[:,-1] =Jipt[:,0]
        #trgt[:,:]=Jtrgt[:,:_Ninp]
        
         #for *2.npy
        """
        tmp=np.where(Jipt[:,:_Ninp2]>0,1,-1)
        tmp1=np.where(Jtrgt[:,_Ninp2:_Ninp2*2]>0,1,-1)

        for i in range(_Ninp2):
            ipt[:,i*_Ninp2:(i+1)*_Ninp2]=np.tile(tmp[:,i:i+1],(1,_Ninp2))
            trgt[:,i::_Ninp2]=np.tile(tmp1[:,i:i+1],(1,_Ninp2))
        """    
            
        # for *3.npy
        Ninp2=(int)(_Ninp/2)
        bNinp2=_Ninp-Ninp2
        ipt[:,:Ninp2]=np.where(Jipt[:,:Ninp2]>0,1,-1)
        tmp=np.where(Jipt[:,Ninp2:Ninp2+1]>0,1,-1)
        ipt[:,Ninp2:]=np.tile(  tmp,(1,bNinp2)   )
        
        trgt[:,Ninp2:]=np.where(Jtrgt[:,:bNinp2]>0,1,-1)
        tmp=np.where(Jtrgt[:,bNinp2:bNinp2+1]>0,1,-1)
        trgt[:,:Ninp2]=np.tile(  tmp,(1, Ninp2)   )
        
    elif _type=="random_hopf":
        Jtrgt=kwargs["Jtrgt"]
        tmp0=gen_random_pat(Jtrgt,[],_Ninp)
        ipt=np.where(tmp0>0,1,-1)
        trgt=np.copy(ipt)
        
    elif _type=="entire_lrn_eig":
        J=kwargs["J"]
        eigs=np.linalg.eig(J)
        tmp=-np.linspace(2,_N-1,_Ninp).astype("int")
        ids_eig=np.argsort(eigs[0])[tmp]
        trgt[:,:]=np.where(eigs[1][:,ids_eig]>0,1,-1)
    
        ipt[:,:]=trgt[:,:]
        ipt[(int)(_N/2):,:]*=-1
    elif _type=="entire_lrn_random":
        trgt[:,:]=np.where(random.uniform(-1,1,(_N,_Ninp))>0,1,-1)    
        ipt[:,:]=trgt[:,:]
        ipt[(int)(_N/2):,:]*=-1
    elif _type=="entire_lrn_random_preemb":
        Jipt=kwargs["Jipt"]
        Jtrgt=kwargs["Jtrgt"]
        tmp=gen_random_pat(Jtrgt,Jipt,_Ninp)
        trgt[:,:]=np.where(tmp[:,:_Ninp]>0,1,-1)
        ipt[:,:]=trgt[:,:]
        ipt[(int)(_N/2):,:]*=-1
    elif _type=="entire_lrn_trgt_preemb":
        Jipt=kwargs["Jipt"]
        Jtrgt=kwargs["Jtrgt"]
        if len(Jipt[0])<_Ninp:
            print("invalid: Nmap=%d>alphaN=%d"%(_Ninp,len(Jipt[0])))
        ipt[:,:-1]=Jipt[:,1:_Ninp]
        ipt[:,-1] =Jipt[:,0]
        trgt[:,:]=Jtrgt[:,:_Ninp]
        
        # for *3.npy
        """
        Ninp2=(int)(_Ninp/2)
        bNinp2=_Ninp-Ninp2
        ipt[:,:Ninp2]=np.where(Jipt[:,:Ninp2]>0,1,-1)
        tmp=np.where(Jipt[:,Ninp2:Ninp2+1]>0,1,-1)
        ipt[:,Ninp2:]=np.tile(  tmp,(1,bNinp2)   )
        
        trgt[:,Ninp2:]=np.where(Jtrgt[:,:bNinp2]>0,1,-1)
        tmp=np.where(Jtrgt[:,bNinp2:bNinp2+1]>0,1,-1)
        trgt[:,:Ninp2]=np.tile(  tmp,(1, Ninp2)   )
        """
    else:
        print("Error in gen_trgt_ipt !!  invalid trgt_ipt type")
        
    return trgt,ipt


def gen_random_pat(trgt,ipt,_Nrnd):
    """
    orthogonization of a  random input against target and input space
    """

    ipt_rnd=np.zeros((N,_Nrnd))
    if len(ipt)!=0:
        tmp=np.zeros((N,2*int(len(ipt[0]))))
        tmp[:,:int(len(ipt[0]))]=ipt
        tmp[:,int(len(ipt[0])):int(len(ipt[0]))*2]=trgt
    else:
        tmp=np.zeros((N,int(len(trgt[0]))))
        tmp[:,:int(len(trgt[0]))]=trgt
        
    for i in range(_Nrnd):
        ipt_tmp=np.where(random.uniform(-1,1,(N,1))>0,1,-1)
        q,r=np.linalg.qr(tmp)
        ipt_tmp=ipt_tmp-np.dot(q,np.dot(ipt_tmp.T,q).T)
        ipt_tmp/=np.sqrt(np.mean(ipt_tmp*ipt_tmp))
        tmp=np.hstack((tmp,ipt_tmp))
        ipt_rnd[:,i:i+1]=ipt_tmp
        
    return ipt_rnd


def calc_spn_all(inet,_list_beta,type_J):
    if type_J=="random_sym":
        tmp=np.load(ADR+"spn_rndJ_N512_inet%d.npz"%inet,allow_pickle=True)
    elif type_J=="random_asym":
        tmp=np.load(ADR+"spn_asymJ_N512_inet%d.npz"%inet,allow_pickle=True)

    dyn_noise_all=tmp["dyn"]

    var_all=[]
    for i in range(len(dyn_noise_all)):
        var_all.append(np.sum(np.var(dyn_noise_all[i][-9000:,:],axis=0)))
    var_all=np.array(var_all).reshape(len(_list_beta),-1)
    return np.mean(var_all,axis=1)



def load_fulllrn_data(inet,type_J,type_trgt_ipt,**kwargs):

    if type_J=="random_sym":
        tmp=np.load(ADR+"fulllrndyn_rndJ_"+type_trgt_ipt+"ipt_N512_inet%d.npz"%inet, allow_pickle=True)
    elif type_J=="random_asym":
        tmp=np.load(ADR+"fulllrndyn_asymJ_"+type_trgt_ipt+"ipt_N512_inet%d.npz"%inet, allow_pickle=True)
    elif type_J=="preemb":
        alpha=kwargs["alpha"]
        tmp=np.load(ADR+"fulllrndyn_embJ_"+type_trgt_ipt+"ipt2_a%g_N512_inet%d.npz"%(alpha,inet), allow_pickle=True)
    
    return tmp


def get_resp(dyn_lrn,T):
    resp=[]
    for i in range(len(dyn_lrn)):
        resp.append(dyn_lrn[i][T,:])
    return np.array(resp)

    
    



def calc_wgt_prj(_eigs,_b,_vec):
    _eigvals=_eigs[0]
    _eigvecs=_eigs[1]
    coef=(_eigvecs.T @ _vec) /(1- _b * _eigvals).reshape(-1,1)
    return _eigvecs @ coef

def calc_wgt_spn(_eigs,spn_eig,_vec):
    _eigvals=_eigs[0]
    _eigvecs=_eigs[1]
    coef=(_eigvecs.T @ _vec) * spn_eig.reshape(-1,1)
    #coef=(_eigvecs.T @ _vec)
    #coef1=spn_eig.reshape(-1,1)
    return _eigvecs @ (coef/f_d_ratio)

def calc_spn_vec(inet,vecs,_type_J,**kwargs):
    if _type_J=="random_sym":
        tmp=np.load(ADR+"spn_rndJ_N512_inet%d.npz"%inet,allow_pickle=True)
    elif _type_J=="random_asym":
        tmp=np.load(ADR+"spn_asymJ_N512_inet%d.npz"%inet,allow_pickle=True)
    elif _type_J=="preemb":
        alpha=kwargs["alpha"]
        tmp=np.load(ADR+"spn_embJ_a%g_N512_inet%d.npz"%(alpha,inet),allow_pickle=True)
    elif _type_J=="hopf":
        alpha=kwargs["alpha"]
        tmp=np.load(ADR+"spn_hopfJ_a%g_N512_inet%d.npz"%(alpha,inet),allow_pickle=True)
    else:
        print("invalid type_J %s in calc_spn_vec()"%_type_J)

    dyn_noise_all=tmp["dyn"]
    dyn_label=tmp["dyn_label"]
    norm_vecs=vecs/np.tile(np.linalg.norm(vecs,axis=0).reshape(1,-1), (len(vecs),1))

    spn_trgt=[]
    for i in range(len(dyn_noise_all)):
        spn_trgt.append(np.var(dyn_noise_all[i][-9000:,:]@norm_vecs,axis=0))
    spn_trgt=np.array(spn_trgt)

    return spn_trgt,dyn_label
    

def calc_resp(_b,_g,_ipt,_eigs,_type_resp):
    if _type_resp=="analytical":
        tmp=calc_wgt_prj(_eigs,_b,_ipt)
        return _b*_g*tmp
    else:
        print("invalid type_resp="+_type_resp)
        return " "

"""
def gen_fig_lrnspd(tmp_lrnspd,emp_lrnspd,list_param):
    for ia, alpha in enumerate(list_param):
        x=np.linalg.norm(tmp_lrnspd[ia][:,:,0],axis=1)
        pl.scatter(x,emp_lrnspd[ia])
    tmp=np.array([np.linalg.norm(tmp_lrnspd[ia][:,:,0],axis=1) for ia in range(len(list_param))])
    x=[np.min(tmp)*0.95,np.max(tmp)*1.05]
    pl.plot(x,x)
    pl.xscale("log")
    pl.yscale("log")
    pl.tight_layout()
    
    return
"""



def gen_fig_lrnspd(lrnspd_all,list_beta,figname,_type_J="preemb",**kwargs):
    plot_beta=kwargs.get("plot_beta")
    if plot_beta==None:
        plot_beta=list_beta
    
    axes=[]


    plot_expd_beta=kwargs.get("plot_expd_beta")
    if plot_expd_beta==None:
        fig=pl.figure(figsize=(12,6))
        plot_expd_beta=[]
        for i in range(2):
            axes.append(fig.add_subplot(1,2,i+1))
    else:
        fig=pl.figure(figsize=(12,12))
        for i in range(4):
            axes.append(fig.add_subplot(2,2,i+1))

    if _type_J=="preemb":
        axes[0].set_title("spn x spn^2")
        axes[1].set_title("spn x resp^2")
    elif _type_J=="random_sym" or _type_J=="random_asym":
        axes[0].set_title("o: spn x spn^2  x: spn x resp^2")
    elif _type_J=="hopf":
        axes[0].set_title("spn x resp^2")
    else:
        print("invalid type_J in gen_fig_lrnspd")


    vmax,vmin=np.max(plot_beta),np.min(plot_beta)

    xrang1,xrang2=[],[]

    for ib,beta in enumerate(list_beta):
        if not np.round(beta,10) in np.round(plot_beta,10):
            continue

        if _type_J == "random_sym" or _type_J=="random_asym":
            y=lrnspd_all["lrnspd_emp"][ib]
            x=np.linalg.norm(lrnspd_all["spnspn"][ib][:,:,0],axis=1)
            im0=axes[0].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=100,marker="o")

            x=np.linalg.norm(lrnspd_all["respspn"][ib][:,:,0],axis=1)
            axes[0].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=120,marker="x")
            xrang1.append([np.min(x),np.max(x)])
        elif _type_J =="preemb":
            y=lrnspd_all["lrnspd_emp"][ib][::2]
            x=np.linalg.norm(lrnspd_all["spnspn"][ib][:,:,0],axis=1)[::2]
            im0=axes[0].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=70,marker="o")

            x=np.linalg.norm(lrnspd_all["respspn"][ib][:,:,0],axis=1)[::2]
            axes[0].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=120,marker="x")
            xrang1.append([np.min(x),np.max(x)])
            
        elif _type_J=="hopf":
            y=lrnspd_all["lrnspd_emp"][ib]
            x=np.linalg.norm(lrnspd_all["respspn_hebb"][ib][:,:,0],axis=1)
            im0=axes[0].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=50,marker="x")
            xrang1.append([np.min(x),np.max(x)])
            

            x=np.linalg.norm(lrnspd_all["spnspn_hebb"][ib][:,:,0],axis=1)
            axes[1].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=50,marker="o")
            xrang2.append([np.min(x),np.max(x)])

            
            
        
        """
        x=np.linalg.norm(lrnspd_all["respspn"][ib][:,:,0],axis=1)
        xrang2.append([np.min(x),np.max(x)])
        axes[1].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=60,marker="x")
        """
    if not plot_expd_beta==None:
        if _type_J in ["random_sym","preemb","random_asym"]:
            for ib,beta in enumerate(list_beta):
                if not np.round(beta,10) in np.round(plot_expd_beta,10):
                    continue
                y=lrnspd_all["lrnspd_emp"][ib]
                x=np.linalg.norm(lrnspd_all["spnspn"][ib][:,:,0],axis=1)
                xrang1.append([np.min(x),np.max(x)])
                axes[2].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=60,marker="x")
                x=np.linalg.norm(lrnspd_all["respspn"][ib][:,:,0],axis=1)
                xrang2.append([np.min(x),np.max(x)])
                axes[3].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=60,marker="x")
                
        else:
            for ib,beta in enumerate(list_beta):
                if not np.round(beta,10) in np.round(plot_expd_beta,10):
                    continue
                y=lrnspd_all["lrnspd_emp"][ib]
                x=np.linalg.norm(lrnspd_all["spnspn_hebb"][ib][:,:,0],axis=1)
                axes[2].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=50,marker="o")
                x=np.linalg.norm(lrnspd_all["respspn_hebb"][ib][:,:,0],axis=1)
                axes[3].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="cool",s=50,marker="x")
            

    if _type_J=="hopf":
        xmin,xmax=np.min(np.array(xrang1))*0.5,np.max(np.array(xrang1))*7.05 
    else:
        xmin,xmax=np.min(np.array(xrang1))*0.95,np.max(np.array(xrang1))*1.05
    x=np.linspace(xmin,xmax,4)
    axes[0].plot(x,x)
    
    if not len(xrang2)==0:
        if _type_J=="hopf":
            xmin,xmax=np.min(np.array(xrang2))*0.3,np.max(np.array(xrang2))*3.05  # for Hopfield
        else:
            xmin,xmax=np.min(np.array(xrang2))*0.95,np.max(np.array(xrang2))*1.05
    x=np.linspace(xmin,xmax,4)
    axes[1].plot(x,x)
    fig.colorbar(im0)
    
    """
    xmin,xmax=np.min(np.array(xrang2))*0.95,np.max(np.array(xrang2))*1.05
    x=np.linspace(xmin,xmax,4)
    axes[1].plot(x,x)
    """
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    pl.savefig(figname)

    
def gen_fig_lrnspd1(lrnspd_all,list_beta,figname,**kwargs):
    plot_beta=kwargs.get("plot_beta")
    if plot_beta==None:
        plot_beta=list_beta
    
    axes=[]

    plot_expd_beta=kwargs.get("plot_expd_beta")
    if plot_expd_beta==None:
        fig=pl.figure(figsize=(12,6))
        plot_expd_beta=[]
        for i in range(2):
            axes.append(fig.add_subplot(1,2,i+1))
    else:
        fig=pl.figure(figsize=(12,12))
        for i in range(4):
            axes.append(fig.add_subplot(2,2,i+1))


    axes[0].set_title("spn x spn^2")
    axes[1].set_title("spn x resp^2")
    vmax,vmin=np.max(plot_beta),np.min(plot_beta)

    xrang1,xrang2=[],[]

    for ib,beta in enumerate(list_beta):
        if not np.round(beta,10) in np.round(plot_beta,10):
            continue
        y=lrnspd_all["lrnspd_emp"][ib]
        x=list_beta[ib]*np.ones(len(y))
        axes[0].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=130,marker="o")

        
        """
        x=np.linalg.norm(lrnspd_all["respspn"][ib][:,:,0],axis=1)
        xrang2.append([np.min(x),np.max(x)])
        axes[1].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=60,marker="x")
        """
    if not plot_expd_beta==None:
        for ib,beta in enumerate(list_beta):
            if not np.round(beta,10) in np.round(plot_expd_beta,10):
                continue
            y=lrnspd_all["lrnspd_emp"][ib]
            x=np.linalg.norm(lrnspd_all["spnspn"][ib][:,:,0],axis=1)
            xrang1.append([np.min(x),np.max(x)])
            axes[2].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=60,marker="x")
            x=np.linalg.norm(lrnspd_all["respspn"][ib][:,:,0],axis=1)
            xrang2.append([np.min(x),np.max(x)])
            axes[3].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(x)),cmap="rainbow",s=60,marker="x")


    xmin,xmax=np.min(np.array(xrang1))*0.95,np.max(np.array(xrang1))*1.05
    x=np.linspace(xmin,xmax,4)
    axes[0].plot(x,x)
    xmin,xmax=np.min(np.array(xrang2))*0.95,np.max(np.array(xrang2))*1.05
    x=np.linspace(xmin,xmax,4)
    axes[1].plot(x,x)
    
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    pl.savefig(figname)
    
    
def gen_figs_entire_lrn(inet,type_J,type_trgt_ipt,fname,**kwargs):
    is_logscale=True
    
    if type_J=="random_sym" or type_J=="random_asym":
        tmp=load_fulllrn_data(inet,type_J,type_trgt_ipt)
    else:
        alpha=kwargs["alpha"]
        tmp=load_fulllrn_data(inet,type_J,type_trgt_ipt,alpha=alpha)
        is_logscale=kwargs["is_logscale"]
    gamma=tmp["gamma"]
    list_beta=tmp["list_beta"]
    [ipt,trgt]=tmp["ipt_trgt"]
    dyn_label=tmp["dyn_label"]
    dyn_lrn=[tmp["dyn"+str(i)] for i in range(len(dyn_label))]
    t=[i[:,0] for i in dyn_lrn]
    dyn_lrn=[i[:,1:] for i in dyn_lrn]
    list_id=tmp["list_id"]
    Nmap=len(list_id)
    dt=tmp["dt"]
    Tbin=tmp["Tbin"]
    J=tmp["J"]
    eps=tmp["eps"]
    print(gamma)
    if type_J=="random_sym" or type_J=="random_asym":
        avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,list_beta,type_J)
        avevar_spn_ipt=calc_avevar_spn_vec(inet,ipt,list_beta,type_J)
    else:
        avevar_spn_trgt=calc_avevar_spn_vec(inet,trgt,list_beta,type_J,alpha=alpha)
        avevar_spn_ipt=calc_avevar_spn_vec(inet,ipt,list_beta,type_J,alpha=alpha)

    Nres=42
    resp=get_resp(dyn_lrn,Nres)
    
    print(list_beta)
    fig=pl.figure(figsize=(12,6))
    axes=[]
    for i in range(6):
        axes.append(fig.add_subplot(2,3,i+1))

    axes[0].set_title("spn x spn^2")
    axes[1].set_title("spn ")
    axes[2].set_title("spn x resp^2")
    vmax,vmin=np.max(list_beta),np.min(list_beta)

    for ib,beta in enumerate(list_beta):
        if (type_J in ["random_sym","random_asym"] and ib in [0,1,3,5]) or type_J=="preemb":
            tmp1=np.where(dyn_label==beta)
            y=[1/(len(dyn_lrn[j]-Nres)*(dt*Tbin)) for  j in tmp1[0]]
            print(len(y),len(y[::2]))
            
            # x[::3] -> x[::2] for make 10 samples in asym J.   2025.2.10 (1st revision for Nat comm)
            x=(beta**3)*avevar_spn_trgt[ib]*avevar_spn_ipt[ib]**2*(eps/(N*f_d_ratio))*gamma**2*(eps/(N*f_d_ratio**3))*np.sqrt(N)**3
            axes[0].scatter(x[::2],y[::2],vmin=vmin,vmax=vmax,c=beta*np.ones(len(tmp1[0][::2])),cmap="cool",s=100,marker="o")

            x=beta*avevar_spn_trgt[ib]*(eps/(N*f_d_ratio))
            axes[1].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(tmp1[0])),cmap="cool")

            x=beta*avevar_spn_trgt[ib]*[np.linalg.norm(resp[j])**2 for  j in tmp1[0]]*(eps/(N*f_d_ratio))*np.sqrt(N)
            axes[2].scatter(x[::2],y[::2],vmin=vmin,vmax=vmax,c=beta*np.ones(len(tmp1[0][::2])),cmap="cool",s=160,marker="x")

        #xmin,xmax=1e-4,1e-2
        #x=np.linspace(xmin,xmax,3)
        #axes[2].plot(x,x)
    for ib,beta in enumerate(list_beta[:1]):
        tmp1=np.where(dyn_label==beta)
        y=[1/(len(dyn_lrn[j])*(dt*Tbin)) for  j in tmp1[0]]
        
        x=(beta**3)*avevar_spn_trgt[ib]*avevar_spn_ipt[ib]**2*(eps/(N*f_d_ratio))
        axes[3].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(tmp1[0])),cmap="rainbow")
        
        x=beta*avevar_spn_trgt[ib]*(eps/(N*f_d_ratio))
        axes[4].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(tmp1[0])),cmap="rainbow")
        
        x=beta*avevar_spn_trgt[ib]*[np.linalg.norm(resp[j])**2 for  j in tmp1[0]]*(eps/(N*f_d_ratio))*np.sqrt(N)
        axes[5].scatter(x,y,vmin=vmin,vmax=vmax,c=beta*np.ones(len(tmp1[0])),cmap="rainbow")
  
    
    for ax in axes[:3]:
        ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
        
        if is_logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
    fig.tight_layout()
    #fig.colorbar(im)
    pl.savefig(fname)

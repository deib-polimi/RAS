'''
Created on 14 mag 2021

@author: emilio
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import numpy as np

curpath=os.path.realpath(__file__)

import casadi

class qnTransient:
    model = None
    stateVar = None
    sVar = None
    pVar = None
    tVar = None
    initX = None
    E_abs = None
    
    def __init__(self):
        self.model = None
        self.stateVar = None
        self.sVar = None
        self.pVar = None
        self.tVar = None
        self.initX = None
        self.E_abs = None
    
    def buildOpt(self, MU, P, S,dt,tgt,H):
        self.model = casadi.Opti()
        
        self.sVar = self.model.variable(P.shape[0],1);
        self.pVar = self.model.variable(P.shape[0],P.shape[1])
        self.stateVar = self.model.variable(P.shape[0],H);
        self.tVar = self.model.variable(P.shape[0],H);
        self.initX = self.model.parameter(P.shape[0],1)
        # self.E_abs = self.model.variable(1,H)
        

        #self.model.subject_to(self.model.bounded(0.0,self.sVar,3000))
        self.model.subject_to(self.sVar>=0)
        self.model.subject_to(self.model.bounded(0,casadi.vec(self.pVar),1))
        
        for i in range(P.shape[0]):
            if(S[i,0]>=0):
                self.model.subject_to(self.sVar[i]==S[i,0])
            for j in range(P.shape[1]):
                if(P[i,j]>=0):
                    self.model.subject_to(self.pVar[i,j]==P[i,j])
            #self.model.subject_to((self.pVar[i,0]+self.pVar[i,1]+self.pVar[i,2])==1.0)

        self.model.subject_to(self.stateVar[:,0]==self.initX);
        
        for i in range(P.shape[0]):
            for h in range(H):
                self.model.subject_to(self.sVar[1,0]<=self.stateVar[1,h])
                if(i==0):
                    self.model.subject_to(self.tVar[i,h]==MU[i])
                else:
                    #self.model.subject_to(self.tVar[i,h]==MU[i]*casadi.fmin(self.sVar[i,0],self.stateVar[i,h]))
                    self.model.subject_to(self.tVar[i,h]==MU[i]*self.sVar[i,0])
        
        for i in range(P.shape[0]):
            for h in range(H-1):
                self.model.subject_to(self.stateVar[i,h+1]==(-self.tVar[i,h]+self.pVar[:,i].T@self.tVar[:,h])*dt+self.stateVar[i,h])
                # self.model.subject_to(self.stateVar[i,h+1]==(-MU[i]*casadi.fmin(self.sVar[i],self.stateVar[i,h])
                #                                               +self.pVar[:,i].T@(MU*casadi.fmin(self.sVar,self.stateVar[:,h])))*dt+self.stateVar[i,h])
        
        # for h in range(H):
        #     self.model.subject_to(self.E_abs[0,h]>=(self.stateVar[1,h]-tgt*(1.0/MU[1])*self.tVar[1,h]))
        #     self.model.subject_to(self.E_abs[0,h]>=-(self.stateVar[1,h]-tgt*(1.0/MU[1])*self.tVar[1,h]))
            
        #self.model.minimize(casadi.sumsqr(self.stateVar[1,:]-tgt))
        # obj=0
        # for h in range(H):
        #     obj+=(self.stateVar[1,h]-tgt*MU[1]*self.tVar[1,h])**2
        
        optionsIPOPT={'print_time':False,'ipopt':{'print_level':0}}
        optionsOSQP={'print_time':False,'osqp':{'verbose':False}}
        self.model.solver('ipopt',optionsIPOPT)
        #self.model.solver('osqp',optionsOSQP)
        
if __name__ == "__main__":
    
    import matlab.engine
    
    eng = matlab.engine.start_matlab()
    eng.cd(r"%s/qn_sim/"%(os.path.dirname(curpath)))
    
    P = np.matrix([[0,1,0],[0,0,1],[0,0,1]])
    MU = np.matrix([10,100,0]).T
    S = np.matrix([0, -1,0]).T
    XS=np.zeros([3,10000])
    TS=np.zeros([1,XS.shape[1]])
    #XS[:,0]=np.random.randint(low=0,high=100,size=[1,3])
    Us=np.zeros(XS.shape)
    dt=0.05*10**(-1)
    H=10
    TF=H*dt;
    Time=np.linspace(0,TF,int(np.ceil(TF/dt))+1)
    ctrl = qnTransient()
    rt=[]
    
    #alfa=np.round(np.random.rand(),4)
    tgt=20
    ctrl.buildOpt(MU, P, S,dt,tgt,H)
    
    sold=None
    
    for s in tqdm(range(0,XS.shape[1])):
        #print(XS[:,[s-1]])
        ctrl.model.set_value(ctrl.initX, XS[:,[s-1]])
        
        obj=0
        for h in range(0,H):
            obj+=(ctrl.stateVar[1,h]-tgt/MU[1,0]*ctrl.tVar[1,h])**2
            #obj+=casadi.fabs(ctrl.stateVar[1,h]-tgt*ctrl.tVar[1,h])
        
        ctrl.model.minimize(obj)
        if(sold is not None):
            ctrl.model.minimize(obj+.0*casadi.sumsqr(ctrl.sVar-sold)+0.000*casadi.sumsqr(ctrl.sVar))
        else:
            ctrl.model.minimize(obj)
        
        start=time.time()
        sol=ctrl.model.solve()
        #print(time.time()-start)

        sold=sol.value(ctrl.sVar)
        Us[:,s]=sol.value(ctrl.sVar)
        
        # #odeSol=QNIVP(P,MU,np.matrix(Us[:,s]).T,XS[:,[s-1]].T.tolist()[0],TF,dt,Time)
        Xsim=np.array(eng.qn_sim(matlab.double(XS[:,s-1].T.tolist()),matlab.double(Us[:,s].tolist()),matlab.double(MU.tolist())
                 ,matlab.double(P.tolist()),H*dt,1,dt))
        
        #print(Xsim[:,:])
        
        XS[:,[s]]=Xsim[:,[1]]
        TS[0,s]=Xsim[2,1]/((s+1)*dt)
        print(TS[0,s])
        # XS[2,[s]]=0
        
        print(np.mean(XS[1,0:s])/np.mean(TS[0,0:s]))
        rt.append(np.mean(XS[1,0:s])/np.mean(TS[0,0:s]))
        #rt.append(np.mean(RT[np.logical_and(~np.isnan(RT),np.logical_not(np.isinf(RT)))]))
        
        
    
    # plt.figure()
    # #plt.plot(TS.T)
    # plt.plot(sol.value(ctrl.stateVar).T[:,0])
    # plt.plot(Xsim.T[:,0])
    #
    # plt.figure()
    # #plt.plot(TS.T)
    # plt.plot(sol.value(ctrl.stateVar).T[:,1])
    # plt.plot(Xsim.T[:,1])
    #
    # plt.figure()
    # #plt.plot(TS.T)
    # plt.plot(sol.value(ctrl.stateVar).T[:,2])
    # plt.plot(Xsim.T[:,2])
    # #plt.axhline(y = alfa*0.441336*np.sum(XS[:,0]), color = 'r', linestyle = '--')
    
    plt.figure()
    plt.plot(Us[1,:])
    
    plt.figure()
    plt.plot(rt)
    plt.axhline(y = tgt/MU[1,0], color = 'r', linestyle = '--')
    plt.show()
    
    
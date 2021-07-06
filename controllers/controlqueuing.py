from .controller import Controller
from pyscipopt import Model
import numpy as np
from estimator import QNEstimaator


class OPTCTRL(Controller):
    
    esrimationWindow=20;
    rtSamples=None
    cSamples=None
    userSamples=None
    
    def __init__(self, period, init_cores, stime,maxCores,st=0.8):
        super().__init__(period, init_cores,st)
        self.stime=stime
        self.generator=None
        self.estimator=QNEstimaator()
        self.rtSamples=[[]]
        self.cSamples=[[]]
        self.userSamples=[[]]
        self.maxCores=maxCores
    
    
    def OPTController(self,e, tgt, C,maxCore):
        optCTRL = Model() 
        optCTRL.hideOutput()
        
        nApp=len(tgt)
        
        T=[optCTRL.addVar("t%d"%(i), vtype="C", lb=0, ub=None) for i in range(nApp)]
        S=[optCTRL.addVar("s%d"%(i), vtype="C", lb=10**-3, ub=maxCore) for i in range(nApp)]
        D=[optCTRL.addVar("d%d"%(i), vtype="B") for i in range(nApp)]
        E_l1 = [optCTRL.addVar(vtype="C", lb=0, ub=None) for i in range(nApp)]
        
        sSum=0
        obj=0;
        for i in range(nApp):
            sSum+=S[i]
            obj+=E_l1[i]/tgt[i]
            
        optCTRL.addCons(sSum<=maxCore)
        
        for i in range(nApp):
            optCTRL.addCons(T[i] <= S[i] / e[i])
            optCTRL.addCons(T[i] <= C[i] / e[i])
            optCTRL.addCons(T[i] >= S[i] / e[i] - C[i] / e[i] * D[i])
            optCTRL.addCons(T[i] >= C[i] / e[i] - C[i] / e[i] * (1 - D[i]))
            optCTRL.addCons(E_l1[i] >= ((C[i]/T[i])-tgt[i]))
            optCTRL.addCons(E_l1[i] >= -((C[i]/T[i])-tgt[i]))
        
        
        optCTRL.setObjective(obj)
        
        optCTRL.optimize()
        sol = optCTRL.getBestSol()
        return [sol[S[i]] for i in range(nApp)]
    
    def addRtSample(self,rt,u,c):
        if(len(self.rtSamples)>=self.esrimationWindow):
            self.rtSamples=np.roll(self.rtSamples,[-1,-1])
            self.cSamples=np.roll(self.cSamples,[-1,-1])
            self.userSamples=np.roll(self.userSamples,[-1,-1])
            
            self.rtSamples[-1]=rt
            self.cSamples[-1]=c
            self.userSamples[-1]=u
        else:
            self.rtSamples.append(rt)
            self.cSamples.append(c)
            self.userSamples.append(u)
        
    def control(self, t):
        rt = self.monitoring.getRT()
        users=self.monitoring.getUsers()
        
        self.addRtSample(rt,users,self.cores)
        
        mRt=np.array(self.rtSamples).mean(axis=0)
        mCores=np.array(self.cSamples).mean(axis=0)
        mUsers=np.array(self.userSamples).mean(axis=0)
        
        #i problemi di stima si possono parallelizzare
        for app in range(len(rt)):
            self.stime[app]=self.estimator.estimate(mRt[app], mCores[app],mUsers[app])
           
        if(self.generator!=None):
            users=self.generator.f(t+1)    
        else:
            users=int(self.monitoring.getUsers())
            
        #risolvo il problema di controllo ottimo
        self.cores=self.OPTController(self.stime, self.setpoint, users,self.maxCores)
    
    def reset(self):
        super().reset()
        self.rtSamples=[]
        self.cSamples=[]
        self.userSamples=[]
    
    def setSLA(self, sla):
        self.sla = sla
        self.setpoint = [s*self.st for s in self.sla]
      

    def __str__(self):
        return super().__str__() + " OPTCTRL: %.2f, l: %.2f h: %.2f " % (self.step, self.l, self.h)

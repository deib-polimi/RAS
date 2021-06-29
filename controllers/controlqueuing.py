from .controller import Controller
from pyscipopt import Model
import numpy as np
from estimator import QNEstimaator
# from Crypto.Random.random import sample


class OPTCTRL(Controller):
    
    esrimationWindow=20;
    rtSamples=None
    cSamples=None
    userSamples=None
    
    def __init__(self, period, init_cores, stime,st=0.8):
        super().__init__(period, init_cores,st)
        self.stime=stime
        self.generator=None
        self.estimator=QNEstimaator()
        self.rtSamples=[[]]
        self.cSamples=[[]]
        self.userSamples=[[]]
    
    
    def OPTController(self,e, tgt, C):
        optCTRL = Model()  
        optCTRL.hideOutput()
        t = optCTRL.addVar("t", vtype="C", lb=0, ub=None)
        s = optCTRL.addVar("s", vtype="C", lb=1, ub=None)
        d = optCTRL.addVar("y", vtype="B")
        e_l1 = optCTRL.addVar("e_l1", vtype="C", lb=0, ub=None)
        
        optCTRL.addCons(t <= s / e, name="d")
        optCTRL.addCons(t <= C / e)
        optCTRL.addCons(t >= s / e - C / e * d)
        optCTRL.addCons(t >= C / e - C / e * (1 - d))
        optCTRL.addCons(e_l1 >= C - t * tgt)
        optCTRL.addCons(e_l1 >= -C + t * tgt)
        
        optCTRL.setObjective(e_l1)
        
        optCTRL.optimize()
        sol = optCTRL.getBestSol()
        return sol[s]
    
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
        
        for app in range(len(rt)):
            self.stime[app]=self.estimator.estimate(mRt[app], mCores[app],mUsers[app])
           
        if(self.generator!=None):
            users=self.generator.f(t+1)    
        else:
            users=int(self.monitoring.getUsers())
        
        for app in range(len(rt)):
            self.cores[app]=self.OPTController(self.stime[app], self.setpoint[app], users[app])
    
    def resetEstimate(self):
        self.rtSamples=[]
        self.cSamples=[]
        self.userSamples=[]
    
    def setSLA(self, sla):
        self.sla = sla
        self.setpoint = [s*self.st for s in self.sla]
      

    def __str__(self):
        return super().__str__() + " OPTCTRL: %.2f, l: %.2f h: %.2f " % (self.step, self.l, self.h)

from .controller import Controller
import casadi
import numpy as np
from .estimator import QNEstimaator
from .circular import CircularArray


class OPTCTRL(Controller):
    
    esrimationWindow = 30
    rtSamples = None
    cSamples = None
    userSamples = None

    def __init__(self, period, init_cores, min_cores, max_cores=1000, st=0.8, name=None):
        super().__init__(period, init_cores, st, name=name)
        
        self.stime = []
        self.generator = None
        self.estimator = QNEstimaator()
        self.rtSamples = [[]]
        self.cSamples = [[]]
        self.userSamples = [[]]
        self.maxCores = max_cores
        self.min_cores = min_cores
        self.Ik=0
        self.noise=CircularArray(size=100)
    

    def setServiceTime(self, stime):
        if not isinstance(stime, list):
            self.stime = [stime]
        else:
            self.stime = stime
    
    def OPTController(self, e, tgt, C, maxCore):
        #print("stime:=", e, "tgt:=", tgt, "user:=", C)
        if(np.sum(C)>0):
            self.model = casadi.Opti() 
            nApp = len(tgt)
            
            T = self.model.variable(1, nApp);
            S = self.model.variable(1, nApp);
            E_l1 = self.model.variable(1, nApp);
            
            self.model.subject_to(T >= 0)
            #self.model.subject_to(self.model.bounded(0, S, maxCore))
            self.model.subject_to(E_l1 >= 0)
        
            sSum = 0
            obj = 0;
            for i in range(nApp):
                sSum += S[0, i]
                # obj+=E_l1[0,i]
                obj += (T[0, i] / C[i] - 1 / tgt[i]) ** 2
        
            for i in range(nApp):
                self.model.subject_to(T[0, i] == casadi.fmin(S[0, i] / e[i], C[i] / e[i]))
        
            self.model.minimize(obj)    
            optionsIPOPT = {'print_time':False, 'ipopt':{'print_level':0}}
            # self.model.solver('osqp',{'print_time':False,'error_on_fail':False})
            self.model.solver('ipopt', optionsIPOPT) 
            
            sol = self.model.solve()
            if(nApp==1):
                return min(maxCore, sol.value(S))
            else:
                return sol.value(S).tolist()
        else:
            return 10**(-3)
    
    def addRtSample(self, rt, u, c):
        if(len(self.rtSamples) >= self.esrimationWindow):
            # print("rolling",rt, u, c)
            # print(self.cSamples)
            self.rtSamples = np.roll(self.rtSamples, [-1, 0], 0)
            self.cSamples = np.roll(self.cSamples, [-1, 0], 0)
            self.userSamples = np.roll(self.userSamples, [-1, 0], 0)
            
            # print(self.cSamples)
            self.rtSamples[-1] = rt
            self.cSamples[-1] = c
            self.userSamples[-1] = u
            
            # print(self.cSamples[-1],c,np.round(c,5))
        else:
            #print("adding", rt, u, c)
            self.rtSamples.append(rt)
            self.cSamples.append(c)
            self.userSamples.append(u)

    def cmpNoise(self,core=None,users=None,st=None,rtm=None):
        pred=users/(core/st)
        noise=rtm-pred
        #print(f"###pred={pred}; noise={noise};")
        return max(noise,0)
        
    def control(self, t):
        if(len(self.monitoring.getAllRTs())>0):
            sIdx=max(len(self.monitoring.getAllRTs())-self.esrimationWindow,0)
            eIdx=min(sIdx+self.esrimationWindow,len(self.monitoring.getAllRTs()))

        self.addRtSample(self.monitoring.getAllRTs()[-1], self.monitoring.getAllUsers()[-1],self.monitoring.getAllCores()[-1])

        mRt = self.monitoring.getAllRTs()[sIdx:eIdx+1]
        mUsers = self.monitoring.getAllUsers()[sIdx:eIdx+1]
        mCores= self.monitoring.getAllCores()[sIdx:eIdx+1]
        
        #print(f"rt={len(mRt)},{len(mUsers)},{len(mCores)}")

        # i problemi di stima si possono parallelizzare 
        self.stime[0] = self.estimator.estimate(rt=self.rtSamples, 
                                                s=self.cSamples,
                                                c=self.userSamples)

        print(f"###estim {self.stime}")        
        # risolvo il problema di controllo ottimo
        if(t>0):
            self.cores=round(max(self.OPTController(self.stime, self.setpoint, [self.generator.f(t)], self.maxCores), self.min_cores),5)
           
        else:
            self.cores=self.init_cores
    
    def reset(self):
        super().reset()
        self.rtSamples = []
        self.cSamples = []
        self.userSamples = []
    
    def setSLA(self, sla):
        if(not isinstance(sla, list)):
            sla = [sla]
            
        self.sla = sla
        self.setpoint = [s * self.st for s in self.sla]
    
    def __str__(self):
        return super().__str__() + " OPTCTRL: %.2f, l: %.2f h: %.2f " % (self.step, self.l, self.h)

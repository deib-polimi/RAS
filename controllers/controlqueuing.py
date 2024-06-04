from .controller import Controller
import casadi
import numpy as np
from .estimator import QNEstimaator


class OPTCTRL(Controller):
    
    esrimationWindow = 30;
    rtSamples = None
    cSamples = None
    userSamples = None

<<<<<<< Updated upstream
    def __init__(self, period, init_cores, stime, maxCores=1000, st=0.8, name=None):
        super().__init__(period, init_cores, st, name=name)
        if not isinstance(stime, list):
=======
    def __init__(self, period, init_cores, stime, maxCores=100000, st=0.8):
        super().__init__(period, init_cores, st)
        if(not isinstance(stime, list)):
>>>>>>> Stashed changes
            self.stime = [stime]
        self.generator = None
        self.estimator = QNEstimaator()
        self.rtSamples = [[]]
        self.cSamples = [[]]
        self.userSamples = [[]]
        self.maxCores = maxCores
        self.Ik=0
    

    def setServiceTime(self, stime):
        if not isinstance(stime, list):
            self.stime = [stime]
        else:
            self.stime = stime
    # def OPTController(self,e, tgt, C,maxCore):
    #     optCTRL = Model() 
    #     optCTRL.hideOutput()
    #
    #     nApp=len(tgt)
    #
    #     T=[optCTRL.addVar("t%d"%(i), vtype="C", lb=0, ub=None) for i in range(nApp)]
    #     S=[optCTRL.addVar("s%d"%(i), vtype="C", lb=10**-3, ub=maxCore) for i in range(nApp)]
    #     D=[optCTRL.addVar("d%d"%(i), vtype="B") for i in range(nApp)]
    #     E_l1 = [optCTRL.addVar(vtype="C", lb=0, ub=None) for i in range(nApp)]
    #
    #     sSum=0
    #     obj=0;
    #     for i in range(nApp):
    #         sSum+=S[i]
    #         obj+=E_l1[i]/tgt[i]
    #
    #     optCTRL.addCons(sSum<=maxCore)
    #
    #     for i in range(nApp):
    #         optCTRL.addCons(T[i] <= S[i] / e[i])
    #         optCTRL.addCons(T[i] <= C[i] / e[i])
    #         optCTRL.addCons(T[i] >= S[i] / e[i] - C[i] / e[i] * D[i])
    #         optCTRL.addCons(T[i] >= C[i] / e[i] - C[i] / e[i] * (1 - D[i]))
    #         optCTRL.addCons(E_l1[i] >= ((C[i]/T[i])-tgt[i]))
    #         optCTRL.addCons(E_l1[i] >= -((C[i]/T[i])-tgt[i]))
    #
    #
    #     optCTRL.setObjective(obj)
    #
    #     optCTRL.optimize()
    #     sol = optCTRL.getBestSol()
    #     return [sol[S[i]] for i in range(nApp)]
    
    def OPTController(self, e, tgt, C, maxCore):
        #print("stime:=", e, "tgt:=", tgt, "user:=", C)
        if(np.sum(C)>0):
            self.model = casadi.Opti() 
            nApp = len(tgt)
            
            T = self.model.variable(1, nApp);
            S = self.model.variable(1, nApp);
            E_l1 = self.model.variable(1, nApp);
            
            self.model.subject_to(T >= 0)
            self.model.subject_to(self.model.bounded(0, S, maxCore))
            self.model.subject_to(E_l1 >= 0)
        
            sSum = 0
            obj = 0;
            for i in range(nApp):
                sSum += S[0, i]
                # obj+=E_l1[0,i]
                obj += (T[0, i] / C[i] - 1 / tgt[i]) ** 2
            
           # self.model.subject_to(sSum <= maxCore)
        
            for i in range(nApp):
                # optCTRL.addCons(T[i] <= S[i] / e[i])
                # optCTRL.addCons(T[i] <= C[i] / e[i])
                # optCTRL.addCons(T[i] >= S[i] / e[i] - C[i] / e[i] * D[i])
                # optCTRL.addCons(T[i] >= C[i] / e[i] - C[i] / e[i] * (1 - D[i]))
                # optCTRL.addCons(E_l1[i] >= ((C[i]/T[i])-tgt[i]))
                # optCTRL.addCons(E_l1[i] >= -((C[i]/T[i])-tgt[i]))
                self.model.subject_to(T[0, i] == casadi.fmin(S[0, i] / e[i], C[i] / e[i]))
                
            # self.model.subject_to((E_l1[0,i]+tgt[i])>=((C[i]/T[0,i])))
            # self.model.subject_to((E_l1[0,i]-tgt[i])>=-((C[i]/T[0,i])))
        
            self.model.minimize(obj)    
            optionsIPOPT = {'print_time':False, 'ipopt':{'print_level':0}}
            # self.model.solver('osqp',{'print_time':False,'error_on_fail':False})
            self.model.solver('ipopt', optionsIPOPT) 
            
            sol = self.model.solve()
            if(nApp==1):
                return sol.value(S)
            else:
                return sol.value(S).tolist()
        else:
            return 10**(-3)
        
        # optCTRL.optimize()
        # sol = optCTRL.getBestSol()
        # return [sol[S[i]] for i in range(nApp)]
    
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
            print("adding", rt, u, c)
            self.rtSamples.append(rt)
            self.cSamples.append(c)
            self.userSamples.append(u)
        
    def control(self, t):
        if(len(self.monitoring.getAllRTs())>0):
            sIdx=max(len(self.monitoring.getAllRTs())-self.esrimationWindow,0)
            eIdx=min(sIdx+self.esrimationWindow,len(self.monitoring.getAllRTs()))

        self.addRtSample(self.monitoring.getAllRTs()[-1], self.monitoring.getAllUsers()[-1],self.monitoring.getAllCores()[-1])

        mRt = self.monitoring.getAllRTs()[sIdx:eIdx+1]
        mUsers = self.monitoring.getAllUsers()[sIdx:eIdx+1]
        mCores= self.monitoring.getAllCores()[sIdx:eIdx+1]
        
        print(f"rt={len(mRt)},{len(mUsers)},{len(mCores)}")

        # i problemi di stima si possono parallelizzare 
        self.stime[0] = self.estimator.estimate(np.array(self.rtSamples), 
                                                np.array(self.cSamples),
                                                np.array(self.userSamples))

        print(f"###estim {self.stime}")
        
        # risolvo il problema di controllo ottimo
        if(t>0):
            self.Ik+=mRt[-1]-self.setpoint[0]
        
        #print(f"{mRt[-1]}  {mUsers[-1]}  {mCores[-1]}")

        #if(t>self.esrimationWindow):
        self.cores=round(max(self.OPTController(self.stime, self.setpoint, [self.generator.f(t+1)], self.maxCores)+0.1*self.Ik,0.5),3,)
        #else:
        #    self.cores=mUsers[-1]
    
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

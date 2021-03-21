from .controller import Controller
from pyscipopt import Model
import numpy as np


class OPTCTRL(Controller):
    
    def __init__(self, period, init_cores, stime,st=0.8):
        super().__init__(period, init_cores,st)
        self.stime=stime
    
    
    def OPTController(self,e, tgt, C):
        optCTRL = Model()  
        optCTRL.hideOutput()
        t = optCTRL.addVar("t", vtype="C", lb=0, ub=None)
        s = optCTRL.addVar("s", vtype="C", lb=0, ub=None)
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
        
    def control(self, t):
        rt = self.monitoring.getRT()
        self.cores=np.ceil(self.OPTController(self.stime, self.setpoint, int(self.monitoring.getUsers())))

    def __str__(self):
        return super().__str__() + " OPTCTRL: %.2f, l: %.2f h: %.2f " % (self.step, self.l, self.h)

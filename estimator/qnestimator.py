'''
Created on 31 mar 2021

@author: emilio
'''

from pyscipopt import Model
import numpy as np


class QNEstimaator():
    
    model=None
    
    def estimate(self,rt,s,c):
        self.model = Model()  
        self.model.hideOutput()
        e = self.model.addVar("e", vtype="C", lb=0, ub=None)
        er_l1 = self.model.addVar("er_l1", vtype="C", lb=0, ub=None)
        
        if(c<s):
            self.model.addCons(er_l1 >= rt-e)
            self.model.addCons(er_l1 >= -rt+e)
        else:
            self.model.addCons(er_l1 >= rt-(c/s)*e)
            self.model.addCons(er_l1 >= -rt+(c/s)*e)
        
        self.model.setObjective(er_l1)
        self.model.optimize()
        sol = self.model.getBestSol()
        return sol[e]


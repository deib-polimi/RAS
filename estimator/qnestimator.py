'''
Created on 31 mar 2021

@author: emilio
'''
import casadi
class QNEstimaator():
    
    model=None
    
    def estimate(self,rt,s,c):
        self.model = casadi.Opti()
        
        e = self.model.variable(1,1);
        er_l1 = self.model.variable(1,1);
        
        self.model.subject_to(e>=0)
        self.model.subject_to(er_l1>=0)
        
        if(c<s):
            self.model.subject_to(er_l1 >= rt-e)
            self.model.subject_to(er_l1 >= -rt+e)
        else:
            self.model.subject_to(er_l1 >= rt-(c/s)*e)
            self.model.subject_to(er_l1 >= -rt+(c/s)*e)
        
        self.model.minimize(er_l1)    
        optionsIPOPT={'print_time':False,'ipopt':{'print_level':0}}
        self.model.solver('ipopt',optionsIPOPT) 
        
        sol=self.model.solve()
        return sol.value(e)


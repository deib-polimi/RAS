'''
Created on 31 mar 2021

@author: emilio
'''
import casadi
class QNEstimaator():
    
    model=None
    
    def estimate(self,rt,s,c):
        print(len(rt))
        self.model = casadi.Opti()
        
        e = self.model.variable(1,1);
        er_l1 = self.model.variable(1,len(rt));
        
        self.model.subject_to(e>=0)
        self.model.subject_to(er_l1>=0)
        
        obj=0
        for i in range(len(rt)):
            if(c[i]<s[i]):
                self.model.subject_to(er_l1[0,i] >= rt[i]-e)
                self.model.subject_to(er_l1[0,i] >= -rt[i]+e)
            else:
                self.model.subject_to(er_l1[0,i] >= rt[i]-(c[i]/s[i])*e)
                self.model.subject_to(er_l1[0,i] >= -rt[i]+(c[i]/s[i])*e)
            obj+=er_l1[0,i]
        
        self.model.minimize(obj)    
        optionsIPOPT={'print_time':False,'ipopt':{'print_level':0}}
        self.model.solver('ipopt',optionsIPOPT) 
        
        sol=self.model.solve()
        return sol.value(e)


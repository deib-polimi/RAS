import casadi
class QNEstimaator():
    
    model=None
    def estimate(self,think=10**-5,rt=None,s=None,c=None):
        self.model = casadi.Opti()
        
        e = self.model.variable(1,1);
        er_l1 = self.model.variable(1,len(rt));
        
        self.model.subject_to(e>=0)
        self.model.subject_to(er_l1>=0)
        
        obj=0
        for i in range(len(rt)):
            if(c[i]<s[i]):
                self.model.subject_to(er_l1[0,i] >= rt[i]-(e+think))
                self.model.subject_to(er_l1[0,i] >= -rt[i]+(e+think))
            else:
                self.model.subject_to(er_l1[0,i] >= rt[i]-(c[i]/s[i])*(e+think))
                self.model.subject_to(er_l1[0,i] >= -rt[i]+(c[i]/s[i])*(e+think))
            obj+=er_l1[0,i]
        
        self.model.minimize(obj)    
        optionsIPOPT={'print_time':False,'ipopt':{'print_level':0}}
        self.model.solver('ipopt',optionsIPOPT) 
        
        sol=self.model.solve()
        return sol.value(e)
'''
Created on 31 mar 2021

@author: emilio
'''
import casadi
class QNEstimaator():
    
    model=None

    # def estimate(self,rt,s,c):
    #     self.model = casadi.Opti()
    #     #Ti=min(C/(1+e),s/e)
    #     e = self.model.variable(1,1);
    #     err_abs= self.model.variable(1,rt.shape[0])
    #     self.model.set_initial(e,0.0001)
    #     t = self.model.variable(rt.shape[0],1);
    #     self.model.subject_to(e>=0)
    #     self.model.subject_to(t>=0)
    #     #self.model.subject_to(err_abs>=0)
        
    #     obj=0;
    #     for i in range(rt.shape[0]):
    #         self.model.subject_to(t[i,0]==casadi.fmin(c[i]/(0.001+e),s[i]/e))
    #         self.model.subject_to(err_abs[0,i]>=c[i]-(rt[i]+0.001)*t[i,0])
    #         self.model.subject_to(err_abs[0,i]>=-(c[i]-(rt[i]+0.001)*t[i,0]))
    #         #obj+=(c[i]-(rt[i]+0.001)*t[i,0])**2;
    #         obj+=err_abs[0,i]
        
    #     self.model.minimize(obj)    
    #     optionsIPOPT={'print_time':False,'ipopt':{'print_level':0}}
    #     self.model.solver('ipopt',optionsIPOPT) 
        
    #     sol=self.model.solve()
    #     return sol.value(e)
    
    def estimate(self,rt,s,c):
        self.model = casadi.Opti()
        
        # print("rt",rt)
        # print("server",s)
        # print("client",c)
        
        
        e = self.model.variable(1,1);

        er_l1 = self.model.variable(1,rt.shape[0]);

        
        self.model.subject_to(e>=0)
        self.model.subject_to(er_l1>=0)
        
        for i in range(rt.shape[0]):

            if(c[i]<s[i]):
                self.model.subject_to(er_l1[0,i] >= rt[i]-(e+0.001))
                self.model.subject_to(er_l1[0,i] >= -(rt[i]-(e+0.001)))
            else:
                self.model.subject_to(er_l1[0,i] >= rt[i]-(c[i]/s[i])*(e+0.001))
                self.model.subject_to(er_l1[0,i] >= -rt[i]+(c[i]/s[i])*(e+0.001))
        
        self.model.minimize(casadi.sum2(er_l1))    

        optionsIPOPT={'print_time':False,'ipopt':{'print_level':0}}
        self.model.solver('ipopt',optionsIPOPT) 
        #self.model.solver('qpoase') 
        
        sol=self.model.solve()
        return sol.value(e)

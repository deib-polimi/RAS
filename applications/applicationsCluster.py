'''
Created on 12 gen 2021

@author: emilio
'''

import simpy
from scipy.stats import truncnorm
import numpy as np
import uuid
import matplotlib.pyplot as plt
if __name__ == "__main__":
    from application import Application
else:
    from .application import Application

class App():
    serving=None
    backlog=None
    env=None
    nThreads=None
    cpuQuota=None
    name=None
    stime=None
    rTime=None
    stdSt=None
    mSt=None
    isDeterministic=None

    # env: simulation env
    # cpuQuota: core
    # initUsers: list of initial users
    # mST: avg service rate
    # nThreads: application threads 
    # stdST: std service rate
    # isDetermistic: service rate constant (mST) or not
    def __init__(self,env,cpuQuota,name,initUsers,mSt,nThreads=-1,stdSt=None,isDeterministic=False):
        self.env=env
        self.name=name
        self.isDeterministic=isDeterministic
        
        self.cpuQuota=cpuQuota
        self.nThreads=nThreads
        self.mSt=mSt
        self.stdSt=stdSt
            
        self.serving=simpy.Store(env)
        self.backlog=simpy.Store(env)
        
        #settoLostatoIniziale
        for u in initUsers:
            self.backlog.put(u)
        #istanzio un processo per ogni servente che ho
        if(self.nThreads>=1):
            raise ValueError("Finite threads center not implemented yet")
            for i in range(self.nThreads):
                self.env.process(self.serve())
        else:
            self.env.process(self.serve()) 
            
        self.stime={}
        self.rTime=[]
        
    def serve(self):
        while True:
            user=yield self.backlog.get()
            yield self.serving.put(user)
            
            if(not self.nThreads==-1):
                # distgen = get_truncated_normal(mean=1.0/ep.mean, sd=(ep.scv*(1.0/ep.mean)))
                # #yield self.env.timeout(np.random.exponential(1.0/ep.mean))
                # delay=distgen.rvs()
                # if(not ep.name in self.stime):
                #     self.stime[ep.name]=[]
                # self.stime[ep.name].append(delay)
                # yield self.env.timeout(delay)
                # #record Rtime of this center
                # if(user.issueTime is not None):
                #     self.rTime.append(self.env.now-user.issueTime)
                # else:
                #     self.rTime.append(self.env.now)
                #
                # nextep=self.QN.getNextCenter(ep.name)
                # user.endpointName=nextep.name
                # user.issueTime=self.env.now
                #
                # yield self.serving.get()   
                #
                # yield nextep.center.backlog.put(user)
                # #se la chiamata fosse sincrona qui dovrei aspettare la risposta
                raise ValueError("Finite threads center not implemented yet")
                
            else:
                #nel caso di infinite server eseguo solo il compito
                self.env.process(self.doWork(self,user))
                

        
    def doWork(self,app,user):
        if(self.isDeterministic):
            isTime=1.0/app.mSt
        else:
            isTime=np.random.exponential(1.0/app.mSt)
        
        #simluate processor sharing
        d=(isTime)*(len(app.serving.items))/app.cpuQuota
        yield app.env.timeout(np.maximum(d,isTime))
        
        #record Rtime of this center
        if(user.issueTime is not None):
            app.rTime.append(app.env.now-user.issueTime)
        else:
            app.rTime.append(app.env.now)
        
        yield app.serving.get()
        


class appsCluster(Application):
    
    appNames=None
    srateAvg=None
    stdrateAvg=None
    cpuQuotas=None
    users=None
    cluster=None
    horizon=None
    monitoringWindow=None
    isDeterministic=None
    
    def __init__(self,appNames,srateAvg,monitoringWindow,horizon,cpuQuotas,isDeterministic=True):
        self.appNames=appNames
        self.srateAvg=srateAvg
        self.users=None# mi aspetto che il numero di utenti venga passoto come prameetro della funzione __computeRT__
        self.cpuQuotas=cpuQuotas
        self.stdrateAvg=None
        self.monitoringWindow=monitoringWindow
        self.horizon=horizon
        self.isDeterministic=isDeterministic
    
    
    def deployCluster(self,X0,S,MU,Names,std=None):
    
        cluster={};
        cluster["env"]=simpy.Environment()
        cluster["apps"]={};
        
        #dichiaro tutte le applicazioni del cluster
        for i in range(len(Names)):
            initPop=[]
            for k in range(X0[0,i]):
                initPop.append(User(uuid.uuid4()))       
            cluster["apps"][Names[i]]=App(cluster["env"],S[0,i],Names[i],initPop,MU[0,i],isDeterministic=self.isDeterministic)
            
        self.cluster=cluster
    
    #la differenza rispetto a prima e' che mi aspetto che users sia un vettore con un numero di componentni
    #pari a len(self.appNames)
    def __computeRT__(self, users):
        rtime={}
        
        for h in range(self.monitoringWindow):
            self.deployCluster(users, self.cpuQuotas, self.srateAvg, self.appNames, self.stdrateAvg)
            self.cluster["env"].run(until=self.horizon)
            for key,val in enumerate(self.cluster["apps"]):
                if(not val in rtime):
                    rtime[val]=[]
                rtime[val]=rtime[val]+self.cluster["apps"][val].rTime
        
        return rtime
        
        


def get_truncated_normal(mean=0, sd=1, low=0, upp=100):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
class User(object):
    ID=None
    issueTime=None
    ev=None
    
    def __init__(self,ID,issueTime=0):
        self.ID=ID
        self.issueTime=issueTime;
            
    
if __name__ == "__main__":
    #applications names
    Names=["App1","App2","App3"];
    #average service rate per applications
    srateAvg=np.matrix([1,1,1]);
    #numper of users per applications
    X0=np.matrix([1,1,1])
    #reserved cpus quaota per applications
    cpuQuotas=np.matrix([1,1,1])
    #numper of observation windows
    rep=1;
    #width of each observation windows
    T=1.1
    
    cluster=appsCluster(appNames=Names,srateAvg=srateAvg,
                         monitoringWindow=1,horizon=100,
                         cpuQuotas=cpuQuotas,isDeterministic=False)
    
    rtime=cluster.__computeRT__(X0)
    
    fig, axs = plt.subplots(len(rtime),1)
    for key,val in enumerate(rtime):
        print("%s mean=%f max=%f min=%f"%(val,np.mean(rtime[val]),np.max(rtime[val]),np.min(rtime[val])))
        if(len(rtime)>1):
            axs[key].hist(rtime[val],20,density=True, histtype='step',cumulative=True, label='Empirical')    
            axs[key].set_title('ECDF response time of %s'%(val))
        else:
            axs.hist(rtime[val],20,density=True, histtype='step',cumulative=True, label='Empirical')    
            axs.set_title('RT dist %s'%(val))
    
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    #fig.subplots_adjust(hspace=0.5)
    plt.show()
    
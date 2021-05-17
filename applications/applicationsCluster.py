'''
Created on 12 gen 2021

@author: emilio
'''

import simpy
from scipy.stats import truncnorm
import numpy as np
import uuid
import matplotlib.pyplot as plt

class App(object):
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

    def __init__(self,env,cpuQuota,name,initUsers,mSt,nThreads=-1,stdSt=None):
        
        self.env=env
        self.name=name
        
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
        #isTime=np.random.exponential(1.0/app.mSt)
        isTime=1.0/app.mSt
        #simluate processor sharing
        d=(isTime)*(len(app.serving.items))/app.cpuQuota
        yield app.env.timeout(np.maximum(d,isTime))
        #yield app.env.timeout(np.maximum(d,1.0/app.mSt))
        
        #record Rtime of this center
        if(user.issueTime is not None):
            app.rTime.append(app.env.now-user.issueTime)
        else:
            app.rTime.append(app.env.now)
        
        yield app.serving.get()


def get_truncated_normal(mean=0, sd=1, low=0, upp=100):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
class User(object):
    ID=None
    issueTime=None
    ev=None
    
    def __init__(self,ID,issueTime=0):
        self.ID=ID
        self.issueTime=issueTime;
        
def deployCluster(X0,S,MU,Names,std=None):
    
    cluster={};
    cluster["env"]=simpy.Environment()
    cluster["apps"]={};
    
    #dichiaro tutte le applicazioni del cluster
    for i in range(len(Names)):
        initPop=[]
        for k in range(X0[0,i]):
            initPop.append(User(uuid.uuid4()))       
        cluster["apps"][Names[i]]=App(cluster["env"],S[0,i],Names[i],initPop,MU[0,i])
        
    return cluster
            
    
if __name__ == "__main__":
    #numper of users per applications
    X0=np.matrix([10,3])
    #reserved cpus quaota per applications
    cpuQuotas=np.matrix([2.5,0.5])
    #average service rate per applications
    stimesAvg=np.matrix([1.0,1.0]);
    #applications names
    Names=["App1","App2"];
    #numper of observation windows
    rep=1;
    
    rtime={}
    
    for r in range(rep):
        #faccio partire la simulazione per il tempo prestabilito
        cluster=deployCluster(X0,cpuQuotas,stimesAvg,Names);
        cluster["env"].run(until=100)
        for key,val in enumerate(cluster["apps"]):
            if(not val in rtime):
                rtime[val]=[]
            rtime[val]=rtime[val]+cluster["apps"][val].rTime
        
    fig, axs = plt.subplots(len(rtime),1)
    
    for key,val in enumerate(rtime):
        print("%s mean=%f max=%f min=%f"%(val,np.mean(rtime[val]),np.max(rtime[val]),np.min(rtime[val])))
        if(len(rtime)>1):
            axs[key].hist(rtime[val],20,density=True, histtype='step',cumulative=True, label='Empirical')    
            axs[key].set_title('RT dist %s'%(val))
        else:
            axs.hist(rtime[val],20,density=True, histtype='step',cumulative=True, label='Empirical')    
            axs.set_title('RT dist %s'%(val))
    
    plt.show()
    
'''
Created on 12 gen 2021

@author: emilio
'''

import multiprocessing
import time
import uuid

import numpy as np
import redis
from scipy.stats import truncnorm
import simpy

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
    users=None
    stdSt=None
    mSt=None
    tRate=None
    isDeterministic=None
    toAdd=None
    redis=None 
    samplingInterval=0.01
    queueLength=None
    quantum=10**(-4)

    # env: simulation env
    # cpuQuota: core
    # initUsers: list of initial users
    # mST: avg service rate
    # nThreads: application threads 
    # stdST: std service rate
    # isDetermistic: service rate constant (mST) or not
    def __init__(self,env,cpuQuota,name,initUsers,mSt,nThreads=-1,stdSt=None,tRate=None,isDeterministic=False,isOpen=False):
        self.sla=1.0
        self.disturbance=0
        self.env=env
        self.name=name
        self.isDeterministic=isDeterministic
        
        self.cpuQuota=cpuQuota
        self.nThreads=nThreads
        self.mSt=mSt
        self.stdSt=stdSt
        self.tRate=tRate
        self.toAdd=0
        self.redispool=redis.ConnectionPool(host="localhost")
        self.queueLength=[]
        
        self.serving=simpy.Store(self.env)
        self.backlog=simpy.Store(self.env)
        self.swThreads=simpy.Resource(self.env,capacity=1)
        
        print(isOpen)
        
        #settoLostatoIniziale
        if(not isOpen):
            for u in initUsers:
                self.backlog.put(u)
        else:
            self.tRate=initUsers
            self.env.process(self.think()) 
        
        self.env.process(self.serve()) 
            
        self.stime={}
        self.rTime=[]
        self.users=[]
        
        
        
    def serve(self):
        while True:
            #richiedo il thread software
            req=self.swThreads.request()
            yield req
            #recupero l'utente
            user=yield self.backlog.get()    
            self.serving.put(user)
            user.action=self.env.process(self.doWork(user,req))
            self.interruptExecution()
        
        
    def doWork(self,user,req):
        if(self.isDeterministic):
            isTime=1.0/self.mSt
        else:
            isTime=np.random.exponential(1.0/self.mSt)
            #isTime=sample_G_Stime(1.0/self.mSt,1.01/self.mSt,1)[0]
            
        
        isdone=False
        sf=None
        startExec=None
        while(not isdone and isTime>0):
            try:
                startExec=self.env.now
                d=(isTime)*(len(self.serving.items))/self.cpuQuota  
                sf=np.maximum(d,isTime)/isTime      
                yield self.env.timeout(np.maximum(d,isTime))
                isdone=True
            except simpy.Interrupt:
                isTime-=(self.env.now-startExec)/sf
            
        self.swThreads.release(req)
        self.serving.items.remove(user)
        self.interruptExecution() 
    
        #record Rtime of this center
        if(user.issueTime is not None):
            self.rTime.append(self.env.now-user.issueTime)
        else:
            self.rTime.append(self.env.now)
    
    
    def think(self):
        while True:
            yield self.env.timeout(np.random.exponential(1.0/self.tRate))
            yield self.backlog.put(User(uuid.uuid4(),issueTime=self.env.now))
        
    def sampleQueueLength(self):
        while True:
            yield self.env.timeout(self.samplingInterval)
            self.queueLength.append(len(self.serving.items)+len(self.backlog.items))
    
    def sampleRT(self,resetData=True):
        rt=np.mean(self.rTime)
        if(resetData):
            self.rTime=[]
        return rt
        
        
    def sampleQueue(self):
        return len(self.backlog.items)+len(self.serving.items)
    
    def interruptExecution(self):
        newc=int(np.ceil(self.cpuQuota*1.2))
        for u in self.serving.items:
            u.action.interrupt()
            
        if(newc>self.swThreads.capacity):
            self.swThreads._capacity=newc
        elif(newc<self.swThreads.count):
            self.swThreads._capacity=newc

    def getSwThreads(self):
        return self.swThreads.capacity


class AppsCluster(Application):
    appNames=None
    srateAvg=None
    stdrateAvg=None
    cores=None
    users=None
    horizon=None
    monitoringWindow=None
    isDeterministic=None
    cluster=None
    rdb=None
    env=None
    X0=None
    
    def __init__(self,appNames,srateAvg,initCores, isDeterministic=True,monitoringWindow=1,horizon=None,X0=None,env=None):
        self.init_cores = initCores
        self.disturbance=0
        self.appNames=appNames
        self.srateAvg=srateAvg
        self.users=None# mi aspetto che il numero di utenti venga passoto come prameetro della funzione __computeRT__
        self.cores=initCores
        self.stdrateAvg=None
        self.monitoringWindow=monitoringWindow
        self.horizon=horizon
        self.isDeterministic=isDeterministic
        self.env=env
        self.X0=X0
        
        self.cluster={}
        self.deployCluster(X0, self.cores, self.srateAvg, self.appNames, self.stdrateAvg)
        
        #startCluster(self.cluster)
        
        # p=multiprocessing.Process(target=startCluster, args=(self.cluster,))
        # p.start()
    
    
    def deployCluster(self,X0,S,MU,Names,std=None):
    
        #self.cluster["env"]=simpy.rt.RealtimeEnvironment(factor=1.0,strict=True)
        #self.cluster["env"]=simpy.Environment()
        self.cluster["env"]=self.env
        self.cluster["apps"]={};
        
        #dichiaro tutte le applicazioni del cluster
        for i in range(len(Names)):
            initPop=[]
            # for k in range(X0[i]):
            #     initPop.append(User(uuid.uuid4()))       
            self.cluster["apps"][Names[i]]=App(self.cluster["env"],S[i],Names[i],X0[i],MU[i],isDeterministic=self.isDeterministic,isOpen=True)

        
    
    #la differenza rispetto a prima e' che mi aspetto che users sia un vettore con un numero di componentni
    #pari a len(self.appNames)
    def __computeRT__(self, users):
        rtime=np.zeros([len(self.appNames)])
        if(self.rdb is None):
            rdb=redis.Redis()
        
        for key,val in enumerate(self.cluster["apps"]):
            #aggiorno il think rate dell'applicazione
            rdb.set("%s_u2add"%(val),users[key])
            #campiono il response time misrurato per questa applicazione
            rtime[key]=float(rdb.get("%s_rt"%(val)))
            #aggiorno il numero di core calcolati dal controllore
            rdb.set("%s_quota"%(val),"%.4f"%(self.cores[key]))
        
        print(rtime)
            
        return rtime
    
    def sampleQueue(self):
        return [self.cluster["apps"][val].sampleQueue() for key,val in enumerate(self.cluster["apps"])]
    
    def sampleRT(self,reset=True):
        return [self.cluster["apps"][val].sampleRT(reset) for key,val in enumerate(self.cluster["apps"])]
    
    def getRT(self):
        return self.sampleRT(False)
    
    def updateCores(self,Cores):
        for key,val in enumerate(self.cluster["apps"]):
            self.cluster["apps"][val].cpuQuota=Cores[key]
            self.cluster["apps"][val].interruptExecution()
    
    def updateTRate(self,tRate):
        self.X0=tRate
        for key,val in enumerate(self.cluster["apps"]):
            self.cluster["apps"][val].tRate=tRate[key]
            
    def getSwThreads(self):
        return [self.cluster["apps"][val].getSwThreads() for key,val in enumerate(self.cluster["apps"])]
        


def sample_G_Stime(X,std,nsamples):    
    cx=std/X
    
    k=None
    A=None
    a=None
    
    if(cx<=1):
        raise ValueError("low variance distribution not implemented yet")     
    
    else:
        k=2
        
        mu1=2.0/X
        mu2=1.0/(X*cx**2)
        
        p1=1.0/(2*cx**2)
        
        A=np.matrix([[-mu1,mu1*p1],
                     [0,-mu2]])
        
        a=np.matrix([[1,0]])
    
    
    x = SamplesFromPH(a, A, nsamples)
    return x

        
class User(object):
    ID=None
    issueTime=None
    ev=None
    
    def __init__(self,ID,issueTime=0):
        self.ID=ID
        self.issueTime=issueTime;
        

    
if __name__ == "__main__":
    rdb=redis.Redis()
    #applications names
    Names=["App1"];
    #average service rate per applications
    srateAvg=[1.0/0.2, 1/0.4];
    #arrival rates per applications
    X0=[10]
    #reserved cpus quaota per applications
    cores=[2]
    
    HCores=[]
    Huser=[]
    Hrt=[]
    
    env=simpy.Environment()
    cluster = AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=cores,isDeterministic=False,X0=X0,env=env)
    
    plt.figure()
    for key,val in enumerate(cluster.cluster["apps"]):
        print(np.mean(cluster.cluster["apps"][val].queueLength))
        print(len(cluster.cluster["apps"][val].queueLength))
        plt.plot(cluster.cluster["apps"][val].queueLength)
    plt.show()

    
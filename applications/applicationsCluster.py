'''
Created on 12 gen 2021

@author: emilio
'''

import simpy.rt
from scipy.stats import truncnorm
import numpy as np
import uuid
import matplotlib.pyplot as plt
import multiprocessing
import time
import redis
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
    tRate=None
    isDeterministic=None
    toAdd=None
    sharedData=None
    redis=None
    samplingInterval=30
    thinktime=1

    # env: simulation env
    # cpuQuota: core
    # initUsers: list of initial users
    # mST: avg service rate
    # nThreads: application threads 
    # stdST: std service rate
    # isDetermistic: service rate constant (mST) or not
    def __init__(self,env,cpuQuota,name,initUsers,mSt,nThreads=-1,stdSt=None,tRate=1.0,sharedData=None,isDeterministic=False):
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
        
        self.sharedData=sharedData
        
        self.serving=simpy.Store(env)
        self.backlog=simpy.Store(env)
        
        #settoLostatoIniziale
        for u in initUsers:
            self.backlog.put(u)
        
        self.env.process(self.serve()) 
            
        self.stime={}
        self.rTime=[]
        
        #start monitoring
        self.env.process(self.updateCpuQuota())
        self.env.process(self.sampleRT())
        self.env.process(self.addUser())
        
        
    def serve(self):
        while True:
            user=yield self.backlog.get()
            yield self.serving.put(user)
            
            self.env.process(self.doWork(self,user))
            
        
    def doWork(self,app,user):
        
        redis_conn = redis.Redis(connection_pool=self.redispool)
        
        if(self.isDeterministic):
            isTime=1.0/app.mSt
        else:
            isTime=np.random.exponential(1.0/app.mSt)
        
        #simluate processor sharing
        d=(isTime)*(len(app.serving.items))/app.cpuQuota
        yield self.env.timeout(np.maximum(d,isTime))
        
        #record Rtime of this center
        if(user.issueTime is not None):
            self.rTime.append(app.env.now-user.issueTime)
        else:
            self.rTime.append(app.env.now)
        
        yield self.serving.get()
        redis_conn.close()
    
    def updateCpuQuota(self):
        redis_conn = redis.Redis(connection_pool=self.redispool)
        while True:
            quota=redis_conn.get("%s_quota"%(self.name))
            if(quota is not None):
                self.cpuQuota=float(quota)
            else:
                raise ValueError("No cpu Quota speciied for application %s"%(self.name))
            yield self.env.timeout(self.samplingInterval/2)
        redis_conn.close()
    
    def sampleRT(self):
        redis_conn = redis.Redis(connection_pool=self.redispool)
        while True:
            yield self.env.timeout(self.samplingInterval)
            redis_conn.set("%s_rt"%(self.name),str(np.mean(self.rTime)))
            self.rTime=[]
        redis_conn.close()
    
    #simulo la presenza di un think rate variabile e controllabile dall'esterno
    def addUser(self):
        redis_conn = redis.Redis(connection_pool=self.redispool)
        while True:
            u2add=redis_conn.get("%s_u2add"%(self.name))
            if(u2add is not None):
                try:
                    u2add=int(u2add)
                    #redis_conn.set("%s_u2add"%(self.name),0)
                    for i in range(u2add):
                        yield self.backlog.put(User(uuid.uuid4(),issueTime=self.env.now))
                except:
                    raise ValueError("invalid number of users for application %s"%(self.name))
            else:
                raise ValueError("invalid number of users for application %s"%(self.name))
            yield self.env.timeout(self.thinktime)
        redis_conn.close()

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
    
    def __init__(self,appNames,srateAvg,initCores, isDeterministic=True,monitoringWindow=1,horizon=None):
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
        
        self.cluster={}
        self.sharedData=multiprocessing.Manager().dict()
        
        self.deployCluster([0]*len(self.appNames), self.cores, self.srateAvg, self.appNames, self.stdrateAvg)
        
        rdb=redis.Redis()
        
        for i in range(len(self.appNames)):
            rdb.set("%s_quota"%(self.appNames[i]),"%d"%(self.cores[i]))
            rdb.set("%s_u2add"%(self.appNames[i]),0)
            rdb.set("%s_rt"%(self.appNames[i]),0)
        
        rdb.close()
        
        p=multiprocessing.Process(target=startCluster, args=(self.cluster,))
        p.start()
    
    
    def deployCluster(self,X0,S,MU,Names,std=None):
    
        self.cluster["env"]=simpy.rt.RealtimeEnvironment(factor=1)
        #self.cluster["env"]=simpy.Environment()
        self.cluster["apps"]={};
        
        #dichiaro tutte le applicazioni del cluster
        for i in range(len(Names)):
            initPop=[]
            for k in range(X0[i]):
                initPop.append(User(uuid.uuid4()))       
            self.cluster["apps"][Names[i]]=App(self.cluster["env"],S[i],Names[i],initPop,MU[i],sharedData=self.sharedData,isDeterministic=self.isDeterministic)

        
    
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
            rdb.set("%s_quota"%(val),"%d"%(self.cores[key]))
            
        print(rtime)
        return rtime
    
    def reset(self): pass
        


def get_truncated_normal(mean=0, sd=1, low=0, upp=100):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
class User(object):
    ID=None
    issueTime=None
    ev=None
    
    def __init__(self,ID,issueTime=0):
        self.ID=ID
        self.issueTime=issueTime;
        

def startCluster(cluster):
    cluster["env"].run()
    
            
    
if __name__ == "__main__":
    rdb=redis.Redis()
    #applications names
    Names=["App1","App2"];
    #average service rate per applications
    srateAvg=[1/0.1, 1/0.4];
    #numper of users per applications
    #reserved cpus quaota per applications
    cores=[20,20]
    
    cluster=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=cores,isDeterministic=False)
    while(True):
        cluster.__computeRT__([10,10])
        time.sleep(32)
        
        
    rdb.close()

    
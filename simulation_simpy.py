from generators import *
from controllers import *
from applications import AppsCluster
from math import ceil
from monitoring import Monitoring, MultiMonitoring
import numpy as np
from commons import SN1, SN2, SP2, SP3, RP1, RP2, ALL
import sys
import simpy.rt
from controllers import qnTransient
import casadi
import matplotlib.pyplot as plt
from pathlib import Path
import os
import collections

cwd=os.path.dirname(os.path.abspath(__file__))

class systemMnt():
    rt = None
    
    def __init__(self):
        self.rt=collections.deque(maxlen=1)
        
    def getRT(self):
        if(len(self.rt)==0):
            return None
        else:
            return  [np.mean(self.rt)]
        

def optCtrl(optQueue,opt2Queue,P,MU,S,H,tgt,optdt):
    sold=None
    
    opt=qnTransient()
    opt.buildOpt(MU, P, S,optdt,tgt,H)
    
    while(True):
            
        q=optQueue.get()
        opt.model.set_value(opt.initX,q)
        
        obj=0
        for h in range(H):
            obj+=(opt.stateVar[1,h]-tgt/MU[1,0]*opt.tVar[1,h])**2
            #obj+=opt.E_abs[0,h]
        if(sold is not None):
            opt.model.minimize(obj+0.0*casadi.sumsqr(opt.sVar-sold)+0.000*casadi.sumsqr(opt.sVar))
        else:
            opt.model.minimize(obj)
        
        sol=opt.model.solve()
        opt2Queue.put(sol.value(opt.sVar))
        
    

#control loop
def controlLoop(env,cluster,dt):
    global optQueue,opt2Queue,optProc,tgt,optdt,H,toUpdate,cores_i
    
    #idx=1.0/10
   
    # if(optQueue==None):
    #     optQueue = multiprocessing.Queue()
    #     opt2Queue = multiprocessing.Queue()
    #
    #     P = np.matrix([[0,1,0],[0,0,1],[0,0,1]])
    #     MU = np.matrix([cluster.X0[0],cluster.srateAvg[0],0]).T
    #     S = np.matrix([0, -1,0]).T
    #
    #     optProc = multiprocessing.Process(target=optCtrl, args=(optQueue,opt2Queue,P,MU,S,H,tgt,optdt,))
    #     optProc.start()
        
    while True:
        yield env.timeout(dt)
        # if(toUpdate):
            # toUpdate=False
            # optProc.kill()
            # optQueue = multiprocessing.Queue()
            # opt2Queue = multiprocessing.Queue()
            #
            # P = np.matrix([[0,1,0],[0,0,1],[0,0,1]])
            # MU = np.matrix([cluster.X0[0],cluster.srateAvg[0],0]).T
            # S = np.matrix([0, -1,0]).T
            #
            # optProc = multiprocessing.Process(target=optCtrl, args=(optQueue,opt2Queue,P,MU,S,H,tgt,optdt,))
            # optProc.start()
            
        q=cluster.sampleQueue()[0]
        
        #opt control with optimaztion
        # optQueue.put([0,q,0])
        #
        # opts=opt2Queue.get()
        # print(opts)
        # cluster.cores=[opts[1]]
        #
        
        #opt control
        cluster.cores=[max(q/tgt,0.0001)]
        cluster.updateCores(cluster.cores)
        
        # if(mnt.getRT() is not None):
        #     print("Control",cluster.cores,mnt.getRT(),env.now)
        #     c1.control(env.now)
        #     cluster.cores=c1.cores
        #     cluster.updateCores(cluster.cores)
        
        cores_i.append(cluster.cores[0])
        threads_i.append(cluster.getSwThreads()[0])
        if(len(RT)>1):
            T=np.linspace(0,len(RT),len(RT))
            IAvg=np.divide(np.cumsum(RT),T)
            print("Control",cluster.cores,mnt.getRT(),env.now,IAvg[-1])
        

def monitoringLoop(env,cluster,dt):
    global mnt,MU,users,cores,cores_i,RT,threads_i
    while True:
        yield env.timeout(dt)
        rts=cluster.sampleRT(True)
        if(not np.isnan(rts)):
            mnt.rt.append(rts)
            RT.append(rts)
            users.append(cluster.X0)
            cores.append(np.mean(cores_i))
            cores_i=[]
            threads.append(np.mean(threads_i))
            threads_i=[]
            #print("Monitoring",cluster.cores,mnt.getRT(),users[-1],env.now)


def simulation(env,cluster,dt,g):
    global toUpdate,it
    it=0
    while True:
        print("update trate",g.f(it))
        print(env.now*100/(mtDt*simStep))
        cluster.updateTRate(g.f(it))
        cluster.sampleRT(True)
        toUpdate=True
        yield env.timeout(dt)
        it+=1


users=[]
cores=[]
cores_i=[]
threads=[]
threads_i=[]
RT=[]
tgt=3.0
H=10
sold=None
holdingTime=100
changePoints=80
simStep=changePoints*holdingTime
optQueue=None
opt2Queue=None
optProc=None
toUpdate=False

#optdt=10**(-1) # dt all'interno del problema di ottimo
mtDt=10**(-1)
optdt=10**(-1)
ctrlPeriod=optdt
#applications names
Names=["App1"];
#average service rate per applications
srateAvg=[10];
#arrival rates per applications
X0=[1]
#reserved cpus quaota per applications
cores_init=[1]
#monitor object
mnt=systemMnt()

#workload generator
g = MultiGenerator([SN1])
c1 = CTControllerScaleXNode(ctrlPeriod, cores_init, 100, BCs=[0.05], DCs=[0.05])
c1.cores=cores_init
c1.setSLA([tgt*1/srateAvg[0]])
c1.monitoring=mnt

HCores=[]
Huser=[]
Hrt=[]

env=simpy.rt.RealtimeEnvironment(factor=1.2,strict=True)
#env=simpy.Environment()
cluster=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=cores_init,isDeterministic=False,X0=X0,env=env)
env.process(simulation(env,cluster,holdingTime*mtDt,g))
env.process(monitoringLoop(env,cluster,mtDt))
env.process(controlLoop(env,cluster,ctrlPeriod))
env.run(until=mtDt*simStep)

Path("%s/experiments/"%(cwd)).mkdir(parents=True, exist_ok=True)

T=np.linspace(0,len(RT),len(RT))
IAvg=np.divide(np.cumsum(RT),T)

plt.figure()
plt.plot(RT)
plt.axhline(y = tgt*1.0/srateAvg[0], color = 'r', linestyle = '--')
#plt.ylim([0,tgt*5/srateAvg[0]])
plt.plot(IAvg)
plt.savefig("./experiments/rt.pdf")



fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(users, color="r",label="#arate(req/s)")
ax.set_xlabel('time (k*30s)')
ax.set_ylabel('Workload (req/s)')
ax.legend(loc="upper left")

ax2.plot(cores, color="b",label="#hw_cores")
ax2.plot(threads, color="y",label="#sw_cores")
ax2.set_ylabel('Allocation')
ax2.legend(loc="upper right")
plt.savefig("./experiments/wrall.pdf")

plt.show()

#print(np.mean(RT))

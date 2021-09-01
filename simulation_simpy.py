from generators import *
from controllers import *
from applications import AppsCluster
from math import ceil
from monitoring import Monitoring, MultiMonitoring
import numpy as np
from commons import SN1, SN2, SP2, RP1, RP2, ALL
import sys
import simpy
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
        self.rt=collections.deque(maxlen=50)
        
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
        # optQueue.put([0,q,0])
        #
        # opts=opt2Queue.get()
        #print(opts)
        # cluster.cores=[opts[1]]
        
        # cluster.cores=[max(q/tgt,0.0001)]
        # cluster.updateCores(cluster.cores)
        # cores_i.append(cluster.cores[0])
        
        if(mnt.getRT() is not None):
            c1.control(env.now)
            cluster.cores=c1.cores
            cluster.updateCores(c1.cores)
            cores_i.append(c1.cores)
        # else:
        #     print("reset controller")
        #      cluster.sampleRT(True)
        #      c1.reset()
        #     toUpdate=False
        

def monitoringLoop(env,cluster,dt):
    global mnt,MU,users,cores,cores_i,RT
    while True:
        yield env.timeout(dt)
        rts=cluster.sampleRT(True)
        if(not np.isnan(rts)):
            mnt.rt.append(rts)
            RT.append(rts)
            users.append(cluster.X0)
            cores.append(np.mean(cores_i))
            cores_i=[]
            print("Monitoring",cluster.cores,mnt.getRT(),users[-1],env.now)


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
RT=[]
tgt=3.0
H=10
sold=None
holdingTime=300
changePoints=1
simStep=changePoints*holdingTime
optQueue=None
opt2Queue=None
optProc=None
toUpdate=False

optdt=10**(-1) # dt all'interno del problema di ottimo
mtDt=30
ctrlPeriod=optdt
#applications names
Names=["App1"];
#average service rate per applications
srateAvg=[10];
#arrival rates per applications
X0=[1]
#reserved cpus quaota per applications
cores_init=[10]
#monitor object
mnt=systemMnt()

#workload generator
g = MultiGenerator([SN1])
c1 = CTControllerScaleXNode(1, cores_init, 100, BCs=[10**-3], DCs=[10**-3])
c1.cores=cores_init
c1.setSLA([tgt*1/srateAvg[0]])
c1.monitoring=mnt

HCores=[]
Huser=[]
Hrt=[]

#env=simpy.rt.RealtimeEnvironment(factor=1.0,strict=True)
env=simpy.Environment()
cluster=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=cores_init,isDeterministic=True,X0=X0,env=env)
env.process(simulation(env,cluster,holdingTime*mtDt,g))
env.process(monitoringLoop(env,cluster,mtDt))
env.process(controlLoop(env,cluster,ctrlPeriod))
env.run(until=mtDt*simStep)

Path("%s/experiments/"%(cwd)).mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(RT)
plt.axhline(y = tgt*1.0/srateAvg[0], color = 'r', linestyle = '--')
plt.savefig("./experiments/rt.pdf")

# plt.figure()
# plt.plot(users)
# plt.ylim(0,np.max(users)*1.5)
# plt.savefig("./experiments/workload.pdf")
#
# plt.figure()
# plt.plot(cores)
# plt.ylim(0,np.max(cores)*1.5)
# plt.savefig("./experiments/allocation.pdf")
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(users, color="r")
ax.set_xlabel('time (k*30s)')
ax.set_ylabel('Workload')

ax2.plot(cores, color="b")
ax2.set_ylabel('Allocation')
plt.savefig("./experiments/wrall.pdf")

#plt.show()

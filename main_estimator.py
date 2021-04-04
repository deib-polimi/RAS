'''
Created on 30 mar 2021

@author: emilio
'''
from generators import *
from controllers import *
from runner import Runner
from applications import Application1
import numpy as np
from matplotlib import pyplot as plt
from estimator import QNEstimaator

appSLA = 0.6
app=Application1(appSLA)
qnes=QNEstimaator()



npoint=1200;

app.cores=1;
rt=np.zeros([npoint,1])
rt2=np.zeros([npoint,1])
st=np.zeros([npoint,1])

for i in range(npoint):
    rt[i,0]=app.__computeRT__(i)
    rt2[i,0]=i/rt[i,0]
    st[i,0]=qnes.estimate(rt[i,0], app.cores, i)
    #print(rt[i,0],rt2[i,0])
    #print(qnes.estimate(rt[i,0], 1, i))

print(np.mean(st))    
    
plt.figure()
plt.plot(rt)
plt.show()
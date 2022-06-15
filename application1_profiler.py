from applications import Application1
import numpy as np
import matplotlib.pyplot as plt

app=Application1(sla=0.3, disturbance=0.0, init_cores=1)
RT=[]
for i in range(1,10000):
    app.cores=i
    RT.append(app.__computeRT__(i))
    
plt.figure()
plt.plot(RT)
plt.show()
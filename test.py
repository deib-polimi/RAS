from applications import *
from matplotlib import pyplot as plt
import numpy as np 



app = SockShopMicroservice(target_rt=0.03, target_req=100, target_cores=1.1, sla=1)

app.cores = 1.1
xs = range(10000)
ys = [app.setRT(x) for x in xs]

print(app.setRT(100))
plt.figure()
plt.plot(xs, ys)
plt.show()

if __name__ == "__main__":
	from application import Application
else:
	from .application import Application

import simpy
import random
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from tqdm import tqdm

class ProcessorSharing(simpy.Resource):
    def __init__(self, env, capacity=1):
        super().__init__(env, capacity)

    def request(self, *args, **kwargs):
        # Insert the new request at the end of the queue
        request = super().request(*args, **kwargs)
        if(request in self.queue):
            self.queue.remove(request)  # Remove from default position
            self.queue.append(request)  # Add to the end
        return request

    def release(self, *args, **kwargs):
        return super().release(*args, **kwargs)

class applicationMMC(Application):

	userId=None

	def __init__(self,sla=1.0, disturbance=0.0,stime=1.0,init_cores=1,tag=None):
		super().__init__(sla,disturbance,init_cores)
		self.cores=init_cores
		self.stime=stime
		self.model=None
		self.tree=None
		self.arrivals_time={}
		self.departure_time={}
		self.userId=0
		self.tag=tag
		self.reset()

	def __computeRT__(self, req):
		self.env = simpy.Environment()
		self.server = ProcessorSharing(self.env, capacity=self.cores)
		self.coxian_params = None  # List of tuples: (probability, rate)
		self.service_rate = 1.0/self.stime
		self.queue_length = 0
		self.num_customers = int(req)
		self.rtime=[]
		self.quanto=0.001
		
		self.run(until=100000)

		return np.mean(self.rtime)

	def coxian_arrival(self):
		time = 0
		for prob, rate in self.coxian_params:
			time += np.random.exponential(1.0/rate)
			if random.random() < (1-prob):
				break
		return time


	def customer(self):
		#while True:
		uid=self.userId
		self.userId+=1

		if(self.tag is not None and self.userId>self.tag):
			yield self.env.timeout(np.random.exponential(1.0))
		else:
			pass
			#print(self.userId)

		self.arrivals_time[f"U{uid}"]=float(self.env.now)
		
		self.queue_length += 1
		work=np.random.exponential(1/self.service_rate)
		while work>0:
			srv = self.server.request()
			yield srv
			self.queue_length-=1
			yield self.env.timeout(self.quanto)
			self.server.release(srv)
			work-=self.quanto
		self.departure_time[f"U{uid}"]=self.env.now
		self.rtime+=[self.departure_time[f"U{uid}"]-self.arrivals_time[f"U{uid}"]]


	def run(self,until=1000000):
		for u in range(self.num_customers):
			self.env.process(self.customer()) 
		self.env.run(until=until)



if __name__ == '__main__':
	nusers=100
	#factors=np.linspace(1,100,20,dtype=int);
	#factors=[4]
	#for k in tqdm(factors):
	appMMC=applicationMMC(sla=1.0,stime=1.0,init_cores=10)
	rt=appMMC.__computeRT__(req=nusers)
	print(rt)
	#savemat(f"notag_{k}.mat",{"samples":rt,"arrivals_time":appMMC.arrivals_time,"departure_time":appMMC.departure_time})
	# plt.figure()
	# plt.hist(rt)
	# plt.show()
	#print(np.divide(np.mean(rt[]),np.linspace(0,len(rt),len(rt))))
	#nmean=[np.cumsum(rt)/np.linspace(0,len(rt),len(rt))]
	#print(nmean[0])
	#plt.figure()
	#plt.plot(rt)
	#plt.plot(nmean[0])
	#plt.hlines(0.5,0,len(rt))
	#plt.show()


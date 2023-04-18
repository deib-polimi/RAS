import copy

import networkx as nx

from applications.cartsdelete import CartsDelete
from applications.cartsutil import CartsUtil
from applications.order import Order
from applications.cartscatalogue import CartsCatalogue
from applications.payment import Payment
from applications.shipping import Shipping
from applications.user import User
from dag import DAG
from generators import *
from controllers import *
from applications import Application1
from monitoring import Monitoring
from node import Node
from simulationdagvrs1 import SimulationWithDependeciesDAG

stime = 0.6 #0.9*0.6/4
appSLA = stime
horizon = 1200
monitoringWindow = 10
initCores = 1
period = 1
st=1

c0 = CTControllerScaleX(period, initCores, st=st); c0.setName("ScaleX")
c1 = CTControllerScaleX(period, initCores, st=st); c1.setName("ScaleX")
c2 = CTControllerScaleX(period, initCores, st=st); c2.setName("ScaleX")
c3 = CTControllerScaleX(period, initCores, st=st); c3.setName("ScaleX")
c4 = CTControllerScaleX(period, initCores, st=st); c4.setName("ScaleX")
c5 = CTControllerScaleX(period, initCores, st=st); c5.setName("ScaleX")
c6 = CTControllerScaleX(period, initCores, st=st); c6.setName("ScaleX")

# c0 = CTControllerScaleX(period, initCores); c0.setName("ScaleX")
# c1 = CTControllerScaleX(period, initCores); c1.setName("ScaleX")
# c2 = CTControllerScaleX(period, initCores); c2.setName("ScaleX")
# c3 = CTControllerScaleX(period, initCores); c3.setName("ScaleX")

#appSLA = 0.9*0.6/4

# apps = [Order(appSLA, init_cores=initCores), CartsCatalogue(appSLA, init_cores=initCores), Shipping(appSLA, init_cores=initCores),
#         User(appSLA, init_cores=initCores), Payment(appSLA, init_cores=initCores), CartsUtil(appSLA, init_cores=initCores),
#         CartsDelete(appSLA, init_cores=initCores)]

apps = [Order(appSLA, init_cores=initCores), CartsCatalogue(appSLA/3, init_cores=initCores), Shipping(appSLA/12, init_cores=initCores),
        User(appSLA/12, init_cores=initCores), Payment(appSLA/12, init_cores=initCores), CartsUtil(appSLA/3, init_cores=initCores),
        CartsDelete(appSLA/3, init_cores=initCores)]

# apps = [Order(0.3, init_cores=initCores), CartsCatalogue(0.07, init_cores=initCores), Shipping(0.07, init_cores=initCores),
#         User(0.07, init_cores=initCores), Payment(0.07, init_cores=initCores), CartsUtil(0.07, init_cores=initCores),
#         CartsDelete(0.07, init_cores=initCores)]

#apps = [Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores)]

mns = [Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA),
       Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA),
       Monitoring(monitoringWindow, appSLA)]

#mns = [Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA)]

c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla); c2.setSLA(apps[2].sla); c3.setSLA(apps[3].sla);
c4.setSLA(apps[4].sla); c5.setSLA(apps[5].sla); c6.setSLA(apps[6].sla)
# f = c0.control
# def f2(t):
#     f(t)
#     print(c0.rt)
# c0.control = f2
c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1]); c2.setMonitoring(mns[2]); c3.setMonitoring(mns[3]);
c4.setMonitoring(mns[4]); c5.setMonitoring(mns[5]); c6.setMonitoring(mns[6])

# c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla); c2.setSLA(apps[2].sla); c3.setSLA(apps[3].sla)
#
# c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1]); c2.setMonitoring(mns[2]); c3.setMonitoring(mns[3])

g0 = RampGen(1, 90, 10); g0.setName("RP - Order")
g1 = ZeroGen(); g1.setName("ZG - Catalogue")
g2 = ZeroGen(); g2.setName("ZG - Shipping")
g3 = RampGen(2, 90, 20); g3.setName("RP - User")
g4 = ZeroGen(); g4.setName("ZG - Payment")
g5 = ZeroGen(); g5.setName("ZG - Carts-Util")
g6 = RampGen(1, 90, 10); g6.setName("RP - CartsDelete")

# g0 = SinGen(500, 700, 200); g0.setName("SN1 - App 1")
# g1 = RampGen(10, 800); g1.setName("RP1 - App 2")
# g2 = SinGen(500, 700, 200); g2.setName("SN1 - App 3")
# g3 = RampGen(10, 800); g3.setName("RP1 - App 4")

c0.setGenerator(g0); c1.setGenerator(g1); c2.setGenerator(g2); c3.setGenerator(g3); c4.setGenerator(g4)
c5.setGenerator(g5); c6.setGenerator(g6)

#c0.setGenerator(g0); c1.setGenerator(g1); c2.setGenerator(g2); c3.setGenerator(g3)


dg = nx.DiGraph([("Ord", "Cat", {"sync": 1}), ("Ord", "Shi", {"sync": 1}), ("Ord", "Use", {"sync": 1}),
                 ("Ord", "Pay", {"sync": 1}), ("Ord", "Uti", {"sync": 2}), ("Ord", "Del", {"sync": 3})])

#dg = nx.DiGraph([("A", "B", {"sync": 1}), ("A", "C", {"sync": 2}), ("B", "D", {"sync": 1}), ("C", "D", {"sync": 1})])

# nx.set_node_attributes(dg, dg.nodes, 'total_node_req')
nx.set_node_attributes(dg, dg.nodes, 'node')
nx.set_node_attributes(dg, dg.nodes, 'users')
nx.set_node_attributes(dg, dg.nodes, 'rt')
#nx.set_edge_attributes(dg, dg.edges, 'req')

dg.nodes['Ord']['node'] = Node(horizon, c0, apps[0], monitoring=mns[0], name='Ord', generator=g0)
dg.nodes['Cat']['node'] = Node(horizon, c1, apps[1], monitoring=mns[1], name='Cat', generator=g1)
dg.nodes['Shi']['node'] = Node(horizon, c2, apps[2], monitoring=mns[2], name='Shi', generator=g2)
dg.nodes['Use']['node'] = Node(horizon, c3, apps[3], monitoring=mns[3], name='Use', generator=g3)
dg.nodes['Pay']['node'] = Node(horizon, c4, apps[4], monitoring=mns[4], name='Pay', generator=g4)
dg.nodes['Uti']['node'] = Node(horizon, c5, apps[5], monitoring=mns[5], name='Uti', generator=g5)
dg.nodes['Del']['node'] = Node(horizon, c6, apps[6], monitoring=mns[6], name='Del', generator=g6)

# dg.nodes['A']['node'] = Node(horizon, c0, apps[0], monitoring=mns[0], name='A', generator=g0)
# dg.nodes['B']['node'] = Node(horizon, c1, apps[1], monitoring=mns[1], name='B', generator=g1)
# dg.nodes['C']['node'] = Node(horizon, c2, apps[2], monitoring=mns[2], name='C', generator=g2)
# dg.nodes['D']['node'] = Node(horizon, c2, apps[3], monitoring=mns[3], name='D', generator=g3)

dag=DAG(dg, 'Ord')
simul = SimulationWithDependeciesDAG(horizon, dag, 'Ord')
simul.run()
simul.plot()

# MAP VISUALIZATION
dag.updateDAGUsersRTForVisualization('Ord')

dag.print_dag('users')
dag.print_dag('rt')

#dag.copy()

#print(dag.get_nodes_with_no_children_children())
#print(dag.get_children('A'))
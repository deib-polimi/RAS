import networkx as nx

from dag import DAG
from generators import *
from controllers import *
from applications import Application1
from monitoring import Monitoring
from node import Node
from simulationdag import SimulationWithDependeciesDAG
from simulationlist import SimulationWithDependeciesList

stime = 0.2
appSLA = stime * 3
horizon = 1000
monitoringWindow = 10
initCores = 1000
period = 1

c0 = CTControllerScaleX(period, initCores); c0.setName("ScaleX")
c1 = CTControllerScaleX(period, initCores); c1.setName("ScaleX")
c2 = CTControllerScaleX(period, initCores); c2.setName("ScaleX")
c3 = CTControllerScaleX(period, initCores); c3.setName("ScaleX")
c4 = CTControllerScaleX(period, initCores); c4.setName("ScaleX")
c5 = CTControllerScaleX(period, initCores); c5.setName("ScaleX")
c6 = CTControllerScaleX(period, initCores); c6.setName("ScaleX")
c7 = CTControllerScaleX(period, initCores); c7.setName("ScaleX")
c8 = CTControllerScaleX(period, initCores); c8.setName("ScaleX")
c9 = CTControllerScaleX(period, initCores); c9.setName("ScaleX")



apps = [Application1(appSLA, init_cores=initCores), Application1(appSLA/4, init_cores=initCores), Application1(appSLA/10, init_cores=initCores),
        Application1(appSLA/10, init_cores=initCores), Application1(appSLA/4, init_cores=initCores)]

mns = [Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA),
       Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA)]

c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla); c2.setSLA(apps[2].sla); c3.setSLA(apps[3].sla); c4.setSLA(apps[4].sla)
c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1]); c2.setMonitoring(mns[2]); c3.setMonitoring(mns[3]); c4.setMonitoring(mns[4])

g0 = SinGen(500, 700, 200); g0.setName("SN1 - App 1")
g1 = RampGen(10, 800); g1.setName("RP1 - App 2")
g2 = SinGen(500, 700, 200); g2.setName("SN1 - App 3")
g3 = RampGen(10, 800); g3.setName("RP1 - App 4")
g4 = SinGen(500, 700, 200); g4.setName("SN1 - App 5")

c0.setGenerator(g0); c1.setGenerator(g1); c2.setGenerator(g2); c3.setGenerator(g3); c4.setGenerator(g4)

# CREATE DAG
# REVISE
dg = nx.DiGraph([("A", "B", {"req": 0}), ("B", "C", {"req": 0}),
                 ("B", "D", {"req": 0}), ("A", "E", {"req": 0})])
# nx.set_node_attributes(dg, dg.nodes, 'total_node_req')
nx.set_node_attributes(dg, dg.nodes, 'node')
nx.set_node_attributes(dg, dg.nodes, 'users')
nx.set_node_attributes(dg, dg.nodes, 'rt')
nx.set_edge_attributes(dg, dg.edges, 'req')

dg.nodes['A']['node'] = Node(horizon, c0, apps[0], callers=[], monitoring=mns[0], name="A", generator=g0)
dg.nodes['B']['node'] = Node(horizon, c1, apps[1], callers=[], monitoring=mns[1], name="B", generator=g1)
dg.nodes['C']['node'] = Node(horizon, c2, apps[2], callers=[], monitoring=mns[2], name="C", generator=g2)
dg.nodes['D']['node'] = Node(horizon, c3, apps[3], callers=[], monitoring=mns[3], name="D", generator=g3)
dg.nodes['E']['node'] = Node(horizon, c4, apps[4], callers=[], monitoring=mns[4], name="E", generator=g4)

dag=DAG(dg, 'A')
simul = SimulationWithDependeciesDAG(horizon, dag, 'A')
simul.run()
simul.plot()

# MAP VISUALIZATION
dag.updateDAGUsersRTForVisualization('A')

dag.print_dag('rt')

#dag.copy()

#print(dag.get_nodes_with_no_children_children())
print(dag.get_children('A'))
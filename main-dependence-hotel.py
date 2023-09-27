import copy


import networkx as nx

from applications.hotelreservationservice import*
from applications.sockshopmicroservice import *
from dag import DAG
from generators import *
from controllers import *
from monitoring import Monitoring
from node import Node
from simulationdependences import SimulationWithDependeciesDAG

class MainHotelDependency:
    def __init__(self, filename='', dag=None):
        self.filename = filename
        self.dag = dag

    def run(self, filename=''):
        stime = 0.274
        appSLA = stime
        horizon = 1200
        alfa=0.5
        monitoringWindow = 1
        initCores = 2.5
        period = 1
        search_sla = appSLA
        profile_sla = 0.095
        geo_sla = 0.068
        rate_sla = 0.079

        total_weight = 0.00506
        slas=[appSLA, profile_sla,geo_sla ,rate_sla]


        apps = [Search(appSLA, init_cores=initCores), Profile(appSLA, init_cores=initCores), Geo(appSLA, init_cores=initCores),
                Rate(appSLA, init_cores=initCores)]
        appNames = ['Sch', 'Prf', 'Geo', 'Rat']

        mns = [Monitoring(monitoringWindow, appSLA, local_sla=search_sla),
               Monitoring(monitoringWindow, appSLA, local_sla=profile_sla),
               Monitoring(monitoringWindow, appSLA, local_sla=geo_sla),
               Monitoring(monitoringWindow, appSLA, local_sla=rate_sla)]

        sts=[]
        i = 0
        for app in apps:
            st = alfa * app.weight / total_weight
            localSLA = slas[i] if slas[i] > 0.0 else app.weight* 2
            if localSLA*3/4 < st*appSLA or localSLA/2 > st*appSLA:
                 sts.append(0.5 * localSLA / appSLA)
            else:
                sts.append(st)
            i = i+1

        g=[]

        for i in range(0,4):
            g.append(ZeroGen())
            g[i - 1].setName("ZG -" + str(appNames[i]))

        times = []
        req = [114,20,63,57,72,45,35,36,37,105,102,105,41,47,118,101,58,120,101,34,33,93,112,117]
        for i in range(1,25):
            times.append(i*50)
            req[i-1] = req[i-1] *40#*40 bottleneck


        c0 = CTControllerScaleDependency(period, initCores, BC=0.001, DC=0.02, max_cores=apps[0].max_cores, st=sts[0]); c0.setName("ScaleX")
        c1 = CTControllerScaleDependency(period, initCores,  BC=0.0001, DC=0.002, max_cores=apps[1].max_cores, st=sts[1]); c1.setName("ScaleX")
        c2 = CTControllerScaleDependency(period, initCores, BC=0.0001, DC=0.002, max_cores=apps[2].max_cores, st=sts[2]); c2.setName("ScaleX")
        c3 = CTControllerScaleDependency(period, initCores,  BC=0.0001, DC=0.002, max_cores=apps[3].max_cores, st=sts[3]); c3.setName("ScaleX")



        c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla); c2.setSLA(apps[2].sla); c3.setSLA(apps[3].sla)

        c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1]); c2.setMonitoring(mns[2]); c3.setMonitoring(mns[3])


        g[0] = RampGen(1, 90, 10); g[0].setName("RP - Search")

        #g[1] = RampGen(1, 90, 10); g[1].setName("RP - Prof")
        g[1] = StepGen(times, req); g[1].setName("STP - Prof")

        c0.setGenerator(g[0]); c1.setGenerator(g[1]); c2.setGenerator(g[2]); c3.setGenerator(g[3])

        dg = nx.DiGraph([('Sch', "Prf"), ("Sch", "Geo"), ("Sch", "Rat")])

        dag_model = DAG(dg, 'Sch')
        dg_dep = dag_model.dag
        sync = 1

        for edge in dg_dep.edges:
            dg_dep.edges[edge]['times'] = 1
            dg_dep.edges[edge]['sync'] = sync
        dg_dep.edges[("Sch", "Prf")]['sync'] = 2
        dg_dep.edges[("Sch", "Rat")]['sync'] = 3


        dg_dep.nodes['Sch']['node'] = Node(horizon, c0, apps[0], monitoring=mns[0], name='Sch', generator=g[0], local_sla=search_sla)  # local sla set by user
        dg_dep.nodes['Prf']['node'] = Node(horizon, c1, apps[1], monitoring=mns[1], name='Prf', generator=g[1], local_sla=profile_sla)
        dg_dep.nodes['Geo']['node'] = Node(horizon, c2, apps[2], monitoring=mns[2], name='Geo', generator=g[2], local_sla=geo_sla)
        dg_dep.nodes['Rat']['node'] = Node(horizon, c3, apps[3], monitoring=mns[3], name='Rat', generator=g[3], local_sla=rate_sla)

        for node in dg_dep:
            dg_dep.nodes[node]['users'] = 0
            dg_dep.nodes[node]['node'].total_weight = total_weight  # given by user
            dg_dep.nodes[node]['node'].subtotal_weight = dg.nodes[node]['node'].app.weight

        simul = SimulationWithDependeciesDAG(horizon, dag_model, 'Sch')

        simul.run()

        simul.plot()
        simul.plot(isTotalRT=False)
        # MAP VISUALIZATION
        dag_model.updateForVisualization('Sch')

        dag_model.print_dag('users', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('rt', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('st', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('lrt', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('app', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('cores', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('cores_deviation', show_sync=True, show_times=True,label='hotel_dependency-normal')
        dag_model.print_dag('rt_deviation', show_sync=True, show_times=True,label='hotel_dependency-normal')
        self.dag = dag_model
        dag_model.computeStatiscalTables(filename)

    def computeFinalTable(self):
        self.dag.computeResultTable()

    #
for i in range(11):
 MainHotelDependency().run("experiments/neptuneplus/%s-%d" % ("hotel/statistical", i))

DAG(nx.DiGraph(), 'Ord').computeResultTable(simulatorLebal="neptuneplus/hotel/statistical")


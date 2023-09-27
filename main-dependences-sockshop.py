import copy

import networkx as nx

from applications.sockshopmicroservice import *
from dag import DAG
from generators import *
from controllers import *
from monitoring import Monitoring
from node import Node
from simulationdependences import SimulationWithDependeciesDAG

class MainSockshopDependency:
    def __init__(self, filename=''):
        self.filename = filename

    def run(self, filename=''):
        self.filename = filename
        stime = 0.6
        appSLA = stime
        horizon = 1200
        monitoringWindow = 1
        initCores = 0.5
        inithaeavyCores = 3.5
        period = 1
        alfa = 0.5
        ord_sla = appSLA
        cat_sla = util_sla = del_sla = appSLA/3
        shi_sla = use_sla = pay_sla = appSLA/12

        total_weight=0.2117
        slas=[appSLA, cat_sla,shi_sla,use_sla,pay_sla,util_sla,del_sla]
        apps = [Order(appSLA, init_cores=initCores), CartsCatalogue(appSLA, init_cores=initCores), Shipping(appSLA, init_cores=initCores),
                User(appSLA, init_cores=initCores), Payment(appSLA, init_cores=initCores), CartsUtil(appSLA, init_cores=initCores),
                CartsDelete(appSLA, init_cores=inithaeavyCores)]
        appNames = ['Ord', 'Cat', 'Shi', 'Use', 'Pay', 'util','Del']

        mns = [Monitoring(monitoringWindow, appSLA, local_sla=ord_sla), Monitoring(monitoringWindow, appSLA, local_sla=cat_sla),
               Monitoring(monitoringWindow, appSLA, local_sla=shi_sla),Monitoring(monitoringWindow, appSLA, local_sla=use_sla),
               Monitoring(monitoringWindow, appSLA, local_sla=pay_sla), Monitoring(monitoringWindow, appSLA, local_sla=util_sla),
               Monitoring(monitoringWindow, appSLA, local_sla=del_sla)]
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

        for i in range(0, 7):
            g.append(ZeroGen())
            g[i - 1].setName("ZG -" + str(appNames[i]))

        times = []
        req = [114, 20, 63, 57, 72, 45, 35, 36, 37, 105, 102, 105, 41, 47, 118, 101, 58, 120, 101, 34, 33, 93, 112, 117]
        for i in range(1, 25):
            times.append(i * 50)
            req[i - 1] = req[i - 1]*20


        c0 = CTControllerScaleDependency(period, initCores, BC=0.001, DC=0.02, max_cores=apps[0].max_cores, st=sts[0]); c0.setName("ScaleX")
        c1 = CTControllerScaleDependency(period, initCores,  BC=0.0001, DC=0.002, max_cores=apps[1].max_cores, st=sts[1]); c1.setName("ScaleX")
        c2 = CTControllerScaleDependency(period, initCores, BC=0.0001, DC=0.002, max_cores=apps[2].max_cores, st=sts[2]); c2.setName("ScaleX")
        c3 = CTControllerScaleDependency(period, initCores,  BC=0.0001, DC=0.002, max_cores=apps[3].max_cores, st=sts[3]); c3.setName("ScaleX")
        c4 = CTControllerScaleDependency(period, initCores, BC=0.0001, DC=0.002, max_cores=apps[4].max_cores, st=sts[4]); c4.setName("ScaleX")
        c5 = CTControllerScaleDependency(period, initCores,  BC=0.001, DC=0.02, max_cores=apps[5].max_cores, st=sts[5]); c5.setName("ScaleX")
        c6 = CTControllerScaleDependency(period, initCores, BC=0.001, DC=0.02, max_cores=apps[6].max_cores, st=sts[6]); c6.setName("ScaleX")

        c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla); c2.setSLA(apps[2].sla); c3.setSLA(apps[3].sla)
        c4.setSLA(apps[4].sla); c5.setSLA(apps[5].sla); c6.setSLA(apps[6].sla)

        c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1]); c2.setMonitoring(mns[2]); c3.setMonitoring(mns[3])
        c4.setMonitoring(mns[4]); c5.setMonitoring(mns[5]); c6.setMonitoring(mns[6])

        g[0] = RampGen(1, 90, 10); g[0].setName("RP - Order")
        g[1] = RampGen(1, 90, 10); g[1].setName("RP - Cat")

        g[5] = RampGen(1, 90, 10); g[5].setName("RP - Util")

        # g[6] = RampGen(1, 90, 10); g[6].setName("RP - CartsDelete")
        g[6] = StepGen(times, req); g[6].setName("STP - Del")

        c0.setGenerator(g[0]); c1.setGenerator(g[1]); c2.setGenerator(g[2]); c3.setGenerator(g[3]); c4.setGenerator(g[4])
        c5.setGenerator(g[5]); c6.setGenerator(g[6])

        dg = nx.DiGraph([("Ord", "Cat"), ("Ord", "Shi"), ("Ord", "Use"),
                         ("Ord", "Pay"), ("Ord", "Uti"), ("Ord", "Del")])

        dag_model = DAG(dg, 'Ord')
        dg_dep = dag_model.dag
        sync = 1

        for edge in dg_dep.edges:
            dg_dep.edges[edge]['times'] = 1
            dg_dep.edges[edge]['sync'] = sync

        dg_dep.edges[("Ord", "Uti")]['sync'] = 2
        dg_dep.edges[("Ord", "Del")]['sync'] = 3
        dg_dep.edges[("Ord", "Del")]['times'] = 1
        dg_dep.edges[("Ord", "Pay")]['times'] = 1

        dg_dep.nodes['Ord']['node'] = Node(horizon, c0, apps[0], monitoring=mns[0], name='Ord', generator=g[0], local_sla=ord_sla)  # local sla set by user
        dg_dep.nodes['Cat']['node'] = Node(horizon, c1, apps[1], monitoring=mns[1], name='Cat', generator=g[1], local_sla=cat_sla)
        dg_dep.nodes['Shi']['node'] = Node(horizon, c2, apps[2], monitoring=mns[2], name='Shi', generator=g[2], local_sla=shi_sla)
        dg_dep.nodes['Use']['node'] = Node(horizon, c3, apps[3], monitoring=mns[3], name='Use', generator=g[3], local_sla=use_sla)
        dg_dep.nodes['Pay']['node'] = Node(horizon, c4, apps[4], monitoring=mns[4], name='Pay', generator=g[4], local_sla=pay_sla)
        dg_dep.nodes['Uti']['node'] = Node(horizon, c5, apps[5], monitoring=mns[5], name='Uti', generator=g[5], local_sla=util_sla)
        dg_dep.nodes['Del']['node'] = Node(horizon, c6, apps[6], monitoring=mns[6], name='Del', generator=g[6], local_sla=del_sla)

        for node in dg_dep:
            dg_dep.nodes[node]['users'] = 0
            dg_dep.nodes[node]['node'].total_weight = total_weight  # given by user
            dg_dep.nodes[node]['node'].subtotal_weight = dg.nodes[node]['node'].app.weight

        simul = SimulationWithDependeciesDAG(horizon, dag_model, 'Ord')

        simul.run()
        simul.plot()
        simul.plot(isTotalRT=False)  # for local RT

        # MAP VISUALIZATION
        dag_model.updateForVisualization('Ord')

        dag_model.print_dag('users', show_sync=True, show_times=True,label='dependency_sockshop_normal')
        dag_model.print_dag('rt', show_sync=True, show_times=True,label='dependency_sockshop_normal')    # average (mean)
        dag_model.print_dag('st', show_sync=True, show_times=True,label='dependency_sockshop_normal')
        dag_model.print_dag('lrt', show_sync=True, show_times=True,label='dependency_sockshop_normal')  # average (mean)
        dag_model.print_dag('app', show_sync=True, show_times=True,label='dependency_sockshop_normal')
        dag_model.print_dag('cores', show_sync=True, show_times=True,label='dependency_sockshop_normal')
        dag_model.print_dag('cores_deviation', show_sync=True, show_times=True,label='dependency_sockshop_normal')    # standard deviation (sdt)
        dag_model.print_dag('rt_deviation', show_sync=True, show_times=True,label='dependency_sockshop_normal')  # standard deviation (sdt)
        dag_model.print_dag('lrt_deviation', show_sync=True, show_times=True,label='dependency_sockshop_normal')  # standard deviation (sdt)

        self.dag=dag_model
        dag_model.computeStatiscalTables(filename)
    def computeFinalTable(self):
            self.dag.computeResultTable()

        #
for i in range(11):
    MainSockshopDependency().run("experiments/neptuneplus/%s-%d" % ("sockshop/statistical", i))

DAG(nx.DiGraph(), 'Ord').computeResultTable("neptuneplus/sockshop/statistical", timeWindow=1200)

import copy

import networkx as nx

from applications.complexapplication import ApplicationModel
from complexdag import ComplexDAG
from generators import *
from controllers import *
from monitoring import Monitoring
from node import Node
from simulationNeptune import SimulationNeptune

class MainComplaxApp:
    def __init__(self, filename='', dag=None):
        self.filename = filename
        self.dag = dag

    def run(self, filename=''):
        stime = 3.573
        horizon = 1200
        monitoringWindow = 1
        initCores = 7.0
        period = 1
        appSLA = stime
        st = 0.5

        controls = []
        monitors=[]
        generators=[]
        apps=[]

        slas=[3.573,0.48,0.675,0.978,0.579,0.918,0.372,0.405,0.324,0.18,0.072,0.24,0.474,
               0.342,0.12,0.372,0.333,0.258,0.204,0.174,0.264,0.282,0.204,0.216,0.132]

        nominalRT = [0.034,0.072,0.062,0.042,0.07,0.068,0.02,0.048,0.044,0.12,0.048,0.112,
               0.128,0.092,0.08,0.026,0.222,0.056,0.136,0.116,0.04,0.044,0.136,0.144,0.088]

        i = 0
        for sla in slas:
            apps.append(ApplicationModel(sla, init_cores=initCores, nominalRT=nominalRT[i]))
            controls.append(CTControllerScaleX(period, initCores, st=st, BC=0.001, DC=0.02)); controls[i].setName("ScaleX")
            monitors.append(Monitoring(monitoringWindow, appSLA, local_sla=sla))
            controls[i].setSLA(apps[i].sla)
            controls[i].setMonitoring(monitors[i])
            generators.append(ZeroGen())
            generators[i].setName("ZG -f" + str(i+1))
            controls[i].setGenerator(generators[i])
            i = i+1
        # controls[1].max = 10.0

        generators[0]=RampGen(1, 90, 10) #direct call for function f1
        controls[0].setGenerator(generators[0]);  controls[0].setName("RP -f1")# change the zero generator to RAMP for f1

        times = []
        req1=[]
        req = [114,20,63,57,72,45,35,36,37,105,102,105,41,47,118,101,58,120,101,34,33,93,112,117]
        for i in range(1,25):
            times.append(i*50)

        generators[1] = StepGen(times, req); generators[1].setName("STP - f2")
        controls[1].setGenerator(generators[1]);  controls[1].setName("STP -f2")

        generators[10] = StepGen(times, req); generators[10].setName("STP - f11")
        controls[10].setGenerator(generators[10]); controls[10].setName("STP -f11")

        generators[13] = StepGen(times, req); generators[13].setName("STP - f14")
        controls[13].setGenerator(generators[13]); controls[13].setName("STP -f14")




        for i in range(0,24):
            req1.append(req[i]*10) # * 10 bottleneck

        generators[8] = StepGen(times, req1); generators[8].setName("STP - f9")
        controls[8].setGenerator(generators[8]);  controls[8].setName("STP -f9")

        generators[23] = StepGen(times, req1); generators[23].setName("STP - f24")
        controls[23].setGenerator(generators[23]); controls[23].setName("STP -f24")


        dg = nx.DiGraph([("f1", "f2"),("f2", "f15"),
        ("f2", "f16"),("f16", "f17"),("f18", "f20"),
        ("f1", "f3"),("f3", "f8"),("f8", "f17"),
        ("f9", "f18"),
        ("f3", "f10"),("f3", "f9"),
        ("f22", "f24"),("f22", "f25"),
        ("f1", "f4"),("f4", "f3"),
        ("f4", "f12"),("f12", "f11"),
        ("f21", "f23"),
        ("f1", "f5"),("f5", "f12"),
        ("f5", "f13"),("f13", "f21"), ("f13", "f22"),
        ("f1", "f6"),("f6", "f13"),
        ("f6", "f14"),("f14", "f19"),
        ("f1", "f7"),("f7", "f14")
                         ])

        dag_model = ComplexDAG(dg, 'f1')
        dg_dep = dag_model.dag
        sync = 1
        for edge in dg_dep.edges:
            dg_dep.edges[edge]['times'] = 1
            dg_dep.edges[edge]['sync'] = sync

        dg_dep.edges[("f1", "f3")]['sync'] = 2
        dg_dep.edges[("f1", "f4")]['sync'] = 3
        dg_dep.edges[("f1", "f6")]['sync'] = 4
        dg_dep.edges[("f1", "f7")]['sync'] = 5

        dg_dep.edges[("f3", "f9")]['sync'] = 2
        dg_dep.edges[("f3", "f10")]['sync'] = 2

        dg_dep.edges[("f4", "f12")]['sync'] = 2

        dg_dep.edges[("f6", "f14")]['sync'] = 2
        dg_dep.edges[("f22", "f24")]['times'] = 1

        i = 0

        for app in apps:
            dg_dep.nodes["f"+str(i+1)]['node'] = Node(horizon, controls[i], app, monitoring=monitors[i], name="f"+str(i+1),
                                                    generator=generators[i])
            i = i+1

        simul = SimulationNeptune(horizon, dag_model, 'f1')
        simul.run()
        simul.plot()


        # MAP VISUALIZATION
        dag_model.updateForVisualization('f1')

        # special nodes for arrows only


        # Add a new node

        positions = {
            "f1": (10, 9),
            "f2": (2, 7),
            "f3": (7, 7),
            "f4": (10, 7),
            "f5": (13, 7),
            "f6": (16, 7),
            "f7": (19, 7),
            "f8": (5, 5),
            "f9": (7, 3),
            "f10": (9, 5),
            "f11": (10, 3),
            "f12": (11, 5),
            "f13": (15, 5),
            "f14": (18, 5),
            "f15": (0, 5),
            "f16": (3, 5),
            "f17": (3, 3),
            "f18": (6, 1),
            "f19": (19, 3),
            "f20": (5, -1),
            "f21": (16, 2),
            "f22": (13, 2),
            "f23": (17, -1),
            "f24": (10, 0),
            "f25": (15, 0)
        }


        dag_model.print_dag('users',  pos=positions,show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('rt', pos=positions, show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('st',pos=positions, show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('lrt', pos=positions, show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('app', pos=positions, show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('cores', pos=positions,show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('cores_deviation',pos=positions, show_sync=True, show_times=True,label='dependency-normal')
        dag_model.print_dag('rt_deviation', pos=positions,show_sync=True, show_times=True,label='dependency-normal')

        self.dag = dag_model
        dag_model.computeStatiscalTables(filename)


    def computeFinalTable(self):
        self.dag.computeResultTable()


for i in range(11):
    MainComplaxApp().run("experiments/neptune/%s-%d" % ("complexapp/statistical", i))

ComplexDAG(nx.DiGraph(), 'Ord').computeResultTable(simulatorLebal="neptune/complexapp/statistical", timeWindow=1200)

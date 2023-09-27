from numpy import array
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
plt.rcParams.update({'font.size': 18})

def to_mil(number):
    return round(number*1000, 2)
class SimulationWithDependeciesDAG:
    def __init__(self, horizon, dag, startedNodeName): # horizon, apps: list, generator: list, monitoring: list, controller: list):
        self.startedNodeName = startedNodeName
        self.dag=dag #self.apps = apps
        self.horizon = horizon
        self.violations = 0
        self.listNodes = dag.toList(startedNodeName)
        self.controllers = dag.getControllers()
        self.generators = dag.getGenerators()
        self.monitorings = dag.getMonitorings()
        self.names = ["%s-%s" % (c.name, g.name) for c, g in zip(self.controllers, self.generators)]

    def run(self):
        for t in range(0, self.horizon):
            self.dag.resetUsers(self.startedNodeName)
            self.dag.resetweights(self.startedNodeName)
            if self.dag is not None:
                self.dag.updateUsersDAG(self.startedNodeName, t) # update users on the DAG from a started node and set RT
                self.dag.setST(alfa=0.5)  # new
                self.dag.setAllRT()
                self.dag.setCores(self.startedNodeName, t)
                self.dag.fillStatiscal(starpointUsers=1)
        self.dag.setStatistical()

    def log(self):
        i = 0
        for monitoring in self.monitorings:
            arts = array(monitoring.getAllTotalRTs())
            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            if not isinstance(aviolations, list):
                arts = [arts]
                acores = [acores]
                aviolations = [aviolations]
            output = ""
            for (rts, cores, violations) in zip(arts, acores, aviolations):
                output += "\\textit{%s} & \\textit{%s} & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%d$ & $%d$ \\\\ \hline \n" % (
                    self.controllers[i].name, self.generators[i].name, rts.mean(), rts.std(), rts.min(), rts.max(),
                    violations, cores.mean())
                i += 1
            return output

    def plot(self, isTotalRT = True):
        i = 0
        rt_label = "rt.pdf"
        for monitoring, name in zip(self.monitorings, self.names):
            if isTotalRT:
             arts = array(monitoring.getAllTotalRTs())  # getAllTotalRTs() considering the dependences interests us the local RT not the total
             rt_label = "total-rt.pdf"
            else:
                arts = array(
                    monitoring.getAllRTs())  # getAllTotalRTs() considering the dependences interests us the local RT not the total

            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            ausers = monitoring.getAllUsers()
            if not isinstance(aviolations, list):
                arts = [arts]
                acores = [acores]
                aviolations = [aviolations]
                ausers = [ausers]
            for (rts, cores, users) in zip(arts, acores, ausers):
                newrts = []
                newcore = []
                for (rt, cor) in zip(rts, cores):
                    newrts.append(to_mil(rt))
                    newcore.append(to_mil(cor))
                rts = newrts
                cores = newcore
                fig, ax1 = plt.subplots()
                ax1.set_ylabel('# workload')
                ax1.set_xlabel("time [s]")
                ax1.plot(users, 'r--', linewidth=2)
                ax2 = ax1.twinx()
                ax2.plot(cores, 'b-', linewidth=2)
                ax2.set_ylabel('# millicores')
                fig.tight_layout()
                plt.savefig("experiments/neptuneplus/%s-%d-workcore.pdf" % (name, i))
                plt.close()

                fig, ax1 = plt.subplots()
                ax1.set_ylabel('RT [ms]')
                ax1.set_xlabel("time [s]")
                ax1.plot(rts, 'g-', linewidth=2)
                ax2 = ax1.twinx()
                sla = to_mil(self.listNodes[i].local_sla)
                if sla <= 0:
                    sla = to_mil(
                        self.listNodes[i].app.sla[i] if isinstance(self.listNodes[i].app.sla, list) else self.listNodes[
                            i].app.sla)
                ax2.plot([sla] * len(rts), 'r--', linewidth=2)
                ax2.set_ylabel('RT [ms]')
                m1, M1 = ax1.get_ylim()
                m2, M2 = ax2.get_ylim()
                m = min([m1, m2])
                M = max([M1, M2])
                ax1.set_ylim([m, M])
                ax2.set_ylim([m, M])
                fig.tight_layout()
                plt.savefig("experiments/neptuneplus/%s-%d-%s" % (name, i, rt_label))
                plt.close()
                i += 1

    def getTotalViolations(self):
        for monitoring in self.monitorings:
            aviolations = monitoring.getViolations()
            if not isinstance(aviolations, list):
                aviolations = [aviolations]
            return sum(aviolations, 0.0)

    def exportData(self, outDir="experiments/matfile"):
        os.makedirs(outDir, exist_ok=True)
        i = 0
        for monitoring in self.monitorings:
            arts = array(monitoring.getAllTotalRTs())
           # arts = array(monitoring.getAllRTs())
            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            ausers = monitoring.getAllUsers()
            savemat("%s/%s.mat" % (outDir, self.names[i]),
                    {"rts": arts, "cores": acores, "ausers": ausers, "sla": monitoring.sla})
            i += 1


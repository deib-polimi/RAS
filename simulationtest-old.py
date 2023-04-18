from generators import Generator
from applications import Application
from numpy import array
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

plt.rcParams.update({'font.size': 18})


class SimulationTest:
    def __init__(self, horizon, nodesimu):
       # self.nodesimu.app = app
       # self.nodesimu.generator = generator
       # self.nodesimu.controller = controller
        self.horizon = horizon
       # self.nodesimu.monitoring = monitoring
        self.violations = 0
        self.nodesimu = nodesimu
        self.name = "%s-%s" % (nodesimu.controller.name, nodesimu.generators.name)


    def run(self):
        for t in range(0, self.horizon):
            print(t)
            users = self.nodesimu.generators.tick(t)
            rt = self.nodesimu.apps.setRT(users)
            self.nodesimu.monitorings.tick(t, rt, users, self.nodesimu.apps.cores)
            cores = self.nodesimu.controller.tick(t)
            self.nodesimu.apps.cores = cores

    def log(self):
        arts = array(self.nodesimu.monitorings.getAllRTs())
        acores = array(self.nodesimu.monitorings.getAllCores())
        aviolations = self.nodesimu.monitorings.getViolations()
        if not isinstance(aviolations, list):
            arts = [arts]
            acores = [acores]
            aviolations = [aviolations]
        output = ""
        for (rts, cores, violations) in zip(arts, acores, aviolations):
            output += "\\textit{%s} & \\textit{%s} & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%d$ & $%d$ \\\\ \hline \n" % (
                self.nodesimu.controller.name, self.nodesimu.generators.name, rts.mean(), rts.std(), rts.min(), rts.max(), violations,
                cores.mean())
        return output

    def plot(self):
        i = 0
        arts = array(self.nodesimu.monitorings.getAllRTs())
        acores = array(self.nodesimu.monitorings.getAllCores())
        aviolations = self.nodesimu.monitorings.getViolations()
        ausers = self.nodesimu.monitorings.getAllUsers()
        if not isinstance(aviolations, list):
            arts = [arts]
            acores = [acores]
            aviolations = [aviolations]
            ausers = [ausers]
        for (rts, cores, users) in zip(arts, acores, ausers):
            fig, ax1 = plt.subplots()
            ax1.set_ylabel('# workload')
            ax1.set_xlabel("time [s]")
            ax1.plot(users, 'r--', linewidth=2)
            ax2 = ax1.twinx()
            ax2.plot(cores, 'b-', linewidth=2)
            ax2.set_ylabel('# cores')
            fig.tight_layout()
            plt.savefig("experiments/%s-%d-workcore.pdf" % (self.name, i))
            plt.close()

            fig, ax1 = plt.subplots()
            ax1.set_ylabel('RT [s]')
            ax1.set_xlabel("time [s]")
            ax1.plot(rts, 'g-', linewidth=2)
            ax2 = ax1.twinx()

            sla = self.nodesimu.apps.sla[i] if isinstance(self.nodesimu.apps.sla, list) else self.nodesimu.apps.sla
            ax2.plot([sla] * len(rts),

                     'r--', linewidth=2)
            ax2.set_ylabel('RT [s]')
            m1, M1 = ax1.get_ylim()
            m2, M2 = ax2.get_ylim()
            m = min([m1, m2])
            M = max([M1, M2])
            ax1.set_ylim([m, M])
            ax2.set_ylim([m, M])
            fig.tight_layout()
            plt.savefig("experiments/%s-%d-rt.pdf" % (self.name, i))
            plt.close()
            i += 1

        '''
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('RT [s]')
        ax1.set_xlabel("time [s]")
        ax1.plot(self.nodesimu.monitoring.allRts, 'g-')
        ax1.plot([self.nodesimu.app.sla] * len(self.nodesimu.monitoring.allRts), 'r:')
        ax2 = ax1.twinx()
        ax2.plot(self.nodesimu.monitoring.allCores, 'b--')
        ax2.set_ylabel('# cores')
        plt.savefig("experiments/%s-rt-cores.pdf" % (self.name,))
        fig.tight_layout()
        plt.close()
        plt.plot(self.nodesimu.monitoring.allUsers, 'b-')
        plt.ylabel("# requests per second")
        plt.xlabel("time [s]")
        plt.savefig("experiments/%s-workload.pdf" % (self.name,))
        plt.close()
        '''

    def getTotalViolations(self):
        aviolations = self.nodesimu.monitorings.getViolations()
        if not isinstance(aviolations, list):
            aviolations = [aviolations]
        return sum(aviolations)

    def exportData(self, outDir="experiments/matfile"):

        os.makedirs(outDir, exist_ok=True)

        arts = array(self.nodesimu.monitorings.getAllRTs())
        acores = array(self.nodesimu.monitorings.getAllCores())
        aviolations = self.nodesimu.monitorings.getViolations()
        ausers = self.nodesimu.monitorings.getAllUsers()
        savemat("%s/%s.mat" % (outDir, self.name),
                {"rts": arts, "cores": acores, "ausers": ausers, "sla": self.nodesimu.monitorings.sla})


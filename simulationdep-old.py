from generators import Generator
from applications import Application
from numpy import array
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

plt.rcParams.update({'font.size': 18})


class SimulationWithDependecies:
    def __init__(self, horizon, applist):
        self.app = applist  #["A", "B"] # A --> B
        self.generator = [applist[0].generators, applist[1].generators]  # ["GEN-A", "GEN-B"]
        self.controller = [applist[0].controller, applist[1].controller]  # [...]
        self.horizon = horizon
        self.monitoring = [applist[0].monitorings, applist[1].monitorings]   # [...]
        self.violations = 0
        self.name = []
        for i in range(0, len(self.generator)):
            self.name[i] = "%s-%s" % (self.controller[i].name, self.generator[i].name)  # check

    def run(self):
        for t in range(0, self.horizon):
            app1, gen1 = self.app[0], self.generator[0]
            app2, gen2 = self.app[1], self.generator[1]
            users_app1 = gen1.tick(t)  # check
            users_app2 = gen2.tick(t)+users_app1  # check
            rt_app2 = app2.setRT(users_app2)  # check
            rt_app1 = app1.setRT(users_app1)+rt_app2  # check
            self.monitoring[0].tick(t, rt_app1, users_app1, app1.cores)
            self.monitoring[1].tick(t, rt_app2, users_app2, app2.cores)
            cores_app1 = self.controller[0].tick(t)
            app1.cores = cores_app1
            cores_app2 = self.controller[1].tick(t)
            app2.cores = cores_app2
      
    def log(self):
        i = 0
        for monitoring in self.monitoring:
            arts = array(monitoring.getAllRTs())
            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            if not isinstance(aviolations, list):
                arts = [arts]
                acores = [acores]
                aviolations = [aviolations]
            output = ""
            for (rts, cores, violations) in zip(arts, acores, aviolations):
                output += "\\textit{%s} & \\textit{%s} & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%d$ & $%d$ \\\\ \hline \n" % (self.controller[i].name, self.generator[i].name, rts.mean(), rts.std(), rts.min(), rts.max(), violations, cores.mean())
                i += 1
            return output

    def plot(self):
        i = 0
        for monitoring in self.monitoring:
            arts = array(monitoring.getAllRTs())
            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            ausers = monitoring.getAllUsers()
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

                sla = self.app[i].sla[i] if isinstance(self.app[i].sla, list) else self.app[i].sla
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
        ax1.plot(self.monitoring.allRts, 'g-')
        ax1.plot([self.app.sla] * len(self.monitoring.allRts), 'r:')
        ax2 = ax1.twinx()
        ax2.plot(self.monitoring.allCores, 'b--')
        ax2.set_ylabel('# cores')
        plt.savefig("experiments/%s-rt-cores.pdf" % (self.name,))
        fig.tight_layout()
        plt.close()
        plt.plot(self.monitoring.allUsers, 'b-')
        plt.ylabel("# requests per second")
        plt.xlabel("time [s]")
        plt.savefig("experiments/%s-workload.pdf" % (self.name,))
        plt.close()
        '''
    
    def getTotalViolations(self):
        for monitoring in self.monitoring:
            aviolations = monitoring.getViolations()
            if not isinstance(aviolations, list):
                aviolations = [aviolations]
            return sum(aviolations)  # check
    
    def exportData(self,outDir="experiments/matfile"):
        
        os.makedirs(outDir, exist_ok=True)
        i = 0
        for monitoring in self.monitoring:
            arts = array(monitoring.getAllRTs())
            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            ausers = monitoring.getAllUsers()
            savemat("%s/%s.mat"%(outDir, self.name[i]), {"rts": arts, "cores": acores, "ausers":ausers, "sla":monitoring.sla})
            i += 1


from numpy import array
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

plt.rcParams.update({'font.size': 18})


class Simulation:
    def __init__(self, horizon, app, generator, monitoring, controller, output_folder):
        self.app = app
        self.generator = generator
        self.controller = controller
        self.horizon = horizon
        self.monitoring = monitoring
        self.violations = 0
        self.output_folder = output_folder
        self.name = "%s-%s" % (controller.name, generator.name)

    def run(self):
        for t in range(0, self.horizon):
            users = self.generator.tick(t)
            rt = self.app.setRT(users, t)
            self.monitoring.tick(t, rt, users, self.app.cores)
            cores = self.controller.tick(t)
            self.app.cores = cores
            print(f"sim: {self.name}, time: {t+1}, rt: {rt:.2f}, users: {int(users)}, cores: {cores:.2f}")

      
    def log(self):
        arts = array(self.monitoring.getAllRTs())
        acores = array(self.monitoring.getAllCores())
        aviolations = self.monitoring.getViolations()
        if not isinstance(aviolations, list):
            arts = [arts]
            acores = [acores]
            aviolations = [aviolations]
        output = ""
        for (rts, cores, violations) in zip(arts, acores, aviolations):
            output += "\\textit{%s} & \\textit{%s} & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%d$ & $%.2f$ \\\\ \hline \n" % (self.controller.name, self.generator.name, rts.mean(), rts.std(), rts.min(), rts.max(), violations, cores.mean())
        return output

    def plot(self):
        i = 0
        arts = array(self.monitoring.getAllRTs())
        acores = array(self.monitoring.getAllCores())
        aviolations = self.monitoring.getViolations()
        ausers = self.monitoring.getAllUsers()
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
            plt.savefig(f"{self.output_folder}/%s-%d-workcore.pdf" % (self.name, i))
            plt.close()

            fig, ax1 = plt.subplots()
            ax1.set_ylabel('RT [s]')
            ax1.set_xlabel("time [s]")
            ax1.plot(rts, 'g-', linewidth=2)
            ax2 = ax1.twinx()

            sla = self.app.sla[i] if isinstance(self.app.sla, list) else self.app.sla
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
            plt.savefig(f"{self.output_folder}//%s-%d-rt.pdf" % (self.name, i))
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
        plt.savefig(f"{self.output_folder}//%s-rt-cores.pdf" % (self.name,))
        fig.tight_layout()
        plt.close()
        plt.plot(self.monitoring.allUsers, 'b-')
        plt.ylabel("# requests per second")
        plt.xlabel("time [s]")
        plt.savefig(f"{self.output_folder}//%s-workload.pdf" % (self.name,))
        plt.close()
        '''
    
    def getTotalViolations(self):
        aviolations = self.monitoring.getViolations()
        if not isinstance(aviolations, list):
            aviolations = [aviolations]
        return sum(aviolations)
    
    def exportData(self):
        arts = array(self.monitoring.getAllRTs())
        acores = array(self.monitoring.getAllCores())
        aviolations = self.monitoring.getViolations()
        ausers = self.monitoring.getAllUsers()
        savemat("%s/%s.mat"%(self.output_folder,self.name), {"rts":arts,"cores":acores,"ausers":ausers,"sla":self.monitoring.sla})


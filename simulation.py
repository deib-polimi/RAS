from generators import Generator
from applications import Application
from numpy import array
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

class Simulation:
    def __init__(self, horizon, app, generator, monitoring, controller):
        self.app = app
        self.generator = generator
        self.controller = controller
        self.horizon = horizon
        self.monitoring = monitoring
        self.violations = 0
        self.name = "%s-%s" % (controller.name, generator.name)

    def run(self):
        for t in range(0, self.horizon):
            print(t)
            users = self.generator.tick(t)
            rt = self.app.setRT(users)
            self.monitoring.tick(t, rt, users, self.app.cores)
            cores = self.controller.tick(t)
            self.app.cores = cores
      
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
            output += "\\textit{%s} & \\textit{%s} & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%d$ & $%d$ \\\\ \hline \n" % (self.controller.name, self.generator.name, rts.mean(), rts.std(), rts.min(), rts.max(), violations, cores.mean())
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
            plt.savefig("experiments/%s-%d-workcore.png" % (self.name, i))
            plt.close()

            fig, ax1 = plt.subplots()
            ax1.set_ylabel('RT [s]')
            ax1.set_xlabel("time [s]")
            ax1.plot(rts, 'g-', linewidth=2)
            ax2 = ax1.twinx()
            ax2.plot([self.app.sla[i]] * len(rts),
                    'r--', linewidth=2)
            ax2.set_ylabel('RT [s]')
            m1, M1 = ax1.get_ylim()
            m2, M2 = ax2.get_ylim()
            m = min([m1, m2])
            M = max([M1, M2])
            print(M)
            M = self.app.sla[i]*2
            print(self.app.sla)
            ax1.set_ylim([m, M])
            ax2.set_ylim([m, M])
            fig.tight_layout()
            plt.savefig("experiments/%s-%d-rt.png" % (self.name, i))
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

from generators import Generator
from applications import Application
from numpy import array
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

plt.rcParams.update({'font.size': 18})


class SimulationWithDependeciesList:
    def __init__(self, horizon, apps: list, generator: list, monitoring: list, controller: list):
        self.apps = apps
        self.generators = generator
        self.controllers = controller
        self.horizon = horizon
        self.monitorings = monitoring
        self.violations = 0
        self.names = ["%s-%s" % (c.name, g.name) for c, g in zip(controller, generator)]

    # update number of users for all following Apps
    def updateUsers(self, apps, pos, newusers):
        if (pos + 1 < len(apps)):
            for i in range(pos + 1, len(apps)):
              apps[i].users = apps[i].users + newusers


    def resetUsers(self, apps):
        for app in apps:
            app.users=0


    def run(self):
        for t in range(0, self.horizon):
            self.resetUsers(self.apps)
            if len(self.apps) > 0:
                    for i in range(0,len(self.apps)):
                        app, gen = self.apps[i], self.generators[i]
                        #currentusersapp=app.numberusers # first is zero
                        newusers= gen.tick(t)
                        users_app=app.users= app.users + newusers
                        app.setRT(users_app) # check
                        self.updateUsers(self.apps, i, newusers)
                    invertedAppList = self.apps[::-1]
                    invertedMonitoringList=self.monitorings[::-1]
                    invertedControllersList = self.controllers[::-1]
                    currentrt=0
                    for app, mo, cont in zip(invertedAppList, invertedMonitoringList, invertedControllersList):
                        currentrt= app.setRT(app.users) + currentrt
                        app.RT=currentrt
                        mo.tick(t, currentrt, app.users, app.cores)
                        cores_app = cont.tick(t)
                        app.cores = cores_app


    def log(self):
        i = 0
        for monitoring in self.monitorings:
            arts = array(monitoring.getAllRTs())
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

    def plot(self):
        i = 0
        for monitoring, name in zip(self.monitorings, self.names):
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
                plt.savefig("experiments/%s-%d-workcore.pdf" % (name, i))
                plt.close()

                fig, ax1 = plt.subplots()
                ax1.set_ylabel('RT [s]')
                ax1.set_xlabel("time [s]")
                ax1.plot(rts, 'g-', linewidth=2)
                ax2 = ax1.twinx()

                sla = self.apps[i].sla[i] if isinstance(self.apps[i].sla, list) else self.apps[i].sla
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
                plt.savefig("experiments/%s-%d-rt.pdf" % (name, i))
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
        for monitoring in self.monitorings:
            aviolations = monitoring.getViolations()
            if not isinstance(aviolations, list):
                aviolations = [aviolations]
            return sum(aviolations)  # check

    def exportData(self, outDir="experiments/matfile"):

        os.makedirs(outDir, exist_ok=True)
        i = 0
        for monitoring in self.monitorings:
            arts = array(monitoring.getAllRTs())
            acores = array(monitoring.getAllCores())
            aviolations = monitoring.getViolations()
            ausers = monitoring.getAllUsers()
            savemat("%s/%s.mat" % (outDir, self.name[i]),
                    {"rts": arts, "cores": acores, "ausers": ausers, "sla": monitoring.sla})
            i += 1


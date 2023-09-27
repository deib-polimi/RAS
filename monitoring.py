class Monitoring:
    def __init__(self, window, sla, local_sla=0.0, reducer=lambda x: sum(x)/len(x)):
        self.allRts = []
        self.allTotalRts = []
        self.allUsers = []
        self.allCores = []
        self.rts = []
        self.total_rt = []
        self.users = []
        self.reducer = reducer
        self.window = window
        self.time = []
        self.sla = sla
        self.local_sla = local_sla

    def tick(self, t, rt, total_rt, users, cores):
        if len(self.rts) == self.window:
            del self.rts[0]  # delete Response time in the first position
            del self.users[0]
            del self.total_rt[0]

        self.time.append(t)
        self.rts.append(rt)
        self.total_rt.append(total_rt)
        self.users.append(users)
        self.allRts.append(self.getRT())
        self.allTotalRts.append(self.getTotalRT())
        self.allUsers.append(self.getUsers())
        self.allCores.append(cores)



    def getUsers(self):
        return self.reducer(self.users)

    def getRT(self):
        return self.reducer(self.rts)

    def getTotalRT(self):
        return self.reducer(self.total_rt)

    def getViolations(self): # TODO (DONE) Number of violations per Function
        return sum([1 if rt > self.local_sla else 0 for rt in self.allRts])

    def getNViolations(self): #TODO New for test
        return sum([0 if rt > self.local_sla else 1 for rt in self.allRts])


    def getTotalViolations(self): # check considering totalRT TODO New
        for rt in self.allTotalRts:
            print(self.local_sla , rt)
        return sum([1 if rt > self.local_sla else 0 for rt in self.allTotalRts])


    def getAllRTs(self):
        return self.allRts

    def getAllTotalRTs(self):
        return self.allTotalRts

    def getAllUsers(self):
        return self.allUsers
    def getAllCores(self):
        return self.allCores

class MultiMonitoring(Monitoring):
    def __init__(self, monitorings):
        self.monitorings = monitorings
    
    def tick(self, t, rt, total_rt, users, cores):
        for i in range(len(self.monitorings)):
            self.monitorings[i].tick(t, total_rt[i], users[i], cores[i])

    def getUsers(self):
        return [m.getUsers() for m in self.monitorings]

    def getRT(self):
        return [m.getTotalRT() for m in self.monitorings]
    
    def getViolations(self):
        return [m.getViolations() for m in self.monitorings]
    
    def getAllRTs(self):
        return [m.getAllTotalRTs() for m in self.monitorings]

    def getAllUsers(self):
        return [m.getAllUsers() for m in self.monitorings]

    def getAllCores(self):
        return [m.getAllCores() for m in self.monitorings]

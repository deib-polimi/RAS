class Monitoring:
    def __init__(self, window, sla, reducer=lambda x: sum(x)/len(x)):
        self.allRts = []
        self.allUsers = []
        self.allCores = []
        self.rts = []
        self.users = []
        self.reducer = reducer
        self.window = window
        self.time = []
        self.sla = sla

    def tick(self, t, rt, users, cores):
        if len(self.rts) == self.window:
            del self.rts[0]
            del self.users[0]


        self.time.append(t)
        self.rts.append(rt)
        self.users.append(users)
        self.allRts.append(self.getRT())
        self.allUsers.append(self.getUsers())
        self.allCores.append(cores)

    def getUsers(self):
        return self.reducer(self.users)

    def getRT(self):
        return self.reducer(self.rts)

    def getViolations(self):
        return sum([1 if rt > self.sla else 0 for rt in self.allRts])

    def getAllRTs(self):
        return self.allRts
    def getAllUsers(self):
        return self.allUsers
    def getAllCores(self):
        return self.allCores

class MultiMonitoring(Monitoring):
    def __init__(self, monitorings):
        self.monitorings = monitorings
    
    def tick(self, t, rt, users, cores):
        for i in range(len(self.monitorings)):
            self.monitorings[i].tick(t, rt[i], users[i], cores[0,i])

    def getUsers(self):
        return [m.getUsers() for m in self.monitorings]

    def getRT(self):
        return [m.getRT() for m in self.monitorings]
    
    def getViolations(self):
        return [m.getViolations() for m in self.monitorings]
    
    def getAllRTs(self):
        return [m.getAllRTs() for m in self.monitorings]

    def getAllUsers(self):
        return [m.getAllUsers() for m in self.monitorings]

    def getAllCores(self):
        return [m.getAllCores() for m in self.monitorings]

class Monitoring:
    def __init__(self, window, sla, reducer=lambda x: sum(x)/len(x)):
        self.reducer = reducer
        self.window = window
        self.sla = sla
        self.reset()

    def tick(self, t, rt, users, cores):
        for i in range(1, len(self.time)+1):
            if t - self.time[-i] > self.window:
                try:
                    del self.rts[-i]
                    del self.users[-i]
                except:
                    break

        self.time.append(t)
        self.rts.append(rt)
        self.users.append(users)
        self.allRts.append(self.getRT())
        self.allUsers.append(self.getUsers())
        self.allCores.append(cores)

    def getUsers(self):
        if not len(self.users): return 0
        return self.reducer(self.users)

    def getRT(self):
        if not len(self.rts): return 0
        return self.reducer(self.rts)

    def getViolations(self):
        def appendViolation(rts):
            if self.reducer(rts) > self.sla:
                return 1
            else:
                return 0
        second = int(self.time[0])
        violations = []
        rts = []
        
        for (t, rt) in zip(self.time, self.allRts):
            if int(t) != second:
                violations.append(appendViolation(rts))
                rts = []
                second = int(t)
            rts.append(rt)
        violations.append(appendViolation(rts))
        return sum(violations)

    def getAllRTs(self):
        return self.allRts
    def getAllUsers(self):
        return self.allUsers
    def getAllCores(self):
        return self.allCores
    def getAllTimes(self):
        return self.time
        
    def reset(self):
        self.allRts = []
        self.allUsers = []
        self.allCores = []
        self.rts = []
        self.users = []
        self.time = []
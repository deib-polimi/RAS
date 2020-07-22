class Monitoring:
    def __init__(self, window, reducer=lambda x: sum(x)/len(x)):
        self.allRts = []
        self.allUsers = []
        self.allCores = []
        self.rts = []
        self.users = []
        self.reducer = reducer
        self.window = window
        self.time = []

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

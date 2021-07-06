class Generator:
    def __init__(self):
        self.name = type(self).__name__

    def tick(self, t):
        return self.f(t)

    def __str__(self):
        return "%s - " % (self.name,)

    def setName(self, name):
        self.name = name

    def f(self, x): return x

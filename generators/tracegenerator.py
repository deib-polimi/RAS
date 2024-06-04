from .generator import Generator


class TraceGen(Generator):
    
    def __init__(self, data, shift=0, bias = 10):
        super().__init__()
        self.data = data
        self.maxIndex = len(self.data)
        mx = max(self.data)
        mn = min(self.data)
        self.data = [(v - mn)/(mx-mn)*bias+shift for v in self.data]

    def f(self, x):
        return self.data[int(x) % self.maxIndex]

    def __str__(self):
        return super().__str__()

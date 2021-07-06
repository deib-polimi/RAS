from .generator import Generator


class MultiGenerator(Generator):
    def __init__(self, generators):
        super().__init__()
        self.generators = generators
        self.name = "-".join([g.name for g in self.generators])

    def f(self, x):
        return [int(g.f(x)) for g in self.generators]

    def __str__(self):
        return [g.__str__() for g in self.generators]

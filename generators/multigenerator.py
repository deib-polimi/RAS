from .generator import Generator


class MultiGenerator(Generator):
    def __init__(self, generators):
        self.generators = generators
        self.name = "MultiGen"

    def f(self, x):
        return [g.f(x) for g in self.generators]

    def __str__(self):
        return [g.__str__() for g in self.generators]

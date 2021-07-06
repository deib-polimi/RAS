from .generator import Generator


class StepGen(Generator):
    def __init__(self, intervals, values):
        super().__init__()
        assert len(intervals) == len(values)
        self.intervals = intervals
        self.values = values

    def f(self, x):
        i = 0
        while i < len(self.intervals) and x >= self.intervals[i]:
            i += 1
        return self.values[min(i, len(self.values)-1)]

    def __str__(self):
        return super().__str__() + " intervals: %s values: %s" % (self.intervals, self.values)

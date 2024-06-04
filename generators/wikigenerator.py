from .tracegenerator import TraceGen
import csv


class WikiGen(TraceGen):
    def __init__(self, shift=0, bias = 10):
        with open("./generators/wiki.1190153705.csv", 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = [int(row[0])  for row in reader]
        super().__init__(data, shift, bias)

    def __str__(self):
        return super().__str__()

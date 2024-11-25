import numpy as np
class CircularArray:
    def __init__(self, size):
        self.arr = np.zeros(size)
        self.size = size
        self.head = 0

    def append(self, value):
        self.arr[self.head] = value
        self.head = (self.head + 1) % self.size

    def __getitem__(self, index):
        return self.arr[(self.head + index) % self.size]

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.arr)
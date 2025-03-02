class Generator:
    names = {}

    def __init__(self):
        typeName = type(self).__name__
        genName = typeName[0]+[char for char in typeName[1:] if char.isalpha() and  char not in 'aeiou'][0]
        id = Generator.names.get(genName, 1)
        self.name = f"{genName}{id}".upper()
        Generator.names[genName] = id+1

    def tick(self, t):
        return self.f(t)

    def __str__(self):
        return "%s" % (self.name,)

    def setName(self, name):
        self.name = name

    def f(self, x): return x

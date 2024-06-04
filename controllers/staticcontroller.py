from .controller import Controller


class StaticController(Controller):
    def __init__(self, period, init_cores, name=None):
        super().__init__(period, init_cores, name=name)

    def control(self, t):
        pass

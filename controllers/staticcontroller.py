from .controller import Controller


class StaticController(Controller):
    def __init__(self, period, init_cores):
        super().__init__(period, init_cores)

    def control(self, t):
        pass

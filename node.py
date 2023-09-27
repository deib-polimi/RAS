from applications import Application


class Node:
    def __init__(self, hrz: int, controller, app: Application,  monitoring=None, name="None", generator=None, total_rt=0,
                 subtotal_weight=0.0, total_weight=1.0, fathers={}, local_sla=0.0):
        self.app = app
        self.sla = self.app.sla
        self.horizon = hrz
        self.controller = controller
        self.monitoring = monitoring
        self.name = name
        self.generator = generator
        self.total_rt = total_rt
        self.subtotal_weight = subtotal_weight
        self.total_weight = total_weight
        self.local_sla = local_sla
        self.fathers = fathers


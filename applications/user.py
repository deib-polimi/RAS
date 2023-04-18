if __name__ == "__main__":
    from sockshopmicroservice import SockShopMicroservice
else:
    from .sockshopmicroservice import SockShopMicroservice


class User(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        A1_NOM = 0.00763
        A2_NOM = 0.000026452635352
        A3_NOM = 0.5658

        super().__init__(sla, disturbance, init_cores, A1_NOM, A2_NOM, A3_NOM)
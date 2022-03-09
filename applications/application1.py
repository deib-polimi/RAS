if __name__ == "__main__":
    from application import Application
else:
    from .application import Application

class Application1(Application):
    A1_NOM = 0.00763
    A2_NOM = 0.0018
    A3_NOM = 0.5658
    def __computeRT__(self, req):
        return ((1000.0*self.A2_NOM+self.A1_NOM)*req+1000*self.A1_NOM*self.A3_NOM*self.cores)/(req+1000.0*self.A3_NOM*self.cores)



if __name__ == "__main__":
    app=Application1(sla=0.1)
    for i in range(10):
        print(app.__computeRT__(req=10))
if __name__ == "__main__":
    from application import Application
else:
    from .application import Application
import xml.etree.ElementTree as ET
import subprocess
import numpy as np
import pathlib
import os

class ApplicationMVA(Application):
    
    modelPath="%s/applications/applicationMVA/JMT/tier.jmva"%(os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())))
    JMTPath="%s/applications/applicationMVA/JMT/JMT-singlejar-1.1.1.jar"%(os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())))
    
    def __init__(self,sla=1.0, disturbance=0.0,stime=1.0,init_cores=1):
        super().__init__(sla,disturbance,init_cores)
        self.cores=init_cores
        self.stime=stime
        self.model=None
        self.tree=None
        self.reset()

    def getNUsers(self):
        self.isloaded()
        for classTag in self.model.iter('closedclass'):
            if(classTag.attrib['name'] == 'Users'):
                return classTag.attrib['population']
        
        raise ValueError("User class not defined")


    def getNServers(self):
        self.isloaded()
        for station in self.model.iter('listation'):
            if(station.attrib['name'] == 'tier'):
                return station.attrib['servers']
        
        raise ValueError("Tier station not defined")
    
    
    def getRT(self):
        self.isloaded()
        for res in self.model.iter('stationresults'):
            if(res.attrib['station'] == 'tier'):
                for userClass in res.iter('classresults'):
                    if(userClass.attrib['customerclass'] == 'Users'):
                        for metric in userClass.findall("measure"):
                            if(metric.attrib['measureType'] == 'Residence time'):
                                return metric.attrib['meanValue'] 
        
        raise ValueError("Response time solution not found") 
    
    
    def updateNServers(self,NCores):
        self.isloaded()
        for station in self.model.iter('listation'):
                if(station.attrib['name'] == 'tier'):
                    station.attrib['servers'] = str(NCores)
    
    
    def updateNUsers(self, Nusers):
        self.isloaded()
        for classTag in self.model.iter('closedclass'):
            if(classTag.attrib['name'] == 'Users'):
                classTag.attrib['population'] = str(Nusers)
    
    
    def updateStime(self, stime):
        self.isloaded()
        for station in self.model.iter('listation'):
            if(station.attrib['name'] == 'tier'):
                st=station.find("servicetimes/servicetime[@customerclass='Users']")
                st.text=str(stime)
                
    
    def readModel(self):
        self.tree = ET.parse(self.modelPath)
        self.model= self.tree.getroot()
        
                
    def computeMVART(self):
        exitCode = subprocess.run(["java", "-cp", self.JMTPath, "jmt.commandline.Jmt", "mva", self.modelPath])
        if(exitCode.returncode != 0):
            raise ValueError("JMT Error while computing RT")
        
    def isloaded(self):
        if(self.model==None):
            raise ValueError("Model not loaded! before modifiying the model a base version need to be loaded")
    
    def writeJMVAModel(self):
        self.isloaded()
        self.tree.write(self.modelPath)
    
    def __computeRT__(self, req):
        self.readModel()
        self.updateNUsers(int(req))
        self.updateNServers(int(np.ceil(self.cores)))
        if(req>self.cores):
            self.updateStime(self.stime*np.ceil(self.cores)/self.cores)
        else:
            self.updateStime(self.stime)
        self.writeJMVAModel()
        self.computeMVART()
        self.readModel()
        #print(req,self.cores,int(np.ceil(self.cores)),float(self.getRT()))
        return float(self.getRT())

if __name__ == '__main__':
    import sys,os
    from pathlib import Path
    os.environ["EXTERN"]="True"
    sys.path.append(str(Path(__file__).parent.parent.parent.joinpath("generators")))
    sys.path.append(str(Path(__file__).parent.parent.parent.joinpath("controllers/estimator")))
    import matplotlib.pyplot as plt
    from qnestimator import QNEstimaator
    from singenerator import SinGen
    import casadi
    from application1 import Application1

    class optCtrl():

        def OPTController(self, e, tgt, C, maxCore):
            #print("stime:=", e, "tgt:=", tgt, "user:=", C)
            if(np.sum(C)>0):
                self.model = casadi.Opti() 
                nApp = len(tgt)
                
                T = self.model.variable(1, nApp);
                S = self.model.variable(1, nApp);
                E_l1 = self.model.variable(1, nApp);
                
                self.model.subject_to(T >= 0)
                self.model.subject_to(self.model.bounded(0, S, maxCore))
                self.model.subject_to(E_l1 >= 0)
            
                sSum = 0
                obj = 0;
                for i in range(nApp):
                    sSum += S[0, i]
                    # obj+=E_l1[0,i]
                    obj += (T[0, i] / C[i] - 1 / tgt[i]) ** 2
            
                for i in range(nApp):
                    self.model.subject_to(T[0, i] == casadi.fmin(S[0, i] / e[i], C[i] / e[i]))
            
                self.model.minimize(obj)    
                optionsIPOPT = {'print_time':False, 'ipopt':{'print_level':0}}
                # self.model.solver('osqp',{'print_time':False,'error_on_fail':False})
                self.model.solver('ipopt', optionsIPOPT) 
                
                sol = self.model.solve()
                if(nApp==1):
                    return sol.value(S)
                else:
                    return sol.value(S).tolist()
            else:
                return 10**(-3)


    g = SinGen(500, 700, 200)
    g.setName("SN1") 

    nsample=30
    horizon=300

    estimator=QNEstimaator()
    ctrl=optCtrl()
    appSLA=0.6
    stime=0.05
    initCores=80
    app=ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores)
    #app=Application1(sla=appSLA,init_cores=initCores)

    #cores=np.random.random(nsample)*30
    cores=[initCores]
    usr=[g.f(x) for x in range(0,horizon)]

    rt=[]
    
    e=None
    sIdx=0
    eIdx=0
    e=None
    for i in range(horizon):
        if(len(rt)>0):
            sIdx=max(len(rt)-nsample,0)
            eIdx=min(sIdx+nsample,len(rt))

        #print(f"sIdx={sIdx},eIdx={eIdx}") 

        if(e!=None):
            cores+=[ctrl.OPTController(e=[e], tgt=[appSLA*0.8], C=[usr[i]], maxCore=[10000])]

        app.cores=cores[-1]

        rt+=[app.__computeRT__(usr[i])] 
        e=estimator.estimate(rt=rt[sIdx:eIdx+1],s=cores[sIdx:eIdx+1],c=usr[sIdx:eIdx+1])
        print(f"estim={e},core={app.cores},rt={rt[i]},usr={usr[i]}")

    plt.figure()
    plt.plot(rt,label="Response Time")
    plt.hlines(appSLA,xmin=0,xmax=len(rt),colors="red", linestyles='dashed',label="SLA")
    plt.hlines(0.8*appSLA,xmin=0,xmax=len(rt),colors="blue", linestyles='dashed',label="SET Point")
    plt.grid()
    plt.legend()
    plt.savefig(f"test/rt_{app.__class__.__name__}.pdf")

    plt.figure()
    plt.plot(usr)
    plt.grid()
    plt.savefig(f"test/usr_{app.__class__.__name__}.pdf")

    plt.figure()
    plt.plot(cores)
    plt.grid()
    plt.savefig(f"test/core_{app.__class__.__name__}.pdf")

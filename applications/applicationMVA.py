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
    
    modelPath="%s/JMT/tier.jmva" % (os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())))
    JMTPath="%s/JMT/JMT-singlejar-1.1.1.jar"%(os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())))
    
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
        print(req,self.cores,int(np.ceil(self.cores)),float(self.getRT()))
        return float(self.getRT())

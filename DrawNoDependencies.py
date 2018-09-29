import requests
import cv2
import json

import time

from icansee_api_lib import ICanSeeApiLib

result = None
output = None

# Helper functions

def PostResult(Host, Payload, port = 20000, Uri = None):
    if Uri == None:
            Uri = "http://" + Host + ":" + str(port) + "/MachineMessageApi"
    r = requests.post(Uri, json=Payload)
    return r.text

def ResultToJson(result):
    result = str(result).replace("'", '"')
    result = json.loads(result)
    return result

class DrawProcess:
    Payload = ""
    AlgorithmInSyntax = ""
    Stopper = True
    
    def SetAlgorithmInSyntax(self, algorithmInSyntax: str):
        self.AlgorithmInSyntax = algorithmInSyntax
    
    def SetPayload(self, payloadString : str):
        self.Payload = payloadString
        
    def Draw(self, runOnce:bool = True, infinite:bool = False, limit: int = 1):
        self.Stopper = False
        c = 0
        
        if runOnce == True:
            limit = 1
            infinite = False
        
        while (c < limit):
            globals()["output"] = None
            globals()["result"] = None

            exec(self.AlgorithmInSyntax)
            exec(self.Payload, globals())

            if infinite == False:
                c += 1
            
            if self.Stopper == True:
                break
            
            # time.sleep(0.5)

        return output
    
    def Stop(self):
        self.Stopper = True
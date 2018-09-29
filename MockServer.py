from flask import Flask
from flask_restful import Resource, Api, reqparse
import threading
from DrawNoDependencies import DrawProcess


process = DrawProcess()
process.SetPayload("print(c)")

task = None

app = Flask(__name__)
api = Api(app)

class TaskServer(Resource):
    def get(self):
        return {'hello': 'world'}, 200
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('Fbp', type=list, action='append')
        parser.add_argument('RunOnce', type=bool)
        parser.add_argument('InfiniteLoop', type=bool)
        parser.add_argument('LoopLimit', type=int)
        parser.add_argument('ReturnResult', type=bool)
        args = parser.parse_args()
        
        RunOnce = args["RunOnce"]
        InfiniteLoop = args["InfiniteLoop"]
        LoopLimit = args["LoopLimit"]
        ReturnResult = args["ReturnResult"]
        
        Fbp = []
        separator = ''
        for i in range(len(args["Fbp"])):
            Fbp.append(separator.join(args["Fbp"][i]))
        
        for val in Fbp:
            print(val)
        
        if len(Fbp) < 3:
            return "Fbp should contain 3 lines", 200
        
        process.SetPayload(Fbp[2])
        process.SetAlgorithmInSyntax(Fbp[1])
        
        global task
        if task != None:
            process.Stop()
        
        if ReturnResult == False:
            task = threading.Thread(target=process.Draw, args=(RunOnce, InfiniteLoop, LoopLimit))
            task.start()
        else:
            return (process.Draw(RunOnce, InfiniteLoop, LoopLimit)), 200
        
        return "AckP", 200
    
    def delete(self):
        process.Stopper = True
        return "AckD", 200

api.add_resource(TaskServer, '/task')

app.run(host = "0.0.0.0")
from fastapi import FastAPI
from api.robot_api import RobotApi
from model.response import Position
from neuralnetwork.neural_network import NeuralNetwork
import uvicorn

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"} 

@app.get("/status")
def status():
    return 

@app.get("/getObstacle")
async def getObstacle():
    robot_api = RobotApi()
    response = await robot_api.make_request("GET")
    return response

@app.post("/processObstacle")
async def processObstacle(request: Position):
    robot_api = RobotApi()
    neural_network = NeuralNetwork()
    nn_output = neural_network.process(request=request)
    response = await robot_api.make_request("POST", nn_output)
    if(response.Resp == "KO"):
        return response
    else:
        response = await robot_api.make_request("GET")
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
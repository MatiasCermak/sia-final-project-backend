from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.robot_api import RobotApi
from model.response import Position, Response
from neuralnetwork.neural_network import NeuralNetwork
import uvicorn
from utils.constants import INITIAL_WEIGHTS

app = FastAPI()
neural_network = NeuralNetwork(INITIAL_WEIGHTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    nn_output = neural_network.process(request=request)
    response = await robot_api.make_request("POST", nn_output)
    
    if(response.Resp == "OK"):
        response = await robot_api.make_request("GET")
        response.Resp.__setattr__("M1", nn_output.M1)
        response.Resp.__setattr__("M2", nn_output.M2)
    response.__setattr__("NNResponse", nn_output);
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
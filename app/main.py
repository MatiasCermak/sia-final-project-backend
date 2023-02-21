from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.robot_api import RobotApi
from model.response import Position
from neuralnetwork.neural_network import NeuralNetwork
import uvicorn

app = FastAPI()
neural_network = NeuralNetwork()

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
        datos = nn_output.to_array()
        neural_network.update(x=[datos[0:4]], y1=datos[4], y2=datos[5])
        response = await robot_api.make_request("GET")
        response.Resp.__setattr__("M1", nn_output.M1)
        response.Resp.__setattr__("M2", nn_output.M2)
    response.__setattr__("NNResponse", nn_output);
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
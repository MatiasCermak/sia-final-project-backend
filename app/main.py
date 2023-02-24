from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.robot_api import RobotApi
from model.response import Position, Response
from neuralnetwork.neural_network import NeuralNetwork
import uvicorn
from utils.constants import TESTING_SET, INITIAL_WEIGHTS, TRAINING_SET

app = FastAPI()
neural_network = NeuralNetwork(weights=INITIAL_WEIGHTS,learning_rate=[0.0076, 0.001], use_bias=[True, False], epochs=[10000, 500], bias=[0.76, 0.01])

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
    response = robot_api.make_request("POST", nn_output)
    
    if(response.Resp == "OK"):
        response = robot_api.make_request("GET")
        response.Resp.__setattr__("M1", nn_output.M1)
        response.Resp.__setattr__("M2", nn_output.M2)
    response.__setattr__("NNResponse", nn_output);
    return response

@app.get("/test")
async def testNetwork():
    errors = 0
    robot_api = RobotApi()
    distinct_positions = []
    respuestas = ""
    while len(distinct_positions) < 9 :
        get_response = robot_api.make_request("GET")
        if get_response.Resp in distinct_positions:
            continue
        distinct_positions.append(get_response.Resp)
        nn_output = neural_network.process(request=get_response.Resp)
        post_response = robot_api.make_request("POST", nn_output)
        post_response.__setattr__("NNResponse", nn_output)
        respuestas += "Resp: " + str(post_response.Resp) + "Desc: " + post_response.Desc + "NNResponse: " + str(post_response.NNResponse) + "\n"
        if(post_response.Resp == "KO"):
            errors += 1
    post_response.__setattr__("Resp", "Testeo completo")
    post_response.__setattr__("Desc", "Errores en testeo: "+str(errors))
    post_response.__setattr__("NNResponse", respuestas)
    return post_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
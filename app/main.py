from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.robot_api import RobotApi
from model.response import Position, Response
from neuralnetwork.neural_network import NeuralNetwork
import uvicorn
from utils.constants import TESTING_SET, INITIAL_WEIGHTS, TRAINING_SET, LEARNING_RATE, BIAS, USE_BIAS, EPOCHS

app = FastAPI()

NEURAL_NETWORK_MODELS = {
    'p-v': NeuralNetwork(weights=INITIAL_WEIGHTS, learning_rate=LEARNING_RATE, use_bias=USE_BIAS, epochs=EPOCHS, bias=BIAS, vectorizar=True),
    'p-nv': NeuralNetwork(weights=INITIAL_WEIGHTS, learning_rate=LEARNING_RATE, use_bias=USE_BIAS, epochs=EPOCHS, bias=BIAS, vectorizar=False)
}
NN_MODEL = 'p-v'
NEURAL_NETWORK = NEURAL_NETWORK_MODELS.get(NN_MODEL)
NEURAL_NETWORK_TESTS = []

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
    response = robot_api.make_request("GET")
    return response


@app.post("/processObstacle")
async def processObstacle(request: Position):
    robot_api = RobotApi()
    nn_output = NEURAL_NETWORK.process(request=request)
    response = robot_api.make_request("POST", nn_output)

    if (response.Resp == "OK"):
        response = robot_api.make_request("GET")
        response.Resp.__setattr__("M1", nn_output.M1)
        response.Resp.__setattr__("M2", nn_output.M2)
    response.__setattr__("NNResponse", nn_output)
    return response


@app.get("/test")
def testNetwork():
    global NEURAL_NETWORK_TESTS
    NEURAL_NETWORK_TESTS.clear()
    robot_api = RobotApi()

    for id in NEURAL_NETWORK_MODELS:
        errors = 0
        distinct_positions = []
        respuestas = ""
        while len(distinct_positions) < 9:
            get_response = robot_api.make_request("GET")
            if get_response.Resp in distinct_positions:
                continue
            distinct_positions.append(get_response.Resp)
            nn_output = NEURAL_NETWORK_MODELS.get(id).process(request=get_response.Resp)
            post_response = robot_api.make_request("POST", nn_output)
            post_response.__setattr__("NNResponse", nn_output)
            respuestas += "Resp: " + str(post_response.Resp) + "Desc: " + \
                post_response.Desc + "NNResponse: " + \
                str(post_response.NNResponse) + "\n"
            if (post_response.Resp == "KO"):
                errors += 1
        post_response.__setattr__("Resp", "Testeo completo - " + id)
        post_response.__setattr__("Desc", "Errores en testeo: "+str(errors))
        post_response.__setattr__("NNResponse", respuestas)
        NEURAL_NETWORK_TESTS.append(post_response)
    


@app.get("/changeModel/{model}")
async def changeModel(model: str):
    global NEURAL_NETWORK
    global NN_MODEL
    if model == NN_MODEL:
        return {'status': "OK",
                "bias": BIAS,
                "useBias": USE_BIAS,
                "epochs": EPOCHS,
                "initialWeights": INITIAL_WEIGHTS,
                "learningRate": LEARNING_RATE,
                "tests": NEURAL_NETWORK_TESTS}

    nnetwork = NEURAL_NETWORK_MODELS.get(model)
    if (nnetwork != None):
        NEURAL_NETWORK = nnetwork
        NN_MODEL = model
        return {'status': "OK",
                "bias": BIAS,
                "useBias": USE_BIAS,
                "epochs": EPOCHS,
                "initialWeights": INITIAL_WEIGHTS,
                "learningRate": LEARNING_RATE,
                "tests": NEURAL_NETWORK_TESTS}

    return {'status': "KO",
            "bias": BIAS,
            "useBias": USE_BIAS,
            "epochs": EPOCHS,
            "initialWeights": INITIAL_WEIGHTS,
            "learningRate": LEARNING_RATE,
                "tests": NEURAL_NETWORK_TESTS}

if __name__ == "__main__":
    print("Please wait, running tests...")
    testNetwork()
    uvicorn.run(app, host="0.0.0.0", port=8080)


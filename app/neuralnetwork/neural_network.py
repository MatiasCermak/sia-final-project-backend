from multiple_perceptron import MultiplePerceptron as MP
from model.response import Position
import numpy as np
from utils.constants import TRAINING_SET

class NeuralNetwork():
    def __init__(self):
        self.__net = MP(learning_rate=0.35, class2=-1)
        self.train(TRAINING_SET[1], TRAINING_SET[2], X=TRAINING_SET[0])
    
    def train(self, *args, X):
        self.__net.fit(args=args, X=X)
    
    def process(self, request):
        data = request.to_array()
        result = Position()
        entry = [data[0:4]]
        prediction = self.__net.predict_set(entry)
        prediction = prediction.reshape(1,2)[0]
        result.from_array(entry+prediction)
        return result
    

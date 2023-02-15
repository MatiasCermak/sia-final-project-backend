from neuralnetwork.multiple_perceptron import MultiplePerceptron as MP
from model.response import Position
from utils.constants import TRAINING_SET

class NeuralNetwork():
    __train_set = []

    def __init__(self):
        self.__net = MP(learning_rate=0.35, class2=-1)
        self.__train_set = TRAINING_SET
        self.train(TRAINING_SET[1], TRAINING_SET[2], X=TRAINING_SET[0])
    
    def train(self, args, X):
        self.__net.fit(args=args, X=X)
    
    def process(self, request):
        data = request.to_array()
        result = Position()
        entry = [data[0:4]]
        prediction = self.__net.predict_set(entry)
        prediction = prediction
        result.from_array(entry[0]+prediction)
        return result
    
    def update(self, x, y1, y2):
        if not x in self.__train_set[0]:
            new_entry = self.__train_set[0] + x
            new_target_1 = self.__train_set[1] + y1
            new_target_2 = self.__train_set[2] + y2
            self.train(new_target_1, new_target_2, X=new_entry)
            self.__train_set = [new_entry, new_target_1, new_target_2]

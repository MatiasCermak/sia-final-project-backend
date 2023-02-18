from neuralnetwork.simple_perceptron import SimplePerceptron as SP

class MultiplePerceptron():
    def __init__(self, learning_rate=0.1, bias=0.001, class1=1, class2=0, epochs=1000, use_bias=True):
        self.__learning_rate = learning_rate
        self.__bias = bias
        self.__use_bias = use_bias
        self.__class1 = class1
        self.__class2 = class2
        self.__epochs = epochs
    
    def fit(self, args, X):
        self.__simple_perceptrons = []
        for target in args:
            simple = SP(learning_rate=self.__learning_rate, 
                        bias=self.__bias, 
                        class1=self.__class1, 
                        class2=self.__class2, 
                        epochs=self.__epochs,
                        use_bias=self.__use_bias)
            simple.fit(X=X, y=target)
            self.__simple_perceptrons.append(simple)
    
    def predict_set(self, X):
        outputs = []
        for simple_perceptron in self.__simple_perceptrons:
            outputs.append(simple_perceptron.predict_set(X)[0])
        return outputs
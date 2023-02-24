import numpy as np

class NVSimplePerceptron():
    def __init__(self, learning_rate=0.1, bias=0.001, class1=1, class2=0, epochs=1000, use_bias=True, weights=[]):
        self.__learning_rate = learning_rate
        self.__bias = bias
        self.__use_bias = use_bias
        if not use_bias:
            self.__bias = 0
        self.__class1 = class1
        self.__class2 = class2
        self.__epochs = epochs
        self.__weights = weights
    
    def fit(self, X, y):
        if len(self.__weights) == 0:
            self.__set_weights(X)
        for _ in range(self.__epochs):
            for xi, target in zip(X, y):
                prediction = self.__predict(xi)
                error = target - prediction
                update = self.__learning_rate * error
                for index in range(len(xi)):
                    self.__weights[index] += float(update) * float(xi[index])
                if self.__bias < 1 and self.__use_bias:
                    self.__bias += update
        print(self.__weights)
    
    def __predict(self, X):
        activation = 0
        for index in range(len(X)):
            activation += X[index] * self.__weights[index]
        activation += self.__bias
        return self.__class1 if activation >= 0 else self.__class2
    
    def predict_set(self, X):
        output = []
        for entry in X:
            output.append(self.__predict(entry))
        return output
    
    def __set_weights(self, X):
        for _ in range(len(X[0])):
            self.__weights.append(np.random.uniform(low=-0.1, high=0.1, size=1))


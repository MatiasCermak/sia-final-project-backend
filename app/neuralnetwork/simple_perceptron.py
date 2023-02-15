import numpy as np

class SimplePerceptron():
    def __init__(self, learning_rate=0.1, bias=0, class1=1, class2=-1, epochs=1000):
        self.__learning_rate = learning_rate
        self.__bias = bias
        self.__class1 = class1
        self.__class2 = class2
        self.__epochs = epochs
    
    def fit(self, X, y):
        self.__weights = np.random.random(np.shape(X[1]))
        for _ in range(self.__epochs):
            for xi, target in zip(X, y):
                prediction = self.__predict(xi)
                error = target - prediction
                update = self.__learning_rate * error
                self.__weights += np.multiply(xi, update)
    
    def __predict(self, X):
        activation = np.dot(X, self.__weights) + self.__bias
        return self.__class1 if activation >= 0 else self.__class2
    
    def predict_set(self, X):
        output = []
        for entry in X:
            output.append(self.__predict(entry))
        return output

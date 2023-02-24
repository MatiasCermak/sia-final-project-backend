import numpy as np
    
def tanH(x):
    return np.divide(np.exp(x)-np.exp(np.negative(x)),np.exp(x)+np.exp(np.negative(x)))

def sigmoide(x):
    return np.divide(1,1+np.exp(np.negative(x)))

class BackPropagation():
    SIGMOIDE = sigmoide
    TANH = tanH

    def __init__(self, input_layer_neurons=4, hidden_layers_quantity=1, hidden_layers_neurons=[3],
                 output_layer_neurons=2, error_tolerance=0.01, training_patterns_quantity=7, activation_function=SIGMOIDE, epochs=1000):
        self.__error_tolerance = error_tolerance
        self.__training_pattenrs_quantity = training_patterns_quantity
        self.__n_input_number = input_layer_neurons +1
        self.__n_output_number = output_layer_neurons
        self.__n_hidden_layers = hidden_layers_quantity
        self.__nn_hidden_layers = hidden_layers_neurons
        for index in range(self.__n_hidden_layers):
            if self.__nn_hidden_layers[index] == -1:
                self.__nn_hidden_layers[index] = self.__BaumHaussler(
                    self.__nn_hidden_layers[index-1] if index > 0 else self.__n_input_number,
                    self.__nn_hidden_layers[index-1] if index < self.__n_hidden_layers -1 else self.__n_output_number)
            else:
                self.__nn_hidden_layers[index] += 1
        self.__activation_f = activation_function
        self.__epochs = epochs
    
    def train(self, X, Y):
        '''
            w son los pesos: wij con i =0 a n j=0 a 
            x es el set de entrada: xi con i=1 a a
            h capa oculta: hi con i=1 a h
            s capa salida: si con i=1 a s

            inicializar los pesos con valores aleatorios entre -0.1 y 0.1
            x0 = 1
            h0 = 1
            propagar activaciones desde capa de entrada a capa oculta: hj = activación(sumatoria(pesos * entrada))
            propagar activaciones desde capa oculta a las demás o a la capa de salida (Oj)
            calcular error por cada neurona de salida: d2j: (1 - Oj)(yj - Oj)
            calcular errores de las unidades de capa oculta: d1j : (1 - hj)(sumatoria(d2j * w2ji)) siendo w2ji los pesos entre 
                                la capa de salida y la capa oculta, iterando por cada j por todos los pesos que le qfectan
            ajustar pesos entre capa de salida y capa oculta: alfa * d2j * hi
            ajustar pesos entre capa oculta y capa de entrada: alfa * d1j * xi
            repetir pasos con la siguiente entrada y tantas épocas como sean necesarias hasta consegir un error menor o igual al tolerado
        '''
        if len(X) != len(Y):
            print("NO SE PUEDE REALIZAR EL ENTRENAMIENTO POR FALTA DE DATOS")
        if (len(X) > 0 and len(X[0])+1 != self.__n_input_number) or (len(Y) > 0 and len(Y[0]) != self.__n_output_number):
            print("LOS TAMAÑOS DE LOS SET NO COINCIDEN CON LOS SETEADOS INICIALMENTE. REVISE DATOS")
        
        self.__generate_weights()




    def __BaumHaussler(self, input=1, output=1):
        return np.divide(np.multiply(self.__training_pattenrs_quantity,self.__error_tolerance),
                         np.multiply(input,output))
    
    def __generate_weights(self):
        self.__weights = []
        if self.__n_hidden_layers == 0:
            self.__weights = [np.random.uniform(low=-0.1, high=0.1, 
                                               size=(int(self.__n_input_number), int(self.__n_output_number)))]
        else:
            self.__weights.append(np.random.uniform(low=-0.1, high=0.1, 
                                               size=(int(self.__n_input_number), int(self.__nn_hidden_layers[0]))))
            for index in range(self.__n_hidden_layers -1):
                self.__weights.append(np.random.uniform(low=-0.1, high=0.1, 
                                               size=(int(self.__nn_hidden_layers[index]), 
                                                     int(self.__nn_hidden_layers[index+1]))))
            self.__weights.append(np.random.uniform(low=-0.1, high=0.1, 
                                               size=(int(self.__nn_hidden_layers[self.__n_hidden_layers-1]), 
                                                     int(self.__n_output_number))))
    
    def forward(self, X, W):
        return self.__activation_f(np.dot([1]+X, W.T))
    
    def backward(self, Y, X):
        errors = [self.output_error(X=X,Y=Y)]
        hidden_layers = [self.forward(X,self.__weights[0])]
        for index in range(1,len(self.__weights)):
            hidden_layers.append(self.forward(hidden_layers[index - 1],self.__weights[index]))
        for index in range(len(hidden_layers)-1,-1,-1):
            errors = np.insert(errors, 0, self.delta_error(hidden_layers[index],self.__weights[index+1],errors[0]))
        

    
    def output_error(self, X, Y):
        output = self.predict(X)
        return np.multiply(1-output, Y - output)
    
    def delta_error(self, X, index, error):
        return np.multiply(1-X,np.dot(self.__weights[index],error))

    def predict(self, X):
        if self.__n_hidden_layers == 0:
            return self.forward(X, self.__weights[0])
        else:
            output_layer = []
            for index in range(self.__n_hidden_layers):
                output_layer = self.forward(X=X,W=self.__weights[index])
            return output_layer
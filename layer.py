import numpy as np

class Dense:
    def __init__(self, size=3, activation='sigmoid'):
        self.size = size
        self.activation = activation

    def initializeParameters(self, prevLayerSize):
        self.weights = np.random.randn(self.size, prevLayerSize) * 0.01
        self.bias = np.zeros((self.size, 1))

    def getZScore(self, prevA):
        Z = np.dot(self.weights, prevA) + self.bias
        return Z

    def getActivationValue(self, Z):
        if(self.activation == 'sigmoid'):
            return 1 / (1 + np.exp(-1 * Z))
        elif(self.activation == 'tanh'):
            return np.tanh(Z)
        elif(self.activation == 'relu'):
            return np.maximum(Z, 0)
        elif(self.activation == 'leaky_relu'):
            return np.maximum(Z, 0.01)
        elif(self.activation == 'softmax'):
            tSum = np.sum(np.exp(Z), axis=0)
            return np.exp(Z) / tSum

    def getActivationDerivative(self, Z, activationValue):
        if(self.activation == 'sigmoid'):
            return activationValue * (1 - activationValue)
        elif(self.activation == 'tanh'):
            return 1 - np.square(activationValue)
        elif(self.activation == 'relu'):
            belowZero = Z < 0
            aboveZero = Z >= 0
            Z[belowZero] = 0
            Z[aboveZero] = 1
            return Z
        elif(self.activation == 'leaky_relu'):
            belowZero = Z < 0
            aboveZero = Z >= 0
            Z[belowZero] = 0.01
            Z[aboveZero] = 1
            return Z
        elif(self.activation == 'softmax'):
            pass # TODO Add this

    def gradientDescentStep(self, learningRate, dW, db):
        self.weights -= learningRate * dW
        self.bias -= learningRate * db
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

    def getActivationDerivative(self, activationValue):
        if(self.activation == 'sigmoid'):
            return activationValue * (1 - activationValue)
        elif(self.activation == 'tanh'):
            return 1 - np.square(activationValue)

    def gradientDescentStep(self, learningRate, dW, db):
        self.weights -= learningRate * dW
        self.bias -= learningRate * db
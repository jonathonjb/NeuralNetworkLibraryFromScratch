import numpy as np
import random

class NeuralNetwork:
    def __init__(self, inputSize, layers):
        self.inputSize = inputSize
        self.layers = layers
        self.L = len(layers)
        self.initializeWeights()

    def initializeWeights(self):
        prevSize = self.inputSize
        for layer in self.layers:
            layer.initializeParameters(prevLayerSize=prevSize)
            prevSize = layer.size

    def train(self, X, y, iterations=100000, learningRate=0.1, miniBatch=False, miniBatchSize=32,
              printCost=True, printCostRounds=1000):
        return self.gradientDescent(X, y, iterations, learningRate, miniBatch, miniBatchSize, printCost, printCostRounds)

    def predict(self, X):
        AF, cache = self.forwardPropagation(X)
        return np.round(AF, decimals=0)

    def gradientDescent(self, X, y, iterations, learningRate, miniBatch, miniBatchSize, printCost, printCostRounds):
        if(miniBatch):
            self.m = miniBatchSize
        else:
            self.m = X.shape[1]

        costHistory = []
        X_batch = X
        y_batch = y

        for i in range(iterations):
            if(miniBatch):
                X_batch, y_batch = self.generateMiniBatch(X, y)

            AL, cache = self.forwardPropagation(X_batch)
            cost = self.computeCost(AL, y_batch)
            costHistory.append(cost)
            if(printCost and i % printCostRounds == 0):
                print('Cost after round', i, '-', cost)
            self.backPropagation(AL, y_batch, cache, learningRate)
        return costHistory

    def tempSetM(self, m):
        self.m = m

    def generateMiniBatch(self, X, y):
        indexes = random.sample(range(0, X.shape[1]), self.m)
        return X[:, indexes], y[indexes]

    def forwardPropagation(self, X):
        prevA = X
        l = 1
        cache = {}
        cache['A0'] = X
        for layer in self.layers:
            Z = layer.getZScore(prevA)
            prevA = layer.getActivationValue(Z)
            cache['W' + str(l)] = layer.weights
            cache['A' + str(l)] = prevA
            l += 1
        return prevA, cache

    def computeCost(self, A, y):
        return (-1/self.m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

    def backPropagation(self, AL, y, cache, learningRate):
        startFlag = True
        dA = None
        l = self.L
        for layer in reversed(self.layers):
            W = cache.get('W' + str(l))
            prevA = cache.get('A' + str(l - 1))

            if(startFlag):
                dZ = AL - y
                startFlag = False
            else:
                dZ = dA * layer.getActivationDerivative(cache.get('A' + str(l)))

            dW = (1/self.m) * np.dot(dZ, prevA.T)
            db = (1/self.m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(W.T, dZ)
            layer.gradientDescentStep(learningRate, dW, db)

            l -= 1

    def prettyPrint(self):
        print('INPUT SIZE:', self.inputSize, '\n')
        i = 0
        for layer in self.layers:
            weights = layer.weights
            print('LAYER', i, '\n-------------------------')
            for row in range(weights.shape[0]):
                print('\t', layer.bias[row, 0], end='')
                for col in range(weights.shape[1]):
                    print(' + (' + str(weights[row, col]), '* x' + str(col + 1), end=')')
                print()
            print()
            i += 1
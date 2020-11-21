import numpy as np
import random
import math

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

    def train(self, X, y, epochs=100000, learningRate=0.1, miniBatch=False, miniBatchSize=32,
              printCost=True, printCostRounds=1000):

        return self.gradientDescent(X, y, epochs, learningRate, miniBatch, miniBatchSize, printCost, printCostRounds)

    def predict(self, X):
        AF, cache = self.forwardPropagation(X)
        return np.round(AF, decimals=0)

    def gradientDescent(self, X, y, epochs, learningRate, miniBatch, miniBatchSize, printCost, printCostRounds):
        costHistory = []
        X_batches = [X]
        y_batches = [y]

        round = 0
        for i in range(epochs):
            if(miniBatch):
                X_batches, y_batches = self.generateMiniBatches(X, y, miniBatchSize)

            if(i % 10000 == 0):
                print('Epoch:', i)

            # Each iteration is a gradient descent iteration
            for X_batch, y_batch in zip(X_batches, y_batches):
                self.m = X_batch.shape[1]

                AL, cache = self.forwardPropagation(X_batch)
                cost = self.computeCost(AL, y_batch)
                costHistory.append(cost)
                self.backPropagation(AL, y_batch, cache, learningRate)

                if(printCost and round % printCostRounds == 0):
                    print('Cost after round', round, '-', cost)
                round += 1

        return costHistory

    def generateMiniBatches(self, X, y, miniBatchSize):
        indexes = list(range(0, X.shape[1]))
        random.shuffle(indexes)

        X_batches = []
        y_batches = []

        numFullMiniBatches = math.floor(X.shape[1] / miniBatchSize)
        for i in range(numFullMiniBatches):
            X_batches.append(X[:, indexes[i * miniBatchSize : (i+1) * miniBatchSize]])
            y_batches.append(y[indexes[i * miniBatchSize: (i + 1) * miniBatchSize]])

        X_batches.append(X[:, miniBatchSize * numFullMiniBatches:])
        y_batches.append(y[miniBatchSize * numFullMiniBatches:])

        return X_batches, y_batches

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
import numpy as np
import random
import math
import statistics

class NeuralNetwork:
    def __init__(self, inputSize, layers):
        self.inputSize = inputSize
        self.layers = layers
        self.L = len(layers)
        self.initializeWeights()
        self.optimization = None

    def initializeWeights(self):
        prevSize = self.inputSize
        for layer in self.layers:
            layer.initializeParameters(prevLayerSize=prevSize)
            prevSize = layer.size

    def compile(self, optimization=None):
        self.optimization = optimization
        if(optimization=='adam'):
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = math.pow(10, -8)

    def train(self, X, y, epochs=100000, learningRate=0.1, miniBatch=False, miniBatchSize=32,
              printCosts=True, printCostRounds=1000):

        costHistory = []
        X_batches = [X]
        y_batches = [y]

        for i in range(epochs):
            if(self.optimization == 'adam'):
                self.V = dict()
                self.S = dict()
                for l in range(self.L):
                    layer = self.layers[l]
                    WShape = layer.weights.shape
                    bShape = layer.bias.shape
                    self.V['dW' + str(l + 1)] = np.zeros(WShape)
                    self.V['db' + str(l + 1)] = np.zeros(bShape)
                    self.S['dW' + str(l + 1)] = np.zeros(WShape)
                    self.S['db' + str(l + 1)] = np.zeros(bShape)

            if (miniBatch):
                X_batches, y_batches = self.generateMiniBatches(X, y, miniBatchSize)

            self.gradientDescent(X_batches, y_batches, costHistory, learningRate)
            if (printCosts and i % printCostRounds == 0):
                self.printCost(costHistory, i, miniBatch, miniBatchSize)

        return costHistory

    def predict(self, X):
        AF, cache = self.forwardPropagation(X)
        return np.round(AF, decimals=0)

    def generateMiniBatches(self, X, y, miniBatchSize):
        indexes = list(range(0, X.shape[1]))
        random.shuffle(indexes)

        X_batches = []
        y_batches = []

        numFullMiniBatches = math.floor(X.shape[1] / miniBatchSize)
        for i in range(numFullMiniBatches):
            X_batches.append(X[:, indexes[i * miniBatchSize : (i+1) * miniBatchSize]])
            y_batches.append(y[indexes[i * miniBatchSize: (i + 1) * miniBatchSize]])

        if(X.shape[1] % miniBatchSize != 0):
            X_batches.append(X[:, miniBatchSize * numFullMiniBatches:])
            y_batches.append(y[miniBatchSize * numFullMiniBatches:])

        return X_batches, y_batches

    def gradientDescent(self, X_batches, y_batches, costHistory, learningRate):
        for X_batch, y_batch in zip(X_batches, y_batches):
            self.m = X_batch.shape[1]

            AL, cache = self.forwardPropagation(X_batch)
            cost = self.computeCost(AL, y_batch)
            costHistory.append(cost)
            self.backPropagation(AL, y_batch, cache, learningRate)

    def forwardPropagation(self, X):
        prevA = X
        l = 1
        cache = {}
        cache['A0'] = X
        for layer in self.layers:
            Z = layer.getZScore(prevA)
            prevA = layer.getActivationValue(Z)
            cache['W' + str(l)] = layer.weights
            cache['Z' + str(l)] = Z
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
            Z = cache.get('Z' + str(l))
            A = cache.get('A' + str(l))
            prevA = cache.get('A' + str(l - 1))

            if(startFlag):
                dZ = AL - y
                startFlag = False
            else:
                dZ = dA * layer.getActivationDerivative(Z, A)

            dW = (1/self.m) * np.dot(dZ, prevA.T)
            db = (1/self.m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(W.T, dZ)

            if(self.optimization == 'adam'):
                self.V['dW' + str(l)] = self.beta1 * self.V['dW' + str(l)] + (1 - self.beta1) * dW
                self.V['db' + str(l)] = self.beta1 * self.V['db' + str(l)] + (1 - self.beta1) * db
                self.S['dW' + str(l)] = self.beta2 * self.S['dW' + str(l)] + (1 - self.beta2) * np.square(dW)
                self.S['db' + str(l)] = self.beta2 * self.S['db' + str(l)] + (1 - self.beta2) * np.square(db)

                V_dW_corrected = self.V['dW' + str(l)] / (1 - self.beta1)
                V_db_corrected = self.V['db' + str(l)] / (1 - self.beta1)
                S_dW_corrected = self.S['dW' + str(l)] / (1 - self.beta2)
                S_db_corrected = self.S['db' + str(l)] / (1 - self.beta2)

                dW = V_dW_corrected / (np.sqrt(S_dW_corrected) + self.epsilon)
                db = V_db_corrected / (np.sqrt(S_db_corrected) + self.epsilon)

            layer.gradientDescentStep(learningRate, dW, db)

            l -= 1

    def printCost(self, costHistory, i, miniBatch, miniBatchSize):
        if (miniBatch):
            cost = statistics.mean(costHistory[-1 * miniBatchSize:])
        else:
            cost = costHistory[-1]
        print('Cost after epoch', i, '-', cost)

    def prettyPrint(self):
        print('INPUT SIZE:', self.inputSize, '\n')
        i = 0
        for layer in self.layers:
            weights = layer.weights
            print('LAYER', i, '- Mean:', layer.weights.mean(), ', STD:', layer.weights.std())
            print('-------------------------')
            for row in range(weights.shape[0]):
                print('\t', layer.bias[row, 0], end='')
                for col in range(weights.shape[1]):
                    print(' + (' + str(weights[row, col]), '* x' + str(col + 1), end=')')
                print()
            print()
            i += 1
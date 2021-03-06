import numpy as np
import random
import math
import statistics

# np.seterr(all='ignore')

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

    def compile(self, optimization=None, normalization=None):
        self.optimization = optimization
        self.normalization = normalization
        if(optimization=='adam' or optimization=='momentum' or optimization=='RMSprop'):
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = math.pow(10, -8)
        if(normalization=='instance'):
            self.normalization = 'instance'


    def train(self, X, y, epochs=10000, learningRate=0.1, miniBatchSize=32,
              regularization='L2', lambdaReg=0.1, decayRate=None,
              printCostRounds=1000):

        if(self.normalization == 'instance'):
            X = self.applyInstanceNormalizationTraining(X)

        costHistory = []
        X_batches = [X]
        y_batches = [y]

        for i in range(epochs):
            if(decayRate != None):
                learningRate *= 1 / (1 + decayRate * i)
            self.initOptimization()

            if (miniBatchSize != None):
                X_batches, y_batches = self.generateMiniBatches(X, y, miniBatchSize)

            self.gradientDescent(X_batches, y_batches, costHistory, learningRate, regularization, lambdaReg)
            if (printCostRounds != None and i % printCostRounds == 0):
                self.printCost(costHistory, i, miniBatchSize)

        return costHistory

    def predict(self, X):
        if(self.normalization == 'instance'):
            X = self.applyInstanceNormalizationTest(X)

        AF, cache = self.forwardPropagation(X)
        predictions = np.argmax(AF, axis=0)
        return predictions

    def generateMiniBatches(self, X, y, miniBatchSize):
        indexes = list(range(0, X.shape[1]))
        random.shuffle(indexes)

        X_batches = []
        y_batches = []

        numFullMiniBatches = math.floor(X.shape[1] / miniBatchSize)
        for i in range(numFullMiniBatches):
            X_batches.append(X[:, indexes[i * miniBatchSize : (i+1) * miniBatchSize]])
            y_batches.append(y[:, indexes[i * miniBatchSize: (i + 1) * miniBatchSize]])

        if(X.shape[1] % miniBatchSize != 0):
            X_batches.append(X[:, miniBatchSize * numFullMiniBatches:])
            y_batches.append(y[:, miniBatchSize * numFullMiniBatches:])

        return X_batches, y_batches

    def gradientDescent(self, X_batches, y_batches, costHistory, learningRate, regularization, lambdaReg):
        for X_batch, y_batch in zip(X_batches, y_batches):
            self.m = X_batch.shape[1]

            AL, cache = self.forwardPropagation(X_batch)
            cost = self.computeCost(AL, y_batch, regularization, lambdaReg)
            costHistory.append(cost)
            self.backPropagation(AL, y_batch, cache, learningRate, regularization, lambdaReg)

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

    # uses the cross entropy loss function
    def computeCost(self, A, y, regularization, lambdaReg):
        regularization_cost = 0
        if(regularization == 'L2'):
            WSquaredSum = 0
            for layer in self.layers:
                WSquaredSum += np.sum(np.square(layer.weights))
            regularization_cost = lambdaReg / (2 * self.m) * WSquaredSum
        return (1/self.m) * np.sum(-1 * np.sum(y * np.log(A), axis=0)) + regularization_cost
        # return (-1/self.m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) + regularization_cost

    def backPropagation(self, AL, y, cache, learningRate, regularization, lambdaReg):
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

            if(regularization == 'L2'):
                dW += lambdaReg / self.m * layer.weights

            dW, db = self.optimize(dW, db, l)
            layer.gradientDescentStep(learningRate, dW, db)

            l -= 1

    def initOptimization(self):
        if (self.optimization == 'adam' or self.optimization == 'momentum' or self.optimization == 'RMSprop'):
            self.V = dict()
            self.S = dict()
            for l in range(self.L):
                layer = self.layers[l]
                WShape = layer.weights.shape
                bShape = layer.bias.shape
                if(self.optimization == 'momentum' or self.optimization == 'adam'):
                    self.V['dW' + str(l + 1)] = np.zeros(WShape)
                    self.V['db' + str(l + 1)] = np.zeros(bShape)
                if (self.optimization == 'RMSprop' or self.optimization == 'adam'):
                    self.S['dW' + str(l + 1)] = np.zeros(WShape)
                    self.S['db' + str(l + 1)] = np.zeros(bShape)


    def optimize(self, dW, db, l):
        if (self.optimization == 'adam'):
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
        elif (self.optimization == 'momentum'):
            self.V['dW' + str(l)] = self.beta1 * self.V['dW' + str(l)] + (1 - self.beta1) * dW
            self.V['db' + str(l)] = self.beta1 * self.V['db' + str(l)] + (1 - self.beta1) * db
            dW = self.V['dW' + str(l)] / (1 - self.beta1)
            db = self.V['db' + str(l)] / (1 - self.beta1)
        elif (self.optimization == 'RMSprop'):
            self.S['dW' + str(l)] = self.beta1 * self.S['dW' + str(l)] + (1 - self.beta1) * np.square(dW)
            self.S['db' + str(l)] = self.beta1 * self.S['db' + str(l)] + (1 - self.beta1) * np.square(db)
            dW = self.S['dW' + str(l)] / (1 - self.beta1)
            db = self.S['db' + str(l)] / (1 - self.beta1)

        return dW, db

    def applyInstanceNormalizationTraining(self, X):
        self.mean = np.mean(X, axis=1, keepdims=True)
        X_new = X - self.mean
        self.std = np.std(X_new, axis=1, keepdims=True)
        X_new /= self.std
        return X_new

    def applyInstanceNormalizationTest(self, X):
        X_new = X - self.mean
        X_new /= self.std
        return X_new

    def printCost(self, costHistory, i, miniBatchSize):
        if (miniBatchSize != None):
            cost = statistics.mean(costHistory[-1 * miniBatchSize:])
        else:
            cost = costHistory[-1]
        print('Epoch:', i, '-', cost)

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
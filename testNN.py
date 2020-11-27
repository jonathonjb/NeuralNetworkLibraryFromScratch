from neural_network import NeuralNetwork
from layer import Dense
from metrics import Accuracy, Recall, Precision
from preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle
import math

def create3dData(numZeroValues, numOneValues, numTwoValues, plot=False, xSize=6, ySize=6, zSize=6):
    zeroValues = []
    oneValues = []
    twoValues = []

    xTwoBuffer = math.floor(xSize / 3)
    yTwoBuffer = math.floor(ySize / 3)
    zTwoBuffer = math.floor(zSize / 3)
    for i in range(numOneValues):
        x = random.random() * xTwoBuffer + xTwoBuffer
        y = random.random() * yTwoBuffer + yTwoBuffer
        z = random.random() * zTwoBuffer + zTwoBuffer
        oneValues.append([x, y, z, 1])

    for i in range(numTwoValues):
        x = random.random() * (.30 * xSize) + (.7 * xSize)
        y = random.random() * (.30 * ySize) + (.7 * ySize)
        z = random.random() * (.30 * zSize)
        twoValues.append([x, y, z, 2])

    for i in range(numZeroValues):
        x = random.random() * xSize
        y = random.random() * ySize
        z = random.random() * zSize

        while(((x >= xTwoBuffer and x < xTwoBuffer + xTwoBuffer) and (y >= yTwoBuffer and y < yTwoBuffer + yTwoBuffer) and
              (z >= zTwoBuffer and z < zTwoBuffer + zTwoBuffer))

              or ((x >= .7 * xSize) and y >= (.7 * ySize)) and z < (.3 * zSize)):
            x = random.random() * xSize
            y = random.random() * ySize
            z = random.random() * zSize

        zeroValues.append([x, y, z, 0])

    twoValues = np.array(twoValues)
    oneValues = np.array(oneValues)
    zeroValues = np.array(zeroValues)

    figure = plt.figure()
    ax = Axes3D(figure)

    if(plot):
        ax.scatter(twoValues[:, 0], twoValues[:, 1], twoValues[:, 2], marker='^', color='orange')
        ax.scatter(oneValues[:, 0], oneValues[:, 1], oneValues[:, 2], marker='s', color='green')
        ax.scatter(zeroValues[:, 0], zeroValues[:, 1], zeroValues[:, 2], marker='o', color='blue')
        plt.show()

    return zeroValues, oneValues, twoValues

# plots a bunch of random data using matplotlab to see the decision boundary
def createRandomData3d(num, xSize, ySize, zSize):
    randData = np.random.rand(3, num)
    randData[0] *= xSize
    randData[1] *= ySize
    randData[2] *= zSize
    return randData

def plotDecisionBoundary(neuralNetwork, xSize, ySize, zSize, zeroValues=None, oneValues=None, twoValues=None):
    num = 100000
    randData = createRandomData3d(num, xSize, ySize, zSize)
    predictions = neuralNetwork.predict(randData)

    testTwoValues = []
    testOneValues = []
    testZeroValues = []
    for i in range(num):
        if (predictions[i] == 2):
            testTwoValues.append(i)
        elif (predictions[i] == 1):
            testOneValues.append(i)
        else:
            testZeroValues.append(i)

    testTwoValues = randData[:, testTwoValues]
    testOneValues = randData[:, testOneValues]
    testZeroValues = randData[:, testZeroValues]

    figure = plt.figure()
    ax = Axes3D(figure)
    ax.scatter(testTwoValues[0], testTwoValues[1], testTwoValues[2], alpha=1, marker='^', color='moccasin')
    ax.scatter(testOneValues[0], testOneValues[1], testOneValues[2], alpha=1, marker='s', color='springgreen')
    ax.scatter(testZeroValues[0], testZeroValues[1], testZeroValues[2], alpha=0.01, marker='o', color='cornflowerblue')
    plt.show()

    if(zeroValues != None):
        ax.scatter(zeroValues[:, 0], zeroValues[:, 1], zeroValues[:, 2], marker='o', color='blue')
    if (oneValues != None):
        ax.scatter(oneValues[:, 0], oneValues[:, 1], oneValues[:, 2], marker='s', color='green')
    if (twoValues != None):
        ax.scatter(twoValues[:, 0], twoValues[:, 1], twoValues[:, 2], marker='^', color='orange')
    plt.show()

def saveModel(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def loadModel(filename):
    return pickle.load(open(filename, 'rb'))

def testNeuralNetwork():

    neuralNetwork = NeuralNetwork(inputSize=3, layers=[
        Dense(size=16, activation='relu'),
        Dense(size=16, activation='relu'),
        Dense(size=3, activation='softmax')
    ])

    xSize = 10000
    ySize = 10
    zSize = 1000000
    zeroValues, oneValues, twoValues = create3dData(numZeroValues=500, numOneValues=300, numTwoValues=300, plot=True,
                                                     xSize=xSize, ySize=ySize, zSize=zSize)
    data = np.concatenate((zeroValues, oneValues, twoValues), axis=0)

    X = np.delete(data, -1, axis=1).T
    y = data[:, -1].astype('int64')

    yEncoded = OneHotEncoder(y).yEncoded

    neuralNetwork.compile(
        optimization='adam',
        normalization='instance'
    )

    costHistory = neuralNetwork.train(X, yEncoded, epochs=300, learningRate=0.001, miniBatchSize=32,
                                      regularization='L2', lambdaReg=0.01, decayRate=None,
                                      printCostRounds=100)

    plt.plot(costHistory)
    plt.show()
    plotDecisionBoundary(neuralNetwork, xSize=xSize, ySize=ySize, zSize=zSize,
                         zeroValues=None, oneValues=None, twoValues=None)

    testData = np.concatenate(create3dData(numZeroValues=100000, numOneValues=100000, numTwoValues=100000,
                                           xSize=xSize, ySize=ySize, zSize=zSize), axis=0)
    X_test = np.delete(testData, -1, axis=1).T
    y_test= testData[:, -1]
    predictions = neuralNetwork.predict(X_test)

    print('Accuracy:', Accuracy(predictions, y_test).score)
    print('Precision:', Precision(predictions, y_test).score)
    print('Recall:', Recall(predictions, y_test).score)


def main():
    testNeuralNetwork()

if __name__ == '__main__':
    main()

from neural_network import NeuralNetwork
from layer import Dense
from metrics import Accuracy, Recall, Precision
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle
import math

def create3dData(numTrueValues, numFalseValues, plot=False, xSize=6, ySize=6, zSize=6):
    falseValues = []
    trueValues = []

    xBuffer = math.floor(xSize / 3)
    yBuffer = math.floor(ySize / 3)
    zBuffer = math.floor(zSize / 3)
    for i in range(numTrueValues):
        x = random.random() * xBuffer + xBuffer
        y = random.random() * yBuffer + yBuffer
        z = random.random() * zBuffer + zBuffer
        trueValues.append([x, y, z, 1])

    for i in range(numFalseValues):
        x = random.random() * xSize
        y = random.random() * ySize
        z = random.random() * zSize

        while((x >= xBuffer and x < xBuffer + xBuffer) and (y >= yBuffer and y < yBuffer + yBuffer) and
              (z >= zBuffer and z < zBuffer + zBuffer)):
            x = random.random() * xSize
            y = random.random() * ySize
            z = random.random() * zSize

        falseValues.append([x, y, z, 0])

    trueValues = np.array(trueValues)
    falseValues = np.array(falseValues)

    figure = plt.figure()
    ax = Axes3D(figure)

    if(plot):
        ax.scatter(trueValues[:, 0], trueValues[:, 1], trueValues[:, 2], marker='s', color='green')
        ax.scatter(falseValues[:, 0], falseValues[:, 1], falseValues[:, 2], marker='o', color='blue')
        plt.show()

    return falseValues, trueValues

# plots a bunch of random data using matplotlab to see the decision boundary
def createRandomData3d(num, xSize, ySize, zSize):
    randData = np.random.rand(3, num)
    randData[0] *= xSize
    randData[1] *= ySize
    randData[2] *= zSize
    return randData

def plotDecisionBoundary(neuralNetwork, xSize, ySize, zSize, falseValues=None, trueValues=None):
    num = 100000
    randData = createRandomData3d(num, xSize, ySize, zSize)
    predictions = neuralNetwork.predict(randData)

    testTrue = []
    testFalse = []
    for i in range(num):
        if (predictions[0, i] == 1):
            testTrue.append(i)
        else:
            testFalse.append(i)
    testTrue = randData[:, testTrue]
    testFalse = randData[:, testFalse]

    figure = plt.figure()
    ax = Axes3D(figure)

    ax.scatter(testTrue[0], testTrue[1], testTrue[2], marker='s', color='springgreen')
    ax.scatter(testFalse[0], testFalse[1], testFalse[2], alpha=0.01, marker='o', color='cornflowerblue')
    if(falseValues != None):
        ax.scatter(falseValues[:, 0], falseValues[:, 1], falseValues[:, 2], marker='o', color='blue')
    if (trueValues != None):
        ax.scatter(trueValues[:, 0], trueValues[:, 1], trueValues[:, 2], marker='s', color='green')
    plt.show()

def saveModel(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def loadModel(filename):
    return pickle.load(open(filename, 'rb'))

def testNeuralNetwork():

    neuralNetwork = NeuralNetwork(inputSize=3, layers=[
        Dense(size=16, activation='relu'),
        Dense(size=16, activation='relu'),
        Dense(size=1, activation='sigmoid')
    ])

    xSize = 100000
    ySize = 6
    zSize = 6
    falseValues, trueValues = create3dData(numFalseValues=500, numTrueValues=300, plot=True,
                                           xSize=xSize, ySize=ySize, zSize=zSize)
    data = np.concatenate((falseValues, trueValues), axis=0)

    X = np.delete(data, -1, axis=1).T
    y = data[:, -1]

    neuralNetwork.compile(
        optimization='adam',
        normalization='instance'
    )

    costHistory = neuralNetwork.train(X, y, epochs=1000, learningRate=0.001, miniBatchSize=32,
                                      regularization='L2', lambdaReg=0.1, decayRate=0.0000001,
                                      printCostRounds=100)

    plt.plot(costHistory)
    plt.show()
    plotDecisionBoundary(neuralNetwork, xSize=xSize, ySize=ySize, zSize=zSize, falseValues=None, trueValues=None)

    testData = np.concatenate(create3dData(numTrueValues=100000, numFalseValues=100000), axis=0)
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

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

def createData(numOfClasses, numExamplesPerClass, sizePerClass, xSize, ySize, zSize, normalizationScale=1, plot=False):
    data = []

    size = sizePerClass / 2
    for i in range(numOfClasses):
        xCentroid = random.random() * xSize
        yCentroid = random.random() * ySize
        zCentroid = random.random() * zSize

        for j in range(numExamplesPerClass):
            x = xCentroid + np.random.normal(scale=normalizationScale) * (xSize * size)
            y = yCentroid + np.random.normal(scale=normalizationScale) * (ySize * size)
            z = zCentroid + np.random.normal(scale=normalizationScale) * (zSize * size)
            data.append([x, y, z, i])

    data = np.array(data)

    if(plot):
        colors = ['blue', 'orange', 'green', 'red', 'yellow', 'pink', 'purple', 'lawngreen', 'peru',
                  'lightcoral', 'gray', 'forestgreen', 'limegreen', 'navy', 'plum', 'crimson']
        markers = ['.', ',', 'o', 'v', '^', '>', '<', 's', '1', '2', '3', '4', '*', '+', 'x', 'd']
        figure = plt.figure()
        ax = Axes3D(figure)
        ax.set_xlim3d(0, xSize)
        ax.set_ylim3d(0, ySize)
        ax.set_zlim3d(0, zSize)
        for i in range(numOfClasses):
            currClass = data[data[:, 3] == i]
            ax.scatter(currClass[:,0], currClass[:,1], currClass[:,2], marker=markers[i], color=colors[i])
        plt.show()

    return data

# plots a bunch of random data using matplotlab to see the decision boundary
def createRandomData3d(num, xSize, ySize, zSize):
    randData = np.random.rand(3, num)
    randData[0] *= xSize
    randData[1] *= ySize
    randData[2] *= zSize
    return randData

def plotDecisionBoundary(neuralNetwork, numOfClasses, xSize, ySize, zSize, zeroValues=None, oneValues=None, twoValues=None):
    num = 1500000
    randData = createRandomData3d(num, xSize, ySize, zSize)
    predictions = neuralNetwork.predict(randData)

    colors = ['blue', 'orange', 'green', 'red', 'yellow', 'pink', 'purple', 'lawngreen', 'peru',
              'lightcoral', 'gray', 'forestgreen', 'limegreen', 'navy', 'plum', 'crimson']
    markers = ['.', ',', 'o', 'v', '^', '>', '<', 's', '1', '2', '3', '4', '*', '+', 'x', 'd']

    figure = plt.figure()
    ax = Axes3D(figure)
    ax.set_xlim3d(0, xSize)
    ax.set_ylim3d(0, ySize)
    ax.set_zlim3d(0, zSize)
    for i in range(numOfClasses):
        indexes = (predictions == i)
        currClass = randData[:, indexes]
        ax.scatter(currClass[0, :], currClass[1, :], currClass[2, :], alpha=0.01, marker=markers[i], color=colors[i])
    plt.show()


def saveModel(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def loadModel(filename):
    return pickle.load(open(filename, 'rb'))

def testNeuralNetwork():
    xSize = 100000
    ySize = 100000
    zSize = 100000
    numOfClasses = 10
    testProportion = 0.95

    neuralNetwork = NeuralNetwork(inputSize=3, layers=[
        Dense(size=16, activation='relu'),
        Dense(size=16, activation='relu'),
        Dense(size=numOfClasses, activation='softmax')
    ])

    data = createData(numOfClasses=numOfClasses, numExamplesPerClass=1000, sizePerClass=0.2,
                      xSize=100000, ySize=100000, zSize=100000, normalizationScale=1 ,plot=True).T

    np.random.shuffle(data.T)
    splitIndex = math.floor(data.shape[1] * testProportion)
    testData = data[:, :splitIndex]
    data = data[:, splitIndex:]

    X = np.delete(data, -1, axis=0)
    y = data[-1].astype('int64')

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
    plotDecisionBoundary(neuralNetwork, numOfClasses=numOfClasses, xSize=xSize, ySize=ySize, zSize=zSize)

    X_test = np.delete(testData, -1, axis=0)
    y_test= testData[-1]
    predictions = neuralNetwork.predict(X_test)

    print('Accuracy:', Accuracy(predictions, y_test).score)
    print('Precision:', Precision(predictions, y_test).score)
    print('Recall:', Recall(predictions, y_test).score)


def main():
    testNeuralNetwork()

if __name__ == '__main__':
    main()

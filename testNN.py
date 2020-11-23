from neural_network import NeuralNetwork
from layer import Dense
from metrics import Accuracy, Recall, Precision
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle

def create3dData(numTrueValues, numFalseValues, plot=False):
    falseValues = []
    trueValues = []

    for i in range(numTrueValues):
        x = random.random() * 2 + 2
        y = random.random() * 2 + 2
        z = random.random() * 2 + 2
        trueValues.append([x, y, z, 1])

    for i in range(numFalseValues):
        x = random.random() * 6
        y = random.random() * 6
        z = random.random() * 6

        while((x >= 2 and x < 4) and (y >= 2 and y < 4) and (z >= 2 and z < 4)):
            x = random.random() * 6
            y = random.random() * 6
            z = random.random() * 6

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
def createRandomData3d(num):
    randData = np.random.rand(3, num) * 6
    return randData

def plotDecisionBoundary(falseValues, trueValues, neuralNetwork):
    num = 100000
    randData = createRandomData3d(num)
    predictions = neuralNetwork.predict(randData)

    test = []
    circles = []
    for i in range(num):
        if (predictions[0, i] == 1):
            test.append(i)
        else:
            circles.append(i)
    test = randData[:, test]
    circles = randData[:, circles]

    figure = plt.figure()
    ax = Axes3D(figure)

    ax.scatter(test[0], test[1], test[2], marker='s', color='springgreen')
    ax.scatter(circles[0], circles[1], circles[2], alpha=0.01, marker='o', color='cornflowerblue')
    ax.scatter(falseValues[:, 0], falseValues[:, 1], falseValues[:, 2], marker='o', color='blue')
    ax.scatter(trueValues[:, 0], trueValues[:, 1], trueValues[:, 2], marker='s', color='green')
    plt.show()

def saveModel(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def loadModel(filename):
    return pickle.load(open(filename, 'rb'))

def trainNeuralNetworkSimpleData():

    neuralNetwork = NeuralNetwork(inputSize=3, layers=[
        Dense(size=16, activation='relu'),
        Dense(size=16, activation='relu'),
        Dense(size=1, activation='sigmoid')
    ])

    falseValues, trueValues = create3dData(numFalseValues=500, numTrueValues=300, plot=True)
    data = np.concatenate((falseValues, trueValues), axis=0)

    X = np.delete(data, -1, axis=1).T
    y = data[:, -1]

    neuralNetwork.compile(
        optimization='adam'
    )

    costHistory = neuralNetwork.train(X, y, epochs=1000, learningRate=0.001, miniBatch=True, miniBatchSize=32,
                                      printCosts=True, printCostRounds=100)

    plt.plot(costHistory)
    plt.show()
    plotDecisionBoundary(falseValues, trueValues, neuralNetwork)

    testData = np.concatenate(create3dData(10000, 10000), axis=0)
    X_test = np.delete(data, -1, axis=1).T
    y_test= data[:, -1]
    predictions = neuralNetwork.predict(X_test)

    print('Accuracy:', Accuracy().evaluate(predictions, y_test))
    print('Precision:', Precision().evaluate(predictions, y_test))
    print('Recall:', Recall().evaluate(predictions, y_test))


def main():
    trainNeuralNetworkSimpleData()

if __name__ == '__main__':
    main()
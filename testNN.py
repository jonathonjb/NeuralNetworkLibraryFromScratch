from neural_network import NeuralNetwork
from layer import Dense
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle

def create3dData():
    circles = []
    squares = []

    for i in range(200):
        x = random.random() * 2 + 2
        y = random.random() * 2 + 2
        z = random.random() * 2 + 2
        squares.append([x, y, z, 1])

    for i in range(500):
        x = random.random() * 6
        y = random.random() * 6
        z = random.random() * 6

        while((x >= 2 and x < 4) and (y >= 2 and y < 4) and (z >= 2 and z < 4)):
            x = random.random() * 6
            y = random.random() * 6
            z = random.random() * 6

        circles.append([x, y, z, 0])

    squares = np.array(squares)
    circles = np.array(circles)


    figure = plt.figure()
    ax = Axes3D(figure)

    ax.scatter(squares[:, 0], squares[:, 1], squares[:, 2], marker='s', color='green')
    ax.scatter(circles[:, 0], circles[:, 1], circles[:, 2], marker='o', color='blue')
    plt.show()

    return circles, squares

# for the test set
def createRandomData3d(num):
    randData = np.random.rand(3, num) * 6
    return randData

def plotResults3d(num, origCircles, origSquares, predictions, randData):
    squares = []
    circles = []
    for i in range(num):
        if (predictions[0, i] == 1):
            squares.append(i)
        else:
            circles.append(i)
    squares = randData[:, squares]
    circles = randData[:, circles]

    figure = plt.figure()
    ax = Axes3D(figure)

    ax.scatter(squares[0], squares[1], squares[2], marker='s', color='springgreen')
    ax.scatter(circles[0], circles[1], circles[2], alpha=0.01, marker='o', color='cornflowerblue')
    ax.scatter(origCircles[:, 0], origCircles[:, 1], origCircles[:, 2], marker='o', color='blue')
    ax.scatter(origSquares[:, 0], origSquares[:, 1], origSquares[:, 2], marker='s', color='green')
    plt.show()

def saveModel(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def loadModel(filename):
    return pickle.load(open('filename'), 'rb')

def trainNeuralNetworkSimpleData(plotCost=False):
    # classifies if examples are squares or not

    modelFileName = 'model.p'

    userInput = input('Do you want to load the old model?\n'
                  'Type "y" to load the model, and any other key to create a new model from scratch.')
    if(userInput == 'y'):
        loadModel(modelFileName)

    origCircles, origSquares = create3dData()
    data = np.concatenate((origCircles, origSquares), axis=0)
    X = np.delete(data, -1, axis=1)
    y = data[:, -1]


    neuralNetwork = NeuralNetwork(inputSize=3, layers=[
        Dense(size=5, activation='leaky_relu'),
        Dense(size=5, activation='leaky_relu'),
        Dense(size=1, activation='sigmoid')
    ])

    # neuralNetwork.setup(optimization='adam')

    costHistory = neuralNetwork.train(X.T, y, epochs=20000, learningRate=0.01, miniBatch=True, miniBatchSize=32,
                                      printCosts=True, printCostRounds=1000)


    if(plotCost):
        plt.plot(costHistory)
        plt.show()

    num = 100000
    randData = createRandomData3d(num)
    predictions = neuralNetwork.predict(randData)

    plotResults3d(num, origCircles, origSquares, predictions, randData)

    neuralNetwork.prettyPrint()

    userInput = input('Do you want to store the model that was recently trained? '
                  'Type "y" for yes, or any other key to not save the model.')

    if(userInput == 'y'):
        saveModel(modelFileName, neuralNetwork)

def main():
    trainNeuralNetworkSimpleData(plotCost=True)

if __name__ == '__main__':
    main()
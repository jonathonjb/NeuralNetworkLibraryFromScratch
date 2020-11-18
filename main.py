from neural_network import NeuralNetwork
from layer import Dense
import numpy as np
import matplotlib.pyplot as plt

def createData():
    circles = np.array([[1, 5, 0], [1, 3.8, 0], [1.3, 4, 0], [2, 3.8, 0], [1.5, 1.8, 0], [0.9, 0.9, 0], [2.3, 1, 0],
                        [2.6, 1.9, 0], [3.3, 1.9, 0], [4.2, 1.2, 0], [5.1, 2.1, 0], [5.4, 0.7, 0]])

    squares = np.array([[3, 5, 1], [3.1, 4.2, 1], [3.14, 3.1, 1], [3.9, 3.5, 1], [4.1, 4.8, 1], [4.2, 3, 1],
                        [4.8, 3.1, 1], [4.8, 4.9, 1], [5, 4, 1], [5.2, 3.1, 1], [5.4, 5.1, 1], [6, 3.8, 1]])

    plt.scatter(circles[:, 0], circles[:, 1], marker='o', color='blue')
    plt.scatter(squares[:, 0], squares[:, 1], marker='s', color='green')
    plt.savefig('simpleData')
    plt.show()

    data = np.concatenate((circles, squares), axis=0)
    return data

# for the test set
def createRandomData(num):
    randData = np.random.rand(2, num) * 5.5
    return randData

def trainNeuralNetworkSimpleData(plotCost=False):
    # classifies if examples are squares or not

    data = createData()
    X = np.delete(data, 2, axis=1)
    y = data[:, -1]

    neuralNetwork = NeuralNetwork(inputSize=2, layers=[
        Dense(size=3, activation='tanh'),
        Dense(size=3, activation='tanh'),
        Dense(size=1, activation='sigmoid')
    ])

    costHistory = neuralNetwork.train(X.T, y, iterations=10000, learningRate=0.05, printCost=True, printCostRounds=1000)

    if(plotCost):
        plt.plot(costHistory)
        plt.savefig('costHistory')
        plt.show()

    num = 100000
    randData = createRandomData(num)
    predictions = neuralNetwork.predict(randData)

    squares = []
    circles = []
    for i in range(num):
        if(predictions[0, i] == 1):
            squares.append(i)
        else:
            circles.append(i)

    squares = randData[:, squares]
    circles = randData[:, circles]

    plt.scatter(squares[0], squares[1], marker='s', color='green')
    plt.scatter(circles[0], circles[1], marker='o', color='blue')
    plt.savefig('endResult')
    plt.show()

    neuralNetwork.prettyPrint()

def main():
    trainNeuralNetworkSimpleData(plotCost=True)

if __name__ == '__main__':
    main()
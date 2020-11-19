from neural_network import NeuralNetwork
from layer import Dense
import numpy as np
import matplotlib.pyplot as plt

def createData():
    # circles = np.array([[1, 5, 0], [1, 3.8, 0], [1.3, 4, 0], [2, 3.8, 0], [1.5, 1.8, 0], [0.9, 0.9, 0], [2.3, 1, 0],
    #                     [2.6, 1.9, 0], [3.3, 1.9, 0], [4.2, 1.2, 0], [5.1, 2.1, 0], [5.4, 0.7, 0]])
    #
    # squares = np.array([[3, 5, 1], [3.1, 4.2, 1], [3.14, 3.1, 1], [3.9, 3.5, 1], [4.1, 4.8, 1], [4.2, 3, 1],
    #                     [4.8, 3.1, 1], [4.8, 4.9, 1], [5, 4, 1], [5.2, 3.1, 1], [5.4, 5.1, 1], [6, 3.8, 1]])

    circles = np.array([[1.8, 2.2, 0], [2.6, 1.8, 0], [3.8, 1.9, 0], [4, 2.1, 0], [4.1, 3, 0], [4.1, 4.1, 0],
                        [3.9, 4.3, 0], [1.7, 4, 0], [2, 4.3, 0], [2.1, 2.1, 0], [1.8, 3, 0], [1.9, 3.2, 0],
                        [3, 5, 0], [2.5, 4.5, 0], [2.6, 4.6, 0], [3.5, 4.6, 0], [3, 1.9, 0], [3.3, 1.8, 0], [3.5, 1.75, 0]])
    squares = np.array([[2.2, 2.9, 1], [2.22, 3.9, 1], [3.3, 2.2, 1], [3.8, 2.65, 1], [3.8, 3.6, 1], [3.5, 4, 1],
                        [3, 4.2, 1], [2.9, 2.4, 1], [2.2, 3.5, 1], [2.6, 2.6, 1], [3.6, 2.5, 1], [3.9, 3.1, 1],
                        [2.6, 4, 1]])

    plt.scatter(circles[:, 0], circles[:, 1], marker='o', color='blue')
    plt.scatter(squares[:, 0], squares[:, 1], marker='s', color='green')
    plt.savefig('simpleData')
    plt.show()

    return circles, squares

# for the test set
def createRandomData(num):
    randData = np.random.rand(2, num) * 6
    return randData

def trainNeuralNetworkSimpleData(plotCost=False):
    # classifies if examples are squares or not

    origCircles, origSquares = createData()
    data = np.concatenate((origCircles, origSquares), axis=0)
    X = np.delete(data, -1, axis=1)
    y = data[:, -1]


    neuralNetwork = NeuralNetwork(inputSize=2, layers=[
        Dense(size=5, activation='tanh'),
        Dense(size=5, activation='tanh'),
        Dense(size=1, activation='sigmoid')
    ])

    costHistory = neuralNetwork.train(X.T, y, iterations=1000000, learningRate=0.005, printCost=True, printCostRounds=10000)

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

    plt.scatter(squares[0], squares[1], marker='s', color='springgreen')
    plt.scatter(circles[0], circles[1], marker='o', color='cornflowerblue')

    plt.scatter(origCircles[:, 0], origCircles[:, 1], marker='o', color='blue')
    plt.scatter(origSquares[:, 0], origSquares[:, 1], marker='s', color='green')
    plt.savefig('endResult')
    plt.show()

    neuralNetwork.prettyPrint()

def main():
    trainNeuralNetworkSimpleData(plotCost=True)

if __name__ == '__main__':
    main()
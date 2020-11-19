# Neural Network Library From Scratch

I created a functional neural network library using the NumPy library.

This library definitely isn't nearly as powerful or as efficient as some of the popular Python deep learning libaries out there such as Tensorflow or Pytorch, but it's a great project to allow yourself to gain a better understanding of how these neural networks work.

## Creating the neural network

Creating the neural network is simple. Just create a NeuralNetwork object, and set the inputSize to be equal to the number of features needed, and then set the layers to be a list of Layer objects you want in your network. Here is an example:

```
neuralNetwork = NeuralNetwork(inputSize=2, layers=[
  Dense(size=3, activation='tanh'),
  Dense(size=3, activation='tanh'),
  Dense(size=1, activation='sigmoid')
])
```

To increase the depth of the network, add more layers to the neuralNetwork object. To increase the width of each layers, simply change the size of the layers. You should also specify what activation function you want to be used on each layer - at the moment, you can only use the 'tanh' or the 'sigmoid' activation functions. 

## Training the neural network

To train the network, you need to call the 'train' function, and pass to it the X (features), and the y (end result). You may set the number of iterations used for the gradient descent by changing the 'iterations' value, and you may change the learning rate by modifying the learningRate value.

Example:

```
costHistory = neuralNetwork.train(X.T, y, iterations=10000, learningRate=0.05, printCost=True, printCostRounds=1000)
```

It will return the cost history, which you may plot using the matplotlib library.

## Making predictions

Making predictions is easy. All you need to do is call the predict function and then pass to it the X value or the features of the examples you would like it to classify like below.

```
predictions = neuralNetwork.predict(randData)
```

# Example 1:

I hand-created a very simple dataset of 24 examples below.
![Data 1](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/simpleData.png)

The neural network will be used to classify whether the examples passed to it are squares or circles.

The data was used to train the neural network. The 'train' function returned the cost history, which is shown below.

![Cost history 1](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/costHistory.png)

This is the end result. If the example lies in the blue area, then it will be classified as a circle / non-square. If it lies in the green area, then it will be clssified as a square.

![End result 1](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/endResult.png)

# Example 2:

![Data 2](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/simpleData2.png)

![Cost history 2](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/costHistory2.png)

![End result 2](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/endResult2.png)

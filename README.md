# Neural Network Library From Scratch

I created a functional neural network library using the NumPy library.

This library definitely isn't nearly as powerful or as efficient as some of the popular Python deep learning libaries out there such as Tensorflow or Pytorch, but it's a great project to allow yourself to gain a better understanding of how these neural networks work.

## Creating the neural network

Creating the neural network is simple. Just create a NeuralNetwork object, and set the inputSize to be equal to the number of features needed, and then set the layers to be a list of Layer objects you want in your network. Here is an example:

```
neuralNetwork = NeuralNetwork(inputSize=3, layers=[
      Dense(size=16, activation='relu'),
      Dense(size=16, activation='relu'),
      Dense(size=3, activation='softmax')
  ])
```

To increase the depth of the network, add more layers to the neuralNetwork object. To increase the width of each layers, simply change the size of the layers. You should also specify what activation function you want to be used on each layer - The activation functions you can use are: 'sigmoid', 'tanh', 'relu', and 'leaky_relu'. You may also use the 'softmax' activation function on the output layer.

## Training the neural network

Before training the network, you have the option to optimize to your training algorithm by calling the compile function, like below. Currently, three optimizations are supported: 'momentum', 'RMSprop', or 'adam'. It is recommended to use the 'adam' optimization here.

You also have the option to add normalization to your neural network. Currently, instance normalization is the only type supported.

```
neuralNetwork.compile(
    optimization='adam',
    normalization='instance'
)
```

To train the network, you need to call the 'train' function, and pass to it the X (features), and the y (end result). Here are the list of parameters and it's possible values:

**X**: {NumPy Array}

**y**: {NumPy Array}

**epochs**: {Integer}                   default: 10000    *- Number of times we go through the entire training set when training.*

**learningRate**: {Float}               default: 0.1      *- The rate in which the neural network learns during gradient descent.*

**miniBatchSize**: {None, Integer}      default: 32       *- The batch size used in gradient descent. If you want to use batch gradient descent instead, set this to None.*

**regularization**: {None, 'L2'}        default: 'L2'     *- The regularization algorithm used during training.*

**lambdaReg**: {Float}                  default: 0.1      *- Lambda used for regularization*

**decayRate**: {None, Float}            default: None     *- The rate in which the learning rate decays during training. If you don't want the learning rate to decay, set this to None.*

**printCostRounds**: {None, Integer}    default: 1000     *- Prints out the mean cost on the most recent epoch during training every N rounds. If you don't want to print out the cost every so round, set this to None.*

Example:

```
costHistory = neuralNetwork.train(X, y, epochs=1000, learningRate=0.001, miniBatchSize=1,
                                      regularization='L2', lambdaReg=0.03, decayRate=0.0000001,
                                      printCostRounds=100)
```

It will return the cost history, which you may plot using the matplotlib library.

## Making predictions

Making predictions is easy. All you need to do is call the predict function and then pass to it the X value or the features of the examples you would like it to classify like below.

```
predictions = neuralNetwork.predict(randData)
```

## Evaluating your artificial neural network model

You may evaluate your model's accuracy, precision or recall by using the classes in the *metrics* file like below.

```
from metrics import Accuracy, Recall, Precision

accuracy = Accuracy(predictions, y_test).score
precision = Precision(predictions, y_test).score
recall = Recall(predictions, y_test).score
```

# Simple example:

I hand-created a very simple dataset shown below.
![Data 1](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/simpleData.png)

The neural network will be used to classify whether the examples passed to it are 'true or 'false'.

The data was used to train the neural network. The 'train' function returned the cost history, which is shown below.

![Cost history 1](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/costHistory.png)

This is the end result. If the example lies in the blue area, then it will be classified as 'false'. If it lies in the green area, then it will be classified as 'true'.
![End result 1](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/endResult.png)


# Example using 3D data:

Below is the plot of the dataset I'll be using for this example. X will range between 0-10000, y will range from 0-10, and z will range from 0-1000000. If a data example lies in middle 33% for each axis, it will be labeled as true.

![Data 3d](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/simpleData3d.png)

Now, I train the data using the neural network, which has three layers. The first 2 layers contains 16 neurons, and uses the ReLU activation function. The third layer has only one output neuron, which uses the sigmoid activation function. 

During training, the model will be using mini-batch gradient descent with 32-examples batches, 'adam' optimization, 'L2' regularization, and instance normalization. The cost history won't be smooth due to the fact that we're using the mini-batch gradient descent, however, using the mini-batch gradient descent combined with 'adam' optimization, and instance normalization will cause the gradient descent algorithm to converge much faster.  

![Cost history 3d](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/costHistory3d.png)

And this is what the decision boundary looks like.

![End result 3d](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/endResult3d.png)

# Example with multiple classes

Below is the plot of a dataset which includes 10 different classes. The architecture of the neural network is exactly the same as the one above, except that we will be using the softmax activation function on the final layer, which includes 10 neurons.

![Data 3d](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/10classesData.png)

![Cost history 3d](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/10classesCostHistory.png)

![End result 3d](https://github.com/jonathonjb/NeuralNetworkLibraryFromScratch/blob/main/images/10classesEndResult.png)
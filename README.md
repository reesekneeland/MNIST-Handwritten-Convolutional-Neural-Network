# MNIST-Handwritten-Convolutional-Neural-Network
Creating a fully functional convolutional neural network to solve the MNIST handwritten digit recognition problem

### Single-Layer Linear Perceptron

The Single-Layer Linear Perceptron model is trained by using the **fc()** and **fc_backward()** functions to complete a forward and backward pass across a single layer to update the weights in between each iteration. It does this with no activation function to compute losses, but by instead calling the **loss_euclidean()** function, using Euclidean distance measure to compute error. This model, using gamma=0.0001, lamda=0.8, and 5000 iterations, yielded an accuracy of 0.383

### Single Layer Perceptron

The Single-Layer Perceptron model is trained in exactly the same way as the previous function, except that it uses the **loss_cross_entropy_softmax()** loss function, which introduces a softmax activation function, giving our model some nonlinearity to work with. This model, using gamma=0.005, lamda=0.9, and 5000 iterations, yielded an accuracy of 0.878


### Multi-Layer Perceptron

The Multi-Layer Perceptron model is trained in a similar way as the previous functions, now adding multiple layers to the computation, we first compute using **fc()**, then through a **relu()** function, then **fc()** again, computing the loss, and running backwards through those layers again. This model, using gamma=0.012, lamda=0.8, and 5000 iterations, yielded an accuracy of 0.924

### Convolutional Neural Network

The Convolutional Neural Network model is trained in a similar way as the previous functions, now adding a convolutional layer to the computation, we first compute using a new **conv()** function, which performs a convolutional operation on our input images, then through **relu()** like before, then introducing a max pooling function **pool2x2()**, and a flattening function **flattening()** before finally passing through **fc()** again, computing the loss, and running backwards through those layers again. This model, using gamma=0.005, lamda=0.95, and 9000 iterations, yielded an accuracy of 0.936.

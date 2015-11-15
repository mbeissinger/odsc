"""
This tutorial covers your simplest neural network: a multilayer perceptron (MLP)
Also known as feedforward neural network.
We will learn to classify MNIST handwritten digit images into their correct label (0-9).
"""
import cPickle as pickle
import gzip
from PIL import Image
from opendeep.utils.image import tile_raster_images
import theano
import theano.tensor as T
import numpy
import numpy.random as rng

if __name__ == '__main__':
    # Load our data
    # Download and unzip pickled version from here: http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    ################
    # Explore data #
    ################
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(gzip.open('datasets/mnist.pkl.gz', 'rb'))
    print "Shapes:"
    print train_x.shape, train_y.shape
    print valid_x.shape, valid_y.shape
    print test_x.shape, test_y.shape

    print "--------------"
    print "Example input:"
    print train_x[0]
    print "Example label:"
    print train_y[0]

    # Show example images - using tile_raster_images helper function from OpenDeep to get 28x28 image from 784 array.
    input_images = train_x[:25]
    im = Image.fromarray(
        tile_raster_images(input_images,
                           img_shape=(28, 28),
                           tile_shape=(1, 25),
                           tile_spacing=(1, 1))
    )
    im.save("example_mnist_numbers.png")

    #########
    # Model #
    #########
    # Cool, now we know a little about the input data, let's design the MLP to work with it!
    # An MLP looks like this: input -> hiddens -> output classification
    # Each stage is just a matrix multiplication with a nonlinear function applied after.
    # Inputs are matrices where rows are examples and columns are pixels - so create a symbolic Theano matrix.
    x = T.matrix('xs')
    # Now let's start building the equation for our MLP!

    # The first transformation is the input x -> hidden layer h.
    # We defined this transformation with h = tanh(x.dot(W_x) + b_h)
    # where the learnable model parameters are W_x and b_h.

    # Therefore, we will need a weights matrix W_x and a bias vector b_h.
    # W_x has shape (input_size, hidden_size) and b_h has shape (hidden_size,).
    # Initialization is important in deep learning; we want something random so the model doesn't get stuck early.
    # Many papers in this subject, but for now we will just use a normal distribution with mean=0 and std=0.05.
    # Another good option for tanh layers is to use a uniform distribution with interval +- sqrt(6/sum(shape)).
    # These are hyperparameters to play with.
    # Bias starting as zero is fine.

    W_x = numpy.asarray(rng.normal(loc=0.0, scale=.05, size=(28 * 28, 500)), dtype=theano.config.floatX)
    b_h = numpy.zeros(shape=(500,), dtype=theano.config.floatX)

    # To update a variable used in an equation (for example, while learning),
    # Theano needs it to be in a special wrapper called a shared variable.
    # These are the model parameters for our first hidden layer!
    W_x = theano.shared(W_x, name="W_x")
    b_h = theano.shared(b_h, name="b_h")

    # Now, we can finally write the equation to give our symbolic hidden layer h!
    h = T.tanh(
        T.dot(x, W_x) + b_h
    )
    # Side note - if we used softmax instead of tanh for the activation, this would be performing logistic regression!

    # We have the hidden layer h, let's put that softmax layer on top for classification output y!
    # Same deal as before, the transformation is defined as:
    # y = softmax(h.dot(W_h) + b_y)
    # where the learnable parameters are W_h and b_y.
    # W_h has shape (hidden_size, output_size) and b_y has shape (output_size,).

    # We will use the same random initialization strategy as before.
    W_h = numpy.asarray(rng.normal(loc=0.0, scale=.05, size=(500, 10)), dtype=theano.config.floatX)
    b_y = numpy.zeros(shape=(10,), dtype=theano.config.floatX)
    # Don't forget to make them shared variables!
    W_h = theano.shared(W_h, name="W_h")
    b_y = theano.shared(b_y, name="b_y")

    # Now write the equation for the output!
    y = T.nnet.softmax(
        T.dot(h, W_h) + b_y
    )

    # The output (due to softmax) is a vector of class probabilities.
    # To get the output class 'guess' from the model, just take the index of the highest probability!
    y_hat = T.argmax(y, axis=1)

    # That's everything! Just four model parameters and one input variable.

    #################
    #  Optimization #
    #################
    # The variable y_hat represents the output of running our model, but we need a cost function to use for training.
    # For a softmax (probability) output, we want to maximize the likelihood of P(Y=y|X).
    # This means we want to minimize the negative log-likelihood cost! (For a primer, see machine learning Coursera.)

    # Cost functions always need the truth outputs to compare against (this is supervised learning).
    # From before, we saw the labels were a vector of ints - so let's make a symbolic variable for this!
    correct_labels = T.lvector("labels")  # integer vector

    # Now we can compare our output probability from y with the true labels.
    # Because the labels are integers, we will want to make an indexing mask to pick out the probabilities
    # our model thought was the likelihood of the correct label.
    log_likelihood = T.log(y)[T.arange(correct_labels.shape[0]), correct_labels]
    # We use mean instead of sum to be less dependent on batch size (better for flexibility)
    cost = -T.mean(log_likelihood)

    # Easiest way to train neural nets is with Stochastic Gradient Descent
    # This takes each example, calculates the gradient, and changes the model parameters a small amount
    # in the direction of the gradient.

    # Fancier add-ons to stochastic gradient descent will reduce the learning rate over time, add a momentum
    # factor to the parameters, etc.

    # Before we can start training, we need to know what the gradients are.
    # Luckily we don't have to do any math! Theano has symbolic auto-differentiation which means it can
    # calculate the gradients for arbitrary equations with respect to a cost and parameters.
    parameters = [W_x, b_h, W_h, b_y]
    gradients = T.grad(cost, parameters)
    # Now gradients contains the list of derivatives: [d_cost/d_W_x, d_cost/d_b_h, d_cost/d_W_h, d_cost/d_b_y]

    # One last thing we need to do before training is to use these gradients to update the parameters!
    # Remember how parameters are shared variables? Well, Theano uses something called updates
    # which are just pairs of (shared_variable, new_variable_expression) to change its value.
    # So, let's create these updates to show how we change the parameter values during training with gradients!
    # We use a learning rate to make small steps over time.
    learning_rate = 0.01
    train_updates = [(param, param - learning_rate * gradient) for param, gradient in zip(parameters, gradients)]

    # Now we can create a Theano function that takes in real inputs and trains our model.
    f_train = theano.function(inputs=[x, correct_labels], outputs=cost, updates=train_updates,
                              allow_input_downcast=True)

    # For testing purposes, we don't want to use updates to change the parameters - so create a separate function!
    # We also care more about the output guesses, so let's return those instead of the cost.
    # error = sum(T.neq(y_hat, correct_labels))/float(y_hat.shape[0])
    f_test = theano.function(inputs=[x], outputs=y_hat)

    # Our training can begin!
    # The two hyperparameters we have for this part are minibatch size (how many examples to process in parallel)
    # and the total number of passes over all examples (epochs).
    batch_size = 100
    epochs = 30

    # Given our batch size, compute how many batches we can fit into each data set
    train_batches = len(train_x) / batch_size
    valid_batches = len(valid_x) / batch_size
    test_batches = len(test_x) / batch_size

    # Our main training loop!
    for epoch in range(epochs):
        print epoch + 1, ":",

        train_costs = []
        train_accuracy = []
        for i in range(train_batches):
            # Grab our minibatch of examples from the whole train set.
            batch_x = train_x[i * batch_size:(i + 1) * batch_size]
            batch_labels = train_y[i * batch_size:(i + 1) * batch_size]
            # Compute the costs from the train function (which also updates the parameters)
            costs = f_train(batch_x, batch_labels)
            # Compute the predictions from the test function (which does not update parameters)
            preds = f_test(batch_x)
            # Compute the accuracy of our predictions against the correct batch labels
            acc = sum(preds == batch_labels) / float(len(batch_labels))

            train_costs.append(costs)
            train_accuracy.append(acc)
        # Show the mean cost and accuracy across minibatches (the entire train set!)
        print "cost:", numpy.mean(train_costs), "\ttrain:", str(numpy.mean(train_accuracy) * 100) + "%",

        valid_accuracy = []
        for i in range(valid_batches):
            batch_x = valid_x[i * batch_size:(i + 1) * batch_size]
            batch_labels = valid_y[i * batch_size:(i + 1) * batch_size]

            preds = f_test(batch_x)
            acc = sum(preds == batch_labels) / float(len(batch_labels))

            valid_accuracy.append(acc)
        print "\tvalid:", str(numpy.mean(valid_accuracy) * 100) + "%",

        test_accuracy = []
        for i in range(test_batches):
            batch_x = test_x[i * batch_size:(i + 1) * batch_size]
            batch_labels = test_y[i * batch_size:(i + 1) * batch_size]

            preds = f_test(batch_x)
            acc = sum(preds == batch_labels) / float(len(batch_labels))

            test_accuracy.append(acc)
        print "\ttest:", str(numpy.mean(test_accuracy) * 100) + "%"

"""
This tutorial covers your simplest neural network: a multilayer perceptron (MLP)
Also known as feedforward neural network.
We will learn to classify MNIST handwritten digit images into their correct label (0-9).
"""
import theano.tensor as T
from opendeep import config_root_logger
from opendeep.models import Prototype, Dense, Softmax
from opendeep.models.utils import Noise
from opendeep.monitor import Monitor
from opendeep.optimization.loss import Neg_LL
from opendeep.data import MNIST
from opendeep.optimization import AdaDelta

if __name__ == '__main__':
    # some debugging output to see what is going on under the hood
    config_root_logger()

    #########
    # Model #
    #########
    # build a Prototype container to easily add layers and make a cohesive model!
    mlp = Prototype()
    # need to define a variable for the inputs to this model, as well as the shape
    # we are doing minibatch training (where we don't know the minibatch size), and the image is a (784,) array.
    x = T.matrix('xs')
    x_shape = (None, 28*28)
    # add our first dense (fully-connected) layer!
    mlp.add(Dense(inputs=(x_shape, x), outputs=500, activation='tanh'))
    # noise is used to regularize the layer from overfitting to data (helps generalization)
    # when adding subsequent layers, we can simply provide the class type and any other kwargs
    # (omitting the `inputs` kwarg) and it will route the previous layer's outputs as the current
    # layer's inputs.
    mlp.add(Noise, noise='dropout', noise_level=0.5)
    # add our classification layer
    mlp.add(Softmax, outputs=10, out_as_probs=False)

    ################
    # Optimization #
    ################
    # Now that our model is complete, let's define the loss function to optimize
    # first need a target variable
    labels = T.lvector('ys')
    # negative log-likelihood for classification cost
    loss = Neg_LL(inputs=mlp.models[-1].p_y_given_x, targets=labels, one_hot=False)
    # make a monitor to view average accuracy per batch
    accuracy = Monitor(name='Accuracy',
                       expression=1-(T.mean(T.neq(mlp.models[-1].y_pred, labels))),
                       valid=True, test=True)

    # Now grab our MNIST dataset.
    mnist = MNIST()

    # finally define our optimizer and train the model!
    optimizer = AdaDelta(
        model=mlp,
        dataset=mnist,
        loss=loss,
        epochs=10,
        batch_size=64
    )
    # train!
    optimizer.train(monitor_channels=accuracy)

import keras.backend as K
import numpy as np


def image_filter_maximizer(model, layer_name, filter_id, img_size=(224, 224, 3), lrate=25., epochs=10, reg=None,
                           disc_dead=False):
    """
    Creates the rgb image that maximizes a given filter output, applying the method described in:
    "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"
    Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, 2014
    The image shows what type of information the filter is trying to capture.

    Parameters
    ----------
    model : keras.engine.training.Model
        Keras model
    layer_name : string
        Name of layer to be visualized
    filter_id : int
        Id of filter/neuron to be visualized
    img_size : (int, int, int)
        size of image to be created (height, width, channels) (default is (224, 244, 3))
    lrate : float
        learning rate for SGD (default is 25)
    epochs : int
        number of epochs for SGD (default is 10)
    reg : int
        Coefficient for L2 regularization. If None, no regularization is applied (default is None)
    disc_dead : boolean
        if True returns None when the activation is dead. i.e zero valued (default is False)

    Returns
    -------
    numpy.array
        Image that maximizes the given filter.
    """

    (H, W, C) = img_size
    layer_dims = len(model.get_layer(layer_name).output_shape)

    # if it is a 2d layer with dimension (BATCH, H, W, F)
    if layer_dims == 4:
        # get 2-d output of conv filter, e.g. shape (224, 224)
        filtr = model.get_layer(layer_name).output[:, :, :, filter_id]
        # loss to maximize is the mean value of the filter's output
        loss = K.mean(filtr)
    # if it is a 1d layer with dimensions (BATCH, D)
    elif layer_dims == 2:
        filtr = model.get_layer(layer_name).output[:, filter_id]
        loss = filtr
    # otherwise raise exception
    else:
        raise ValueError("Cannot handle layer with {} dimensions".format(layer_dims))

    # gradient of the loss wrt input image
    gradient = K.gradients(loss, model.input)[0]
    # derivative of L2 regularization wrt input
    if reg is not None:
        gradient -= 2 * reg * model.input
    gradient /= (K.sqrt(K.mean(K.square(gradient))) + 1e-5)

    compute = K.function([model.input], [loss, gradient])
    image = np.random.random((1, H, W, C)) * 20 + 128.

    # Gradient descent
    for epoch in range(epochs):
        loss_grad = compute([image])
        loss = loss_grad[0]
        if disc_dead and loss == 0:
            return None
        grad = loss_grad[1][0]
        image += lrate * grad
    return image[0]

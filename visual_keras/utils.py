import math
import numpy as np
import keras.backend as K
from keras.preprocessing import image
from keras.layers import Dense
from keras import Model


def preproc_image(image_path, img_size=(224, 244, 3), f_preproc=None):
    """
    Loads an image from a given path and preprocesses it for a Keras model, by adding the batch dimension, resizing
    and applying a preprocessing function needed by the model if needed.

    Parameters
    ----------
    image_path : String
        path of the image
    img_size : tuple
        Target image size to feed to the Keras model (default is (224, 224, 3))
    f_preproc : function
        Function for preprocessing the image. None is no preprocessing is needed (default None)

    Returns
    -------
    numpy.array
        Image as a numpy array.
    """

    img = image.load_img(image_path, target_size=img_size[:2])
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    if f_preproc is not None:
        img_tensor = f_preproc(img_tensor)
    return img_tensor


def get_layers_output(x, model, layer_names):
    """
    Runs an image through the network and returns the output of specififed layers.

    Parameters
    ----------
    x : numpy.array
        Input preprocessed image
    model : keras.engine.training.Model
        Keras model
    layer_names : list
        List of layer names for which to return the output

    Returns
    -------
    list
        A list where elements are the outputs of the target layers.
    """

    layer_outputs = list(map(lambda l: model.get_layer(l).output, layer_names))
    get_outputs = K.function([model.input], layer_outputs)
    outputs = get_outputs([x])
    return outputs


def remove_last_layer_activation(model):
    """
    Creates a new Keras model, where the activation of the last layer has been removed. For classification, typically
    this activation is a sigmoid of a softmax function.
    Only applies where the last layer is of class keras.layers.Dense.

    Parameters
    ----------
    model : keras.engine.training.Model
        Keras model

    Returns
    -------
    keras.Model
        A new Keras model where the activation of the last layer has been removed.

    Raises
    ------
    ValueError
        If the last layer is not of class keras.layers.Dense
    """

    # creates a new layer with the same config as the current last layer but with no activation
    last_layer = model.layers[-1]
    if not isinstance(last_layer, Dense):
        raise ValueError("Layer {} is not dense. Activation removal only possible with dense layer".format(last_layer))
    config = last_layer.get_config()
    config["activation"] = None  # removes activation
    config["name"] = config["name"] + "_noact"
    new_layer = Dense.from_config(config)

    # creates new model replacing the last layer with the new one
    #   model.layers.pop()
    x = model.layers[-2].output
    x = new_layer(x)
    new_model = Model(inputs=model.inputs, outputs=x)

    # setting weights of new layer to those of the original last layer
    new_model.layers[-1].set_weights(last_layer.get_weights())

    return new_model


def floats_to_pixels_standardized(x, bound=255):
    """
    Given a numpy array, returns the array with int8 values standardized
    with stddev 0.1 and bounded to the bound max pixel value.
    
    Parameters
    ----------
    x : numpy.array
        A numpy array with float values
    bound : int
        The max positive value to which values are bounded (default is 255)
    
    Returns
    -------
    numpy.array
        A numpy array of int8 ready to be shown in matplotlob
    """

    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= bound
    x = np.clip(x, 0, bound).astype('uint8')
    return x


def floats_to_pixels_normalized(x, bound=255):
    """
    Given a numpy array, returns the array with int8 values normalized
    between min and max and bounded to the bound max pixel value.
    
    Parameters
    ----------
    x : numpy.array
        A numpy array with float values
    bound : int, optional
        The max positive value to which values are bounded (default is 255)
    
    Returns
    -------
    numpy.array
        a numpy array of int8 ready to be shown in matplotlob
    """
    min_x = np.min(x)
    max_x = np.max(x)
    return (bound * (x - min_x) / (max_x - min_x)).astype('uint8')


def visual_grid_4d_rgb(X, padding=4, pixel_transform="normalized", dim=None):
    """
    Transforms a 4-dim numpy array with shapes (height, width, channels, filter),
    where there are 3 rgb channels, into a 3-dim grid of images (aka filters) with shapes
    (height, width, channels) that can be visualized in matplotlib. Values of the input
    array are bounded to [0,255].
    
    Parameters
    -----------
    x: numpy.array
        A 4-dim numpy array where the third dimension has size 3
    padding : int
        Number of pixels to pad between two images (default is 4)
    pixel_transform : String
        Type of transformation to apply to array values to bound them to [0,255]. Possible values are "normalized"
        and "standardized" (default is "normalized")
    dim : tuple
        tuple (rows, cols) defining how many rows and columns of images will be displayed in the grid. If None,
        dimensions will be inferred (default is None)

    Returns
    -------
    numpy.array
        A 3-d grid of rgb images where values are bounded to [0,255]. 
    """

    (H, W, C, F) = X.shape
    if C != 3:
        raise ValueError("Input array shape {}, has {} channel, required is 3".format(x.shape, x.shape[2]))
    
    if dim is not None:
        (rows, cols) = dim
        if F != rows * cols:
            raise ValueError("number of filters does not match number of rows and columns")
    else:
        rows = cols = int(math.ceil(math.sqrt(F)))
                       
    grid_W = cols * W + (cols - 1) * padding
    grid_H = rows * H + (rows - 1) * padding
    grid = np.zeros((grid_H, grid_W, C), dtype="uint8")
    y = 0
    for row in range(rows):
        x = 0
        for col in range(cols):
            curr_F = col + (row * cols)
            if curr_F >= F:
                break
            if pixel_transform == "normalized":
                img = floats_to_pixels_normalized(X[:, :, :, curr_F])
            else:
                img = floats_to_pixels_standardized(X[:, :, :, curr_F])
            grid[y:y+H, x:x+W] = img
            x = x + W + padding    
        y = y + H + padding
    return grid
    
    
def visual_grid_4d_multichannel(X, padding=4, pixel_transform="normalized"):
    """
    Transforms a 4-dim numpy array with shapes (height, width, channels, filter),
    where there are any number of channels, into a 3-dim grid of grayscale images (aka filters)
    with shapes (height, width, 1) that can be visualized in matplotlib. Values of the input
    array are bounded to [0,255].
    Each row of the output grid contains the channels-number of images of a filtr.

    Parameters
    -----------
    x: numpy.array
        A 4-dim numpy array where the third dimension has size 3
    padding : int
        Number of pixels to pad between two images (default is 4)
    pixel_transform : String
        Type of transformation to apply to array values to bound them to [0,255]. Possible values are "normalized"
        and "standardized" (default is "normalized")

    Returns
    -------
    numpy.array
        A 3-d grid of rgb images where values are bounded to [0,255].
    """

    (H, W, C, F) = X.shape
    grid_W = C * W + (C - 1) * padding
    grid_H = F * H + (F - 1) * padding
    grid = np.zeros((grid_H, grid_W), dtype="uint8")
    y = 0
    for filtr in range(F):
        x = 0
        for channel in range(C):
            if pixel_transform == "normalized":
                img = floats_to_pixels_normalized(X[:, :, channel, filtr])
            else:
                img = floats_to_pixels_standardized(X[:, :, channel, filtr])
            grid[y:y+H, x:x+W] = img
            x = x + W + padding    
        y = y + H + padding
    return grid   


def visual_grid_3d_multichannel(X, padding=4, pixel_transform="normalized", dim=None):
    """
    Transforms a 3-dim numpy array with shapes (height, width, channels), into a 3-dim grid of grayscale images
    (aka filters) with shapes (height, width, 1) that can be visualized in matplotlib.
    Every element (height, width) is transformed into a separate image (height, width, 1).
    Values of the input array are bounded to [0,255].

    Parameters
    -----------
    x: numpy.array
        A 3-dim numpy array
    padding : int
        Number of pixels to pad between two images (default is 4)
    pixel_transform : String
        Type of transformation to apply to array values to bound them to [0,255]. Possible values are "normalized"
        and "standardized" (default is "normalized")
    dim : tuple
        tuple (rows, cols) defining how many rows and columns of images will be displayed in the grid. If None,
        dimensions will be inferred (default is None)

    Returns
    -------
    numpy.array
        A 3-d grid of rgb images where values are bounded to [0,255].
    """

    (H, W, F) = X.shape

    if dim is not None:
        (rows, cols) = dim
        if F != rows * cols:
            raise ValueError("number of filters does not match number of rows and columns")
    else:
        rows = cols = int(math.ceil(math.sqrt(F)))
                       
    grid_W = cols * W + (cols - 1) * padding
    grid_H = rows * H + (rows - 1) * padding
    grid = np.zeros((grid_H, grid_W), dtype="uint8")
    y = 0
    for row in range(rows):
        x = 0
        for col in range(cols):
            curr_F = col + (row * cols)
            if curr_F >= F:
                break
            if pixel_transform == "normalized":
                img = floats_to_pixels_normalized(X[:, :, curr_F])
            else:
                img = floats_to_pixels_standardized(X[:, :, curr_F])
            grid[y:y+H, x:x+W] = img
            x = x + W + padding    
        y = y + H + padding
    return grid


def visual_grid_4d(x, padding=4, pixel_transform="normalized"):
    """
    Transforms a 4-dim numpy array with shapes (height, width, channels, filter), into a 3-dim grid of images
    (aka filters) with shapes (height, width, channels) that can be visualized in matplotlib. Values of the input
    array are bounded to [0,255].

    Parameters
    -----------
    x: numpy.array
        A 4-dim numpy array
    padding : int
        Number of pixels to pad between two images (default is 4)
    pixel_transform : String
        Type of transformation to apply to array values to bound them to [0,255]. Possible values are "normalized"
        and "standardized" (default is "normalized")

    Returns
    -------
    numpy.array
        A 3-d grid of rgb images where values are bounded to [0,255].
    """
    channels = x.shape[2]
    if channels == 3:
        return visual_grid_4d_rgb(x, padding, pixel_transform)
    else:
        return visual_grid_4d_multichannel(x, padding, pixel_transform)

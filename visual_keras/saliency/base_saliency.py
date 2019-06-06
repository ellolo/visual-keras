import numpy as np
from keras import backend as K
from keras.layers import Conv2D
from keras.layers.pooling import _Pooling2D
from .saliency_map import AbstractSaliencyMap
from ..utils import floats_to_pixels_standardized


class BaseSaliencyMap(AbstractSaliencyMap):
    """
    Class that computes vanilla saliency map for a given image tensor, as described in:
    "Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps"
    Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, 2014

    Attributes
    ----------
    model : keras.engine.training.Model
        Keras model
    layer_name : string
        Name of layer to compute saliency map for. Typically this is the input (default is None)
    multiply : boolean
        if True if will multiply the map by the input image (default is False)
    """
    
    def __init__(self, model, layer_name=None, multiply=False):
        """
        Parameters
        ----------
        model : keras.engine.training.Model
            Keras model
        layer_name : string
            Name of layer to compute saliency map for. If left to None it uses the input layer.
            Typically this is the input (default is None)
        multiply : boolean
        if True if will multiply the map by the input image (default is False)

        Raises
        ------
        ValueError
            If layer_name does not refer to either a conv or pool layer
        """

        if layer_name is not None:
            layer = model.get_layer(layer_name)
            if not isinstance(layer, Conv2D) and not isinstance(layer, _Pooling2D):
                raise ValueError("Layer {} is not 2d convolutional or 2d pooling".format(layer_name))
        super(BaseSaliencyMap, self).__init__(model, multiply)
        self.layer_name = layer_name
        if layer_name is None:
            self.layer_is_image = True
        else:
            self.layer_is_image = False

    def get_map(self, x, class_idx):
        """
        Computes saliency map for a given image tensor and class index.

        Parameters
        ----------
        x : numpy.array
            Input image as a numpy array, already preprocessed for the target network model. Shape is: (batch, h, w, c)
        class_idx : int
            Index of the class in the final prediction layer for which to compute saliency

       Returns
       -------
        Saliency map as a [0,255] bounded standardized numpy array.
        """

        if self.layer_is_image:
            layer = self.model.input
        else:
            layer = self.model.get_layer(self.layer_name).output
        
        output = self.model.output[:, class_idx]
        gradient = K.gradients(output, layer)[0]
        
        if self.layer_is_image:
            compute = K.function([self.model.input], [gradient, output])
            gradient_value, output_value = compute([x])
            layer_value = x
        else:
            compute = K.function([self.model.input], [layer, gradient, output])
            layer_value, gradient_value, output_value = compute([x])
        
        if self.multiply:
            gradient_value *= layer_value
        # heatmap takes for each pixel the max gradient across the three rgb channels
        smap = np.max(gradient_value[0], axis=-1)
        smap = np.abs(smap)
        smap /= smap.max()
        smap = floats_to_pixels_standardized(smap)
        return smap

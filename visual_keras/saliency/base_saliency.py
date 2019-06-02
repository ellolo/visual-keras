from keras import backend as K
import numpy as np
from .saliency_map import AbstractSaliencyMap
from ..utils import floats_to_pixels_standardized


class BaseSaliencyMap(AbstractSaliencyMap):
    
    def __init__(self, model, layer_name=None, multiply=False):
        super(BaseSaliencyMap, self).__init__(model, multiply)
        self.layer_name = layer_name
        if layer_name is None:
            self.layer_is_image = True
        else:
            self.layer_is_image = False

    def get_map(self, x, class_idx):
        """
        Compute saliency map for a given image tensor, as described in:
        "Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps"
        Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, 2014

        Parameters:

        modell: keras.engine.training.Model.
            Keras model
        layer_name: string
            Names of prediction layer.
        class_idx: it
            Index of the class for which to predict saliency
        img_tensor: numpy.array of shape (batch, h, w, c)
            Numpy array of the target image.

        Returns:

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

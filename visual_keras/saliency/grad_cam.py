import numpy as np
from keras.layers import Conv2D, Dense
from keras.layers.pooling import _Pooling2D
from keras import backend as K
from .saliency_map import AbstractSaliencyMap
from ..utils import floats_to_pixels_standardized, remove_last_layer_activation


class GradCamMap(AbstractSaliencyMap):
    """
    Class that computes Grad-Cam saliency map for a given image tensor, as described in:
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, 2017

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
        super(GradCamMap, self).__init__(model, multiply)
        self.layer_name = layer_name
        if layer_name is None:
            self.layer_is_image = True
        else:
            self.layer_is_image = False
        # if the last layer is dense we need to remove the final activation function as required by Grad Cam
        last_layer = self.model.layers[-1]
        if isinstance(last_layer, Dense):
            self.model = remove_last_layer_activation(self.model)

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
        # one pooled grad value per filter, i.e. shape is [num_filters]
        pool_grad = K.mean(gradient, axis=(0, 1, 2))
   
        if self.layer_is_image:
            compute = K.function([self.model.input], [pool_grad, output])
            pool_grad_value, output_value = compute([x])
            layer_value = x
        else:
            compute = K.function([self.model.input], [layer, pool_grad, output])
            layer_value, pool_grad_value, output_value = compute([x])

        for filtr in range(layer_value.shape[-1]):
            layer_value[:, :, :, filtr] *= pool_grad_value[filtr]

        # mean across all filters of the layer, i.e. shape is [filter_W, filter_H]
        smap = np.mean(layer_value[0], axis=-1)
        # apply ReLU: keep only positive elements
        smap = np.maximum(smap, 0)
        # normalize for visualization
        smap /= np.max(smap)
        smap = floats_to_pixels_standardized(smap)
        return smap

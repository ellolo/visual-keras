from keras import backend as K
import numpy as np
from .saliency_map import AbstractSaliencyMap
from ..utils import floats_to_pixels_standardized


class GradCamMap(AbstractSaliencyMap):
    
    def __init__(self, model, layer_name=None, multiply=False):
        super(GradCamMap, self).__init__(model, multiply)
        self.layer_name = layer_name
        if layer_name is None:
            self.layer_is_image = True
        else:
            self.layer_is_image = False

    def get_map(self, x, class_idx):
        """
        Compute grad-cam map for a given image tensor.
        
        Parameters:
    
        modell: keras.engine.training.Model.
            Keras model
        layer_name: string
            Names of layer over which to compute grad-cam map.
        img_tensor: numpy.array
            Numpy array of the target image.

        Returns:
    
        grad-cam map as a [0,255] bounded standardized numpy array.
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

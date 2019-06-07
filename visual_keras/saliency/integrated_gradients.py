import numpy as np
from .saliency_map import AbstractSaliencyMap
from .base_saliency import BaseSaliencyMap


class IntegratedGradientMap(AbstractSaliencyMap):
    """
    Class that computes Integrated Gradients saliency map for a given image tensor, as described in:
    "Axiomatic Attribution for Deep Networks"
    M. Sundararajan, A. Taly, Q. Yan, 2017

    Attributes
    ----------
    model : keras.engine.training.Model
        Keras model
    multiply : boolean
        if True if will multiply the map by the input image (default is False)
    """
    
    def __init__(self, model, multiply=False):
        """
        Parameters
        ----------
        model : keras.engine.training.Model
            Keras model
       multiply : boolean
           if True if will multiply the map by the input image (default is False)
        """

        super(IntegratedGradientMap, self).__init__(model, multiply)

    def get_map(self, x, class_idx, samples=50):
        """
        Computes saliency map for a given image tensor and class index.

        Parameters
        ----------
        x : numpy.array
            Input image as a numpy array, already preprocessed for the target network model. Shape is: (batch, h, w, c)
        class_idx : int
            Index of the class in the final prediction layer for which to compute saliency
        samples : int
            number of saliency maps (steps) to compute and to average from (suggested value <300) (default is 50)

       Returns
       -------
       numpy.array
            Saliency map as a [0,255] bounded standardized numpy array.
       """

        baseline = np.zeros(x.shape)
        diff = x - baseline
        smap = np.zeros(x.shape[1:3])
        for step in np.linspace(0, 1, samples):
            curr_img = baseline + step * diff
            curr_grad = BaseSaliencyMap(self.model, multiply=self.multiply).get_map(curr_img, class_idx)
            smap += curr_grad
        smap /= samples
        return smap.astype("uint8")

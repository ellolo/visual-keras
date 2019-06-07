import numpy as np
from .saliency_map import AbstractSaliencyMap
from .base_saliency import BaseSaliencyMap


class SmoothGradMap(AbstractSaliencyMap):
    """
    Class that computes Smmoth Grad saliency map for a given image tensor, as described in:
    "SmoothGrad: removing noise by adding noise"
     D. Smilkov, N. Thorat, B. Kim, F. Viegas, M. Wattenberg, 2017

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

        super(SmoothGradMap, self).__init__(model, multiply)

    def get_map(self, x, class_idx, spread=0.20, samples=50):
        """
        Computes saliency map for a given image tensor and class index.

        Parameters
        ----------
        x : numpy.array
            Input image as a numpy array, already preprocessed for the target network model. Shape is: (batch, h, w, c)
        class_idx : int
            Index of the class in the final prediction layer for which to compute saliency
        spread : float
            controls the magnitude of the standard deviation of gaussian noise (suggested value >0.1 <0.2)
            (default is 0.2)
        samples : int
            number of saliency maps to compute and to average from (suggested value <50) (default is 50)

       Returns
       -------
       numpy.array
           Saliency map as a [0,255] bounded standardized numpy array.
    """

        stdev = spread * (np.max(x) - np.min(x))
        smap = np.zeros(x.shape[1:3])
        for sample in range(samples):
            noise = np.random.normal(0, stdev, x.shape)
            grad = BaseSaliencyMap(self.model, multiply=self.multiply).get_map(x + noise, class_idx)
            smap += grad
        smap /= samples
        return smap.astype("uint8")

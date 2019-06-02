import numpy as np
from .saliency_map import AbstractSaliencyMap
from .base_saliency import BaseSaliencyMap


class SmoothGradMap(AbstractSaliencyMap):
    
    def __init__(self, model, multiply=False):
        super(SmoothGradMap, self).__init__(model, multiply)

    def get_map(self, x, class_idx, spread=0.20, samples=50):
        """
        Implements:
        "SmoothGrad: removing noise by adding noise"
        Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Vi ́egas, Martin Wattenberg, 2017
        """
        stdev = spread * (np.max(x) - np.min(x))
        smap = np.zeros(x.shape[1:3])
        for sample in range(samples):
            noise = np.random.normal(0, stdev, x.shape)
            grad = BaseSaliencyMap(self.model, multiply=self.multiply).get_map(x + noise, class_idx)
            smap += grad
        smap /= samples
        return smap.astype("uint8")

import numpy as np
from .saliency_map import AbstractSaliencyMap
from .base_saliency import BaseSaliencyMap


class IntegratedGradientMap(AbstractSaliencyMap):
    
    def __init__(self, model, multiply=False):
        super(IntegratedGradientMap, self).__init__(model, multiply)

    def get_map(self, x, class_idx, samples=50):
        """
        Implements:
        "Axiomatic Attribution for Deep Networks"
        Mukund Sundararajan, Ankur Taly, Qiqi Yan, 2017
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

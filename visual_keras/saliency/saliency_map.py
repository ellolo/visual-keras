import abc
from abc import abstractmethod

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})  # for compatibility to both Python 2 and 3

class AbstractSaliencyMap(ABC):
    """
    Abstract class for a saliency map.

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

        self.model = model
        self.multiply = multiply
        
    @abstractmethod
    def get_map(self, x, class_idx):
        """
            Computes saliency map for a given image tensor and class index.
        """
        pass

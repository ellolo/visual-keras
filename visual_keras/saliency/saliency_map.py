from abc import ABC, abstractmethod 


class AbstractSaliencyMap(ABC):
    
    def __init__(self, model, multiply=False):
        self.model = model
        self.multiply = multiply
        
    @abstractmethod
    def get_map(self, x, class_idx):
        pass

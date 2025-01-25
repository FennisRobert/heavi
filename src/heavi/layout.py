from abc import ABC, abstractmethod
from .model import Model

class Router(ABC):

    @abstractmethod
    def port(self, *args, **kwargs):
        pass
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error

class CollaborativeMethod(ABC):
    @abstractmethod
    def fit(self, X):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
class RankingMethod(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
class CollaborativePipeline:
    def __init__(self, method: CollaborativeMethod, ranking_method: RankingMethod):
        self.method = method
        self.ranking_method = ranking_method
        
    def run(self, X_train, y_train, X_test):
        self.method.fit(X_train, y_train)
        interations = self.method.predict(X_train)
        
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
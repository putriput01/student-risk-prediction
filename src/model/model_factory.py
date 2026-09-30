from .logistic_model import LogisticModel
from .random_forest_model import RandomForestModel


class ModelFactory:
    @staticmethod
    def create(model_type):
        if model_type == "logistic":
            return LogisticModel()
        if model_type == "random_forest":
            return RandomForestModel()
        raise ValueError(f"Unknown model type: {model_type}")
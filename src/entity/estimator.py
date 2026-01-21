import sys
import pickle
from typing import Any

from src.exception import MyException
from src.logger import logging


class MyModel:
    """
    A wrapper class to bundle the preprocessing object and trained model object for easy serialization and prediction.
    """
    def __init__(self, preprocessing_object: Any, trained_model_object: Any):
        """
        Initialize MyModel with preprocessing and trained model objects.
        
        :param preprocessing_object: The preprocessing pipeline/object
        :param trained_model_object: The trained model object (e.g., RandomForestClassifier)
        """
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, X: Any) -> Any:
        """
        Make predictions using the preprocessing object and trained model.
        
        :param X: Input features
        :return: Predictions
        """
        try:
            logging.info("Entered predict method of MyModel class")
            X_transformed = self.preprocessing_object.transform(X)
            predictions = self.trained_model_object.predict(X_transformed)
            logging.info("Exited predict method of MyModel class")
            return predictions
        except Exception as e:
            raise MyException(e, sys) from e

    def __repr__(self) -> str:
        return f"MyModel(preprocessing_object={self.preprocessing_object}, trained_model_object={self.trained_model_object})"

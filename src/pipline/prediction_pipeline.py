import sys
import os
from pathlib import Path
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
from src.utils.main_utils import load_object
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY


class VehicleData:
    def __init__(self,
                Gender,
                Age,
                Driving_License,
                Region_Code,
                Previously_Insured,
                Annual_Premium,
                Policy_Sales_Channel,
                Vintage,
                Vehicle_Age_lt_1_Year,
                Vehicle_Age_gt_2_Years,
                Vehicle_Damage_Yes
                ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_usvisa_data_as_dict method as VehicleData class")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes]
            }

            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class VehicleDataClassifier:
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of VehicleDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            # If AWS creds are present, use S3; otherwise, fall back to local model
            aws_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            aws_secret = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            if aws_key and aws_secret:
                logging.info("AWS credentials found; using S3 model for prediction")
                model = Proj1Estimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=self.prediction_pipeline_config.model_file_path,
                )
                result = model.predict(dataframe)
            else:
                logging.info("AWS credentials missing; loading latest local trained model")
                # Find latest artifact model.pkl
                artifact_root = Path("artifact")
                if not artifact_root.exists():
                    raise MyException("artifact directory not found for local model fallback", sys)

                latest_model_path = None
                # Sort subdirs by modification time, descending
                subdirs = [d for d in artifact_root.iterdir() if d.is_dir()]
                subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                for d in subdirs:
                    candidate = d / "model_trainer" / "trained_model" / "model.pkl"
                    if candidate.exists():
                        latest_model_path = candidate
                        break

                if latest_model_path is None:
                    raise MyException("No local trained model found in artifact/*/model_trainer/trained_model/", sys)

                logging.info(f"Loading local model from {latest_model_path}")
                local_model = load_object(file_path=str(latest_model_path))
                result = local_model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)
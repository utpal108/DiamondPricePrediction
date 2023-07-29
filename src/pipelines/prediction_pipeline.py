import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import load_object


class PredictionPipeline:

    def predict(self, features):

        try:
            # Preprocessor and Model Pickle file path
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load Preprocessor and Model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Scale the Features and Predict
            features_scale = preprocessor.fit_transform(features)
            predict = model.predict(features_scale)
            return predict

        except Exception as e:
            logging.info('Prediction Exception Occurred')
            raise CustomException(e, sys)

# Generate Dataframe From Input Data
class CustomData:
    def __init__(self, carat:float, depth:float, table:float, x:float, y:float, z:float, cut:str, color:str, clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            # Generate the Dataframe
            custom_input_data = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }

            df = pd.DataFrame(custom_input_data)
            logging.info('Dataframe Generated from Input Data')
            return df

        except Exception as e:
            logging.info('Custom Data Generation Exception')
            raise CustomException(e, sys)







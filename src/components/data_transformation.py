import numpy as np
import pandas as pd
import sys
import os
from src.logger import logging
from src.exception import CustomException

from sklearn.impute import SimpleImputer  # For Handling Missing Values
from sklearn.preprocessing import StandardScaler  # For Feature Scaling
from sklearn.preprocessing import OrdinalEncoder  # For Ordinal Encoding

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')

            # Categorical amd Numerical Features
            categorical_features = ['cut', 'color', 'clarity']
            numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define custom ranking for each ordinal values
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Data Transformation Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',
                     OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            logging.info('Data Transformation Completed')

            return preprocessor



        except Exception as e:
            logging.info('Exception Occurred in Data Transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Reading the train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading Train and Test Data Completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            # Splitting the data into independent and dependent features
            # For Training data
            input_features_train_df = train_df.drop(['id', 'price'], axis=1)
            target_features_train_df = train_df['price']

            # For Test data
            input_features_test_df = test_df.drop(['id', 'price'], axis=1)
            target_features_test_df = test_df['price']

            # Data Transformation
            preprocessor = self.get_data_transformation_object()

            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor.fit_transform(input_features_test_df)

            logging.info('Train and Test Data Transformation Done')

            # Combine/Concatenate Train and Test Features after data Transformation
            train_arr = np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            # Save the preprocessor object
            save_object(self.data_transformation.preprocessor_obj_file_path, preprocessor)

            logging.info('Saved Preprocessed Object Done')

            return train_arr, test_arr, self.data_transformation.preprocessor_obj_file_path

        except Exception as e:
            logging.info('Exception Occurred on Data Transformation Initialization')
            raise CustomException(e, sys)

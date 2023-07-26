import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


# Create the Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_info = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Starts')

        try:
            # Read the Data and Save it as Raw Dataset
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            os.makedirs(os.path.dirname(self.ingestion_info.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_info.raw_data_path, index=False)
            logging.info('Raw data is stored')

            # Split the Data in Train and Test set and Save in CSV File
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_info.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_info.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed')

            return self.ingestion_info.train_data_path, self.ingestion_info.test_data_path

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)

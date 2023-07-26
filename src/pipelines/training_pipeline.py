import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

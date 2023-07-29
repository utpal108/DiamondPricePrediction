import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.train_model_config = ModelTrainerConfig().trained_model_path

    def initiate_model_training(self, train_arr, test_arr):
        try:
            # Split Train, test data
            X_train = train_arr[:, :-1]
            X_test = test_arr[:, :-1]
            y_train = train_arr[:, -1]
            y_test = test_arr[:, -1]

            # List of the Models
            # List of Regression Model
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet(),
                # 'DecisionTreeRegressor': DecisionTreeRegressor(),
                # 'KNeighborsRegressor': KNeighborsRegressor(),
                # 'RandomForestRegressor': RandomForestRegressor()
            }

            # Find the best Model
            model_report, best_model = evaluate_model(models, X_train, X_test, y_train, y_test)
            logging.info(f'Model Evaluation Report : {model_report}')
            logging.info(f'Best Model : {best_model}')

            best_model = models[list(best_model.keys())[0]]

            # Save the Best Model in Pickle File
            save_object(file_path=self.train_model_config, obj=best_model)

        except Exception as e:
            logging.info('Model Trainer Initialization Exception')
            CustomException(e, sys)

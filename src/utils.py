import logging
import sys
import os
import pickle
import numpy as np

from src.exception import CustomException
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(models, X_train, X_test, y_train, y_test):
    try:
        report = {}
        best_model = {'': -np.inf}

        # Evaluate the models base on the r2 scores
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            if list(best_model.values())[0] < score:
                best_model = {model_name: score}

            report[model_name] = score

        return report, best_model

    except Exception as e:
        logging.info('Model Evaluation Exception')
        raise CustomException(e, sys)

import os
import sys 

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_   # ✅ IMPORTANT

            best_model.fit(x_train, y_train)

            y_test_pred = best_model.predict(x_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            trained_models[model_name] = best_model  # ✅ store trained model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
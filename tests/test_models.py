import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException

from models.DecisionTreeRegressor import dtr_param_selector
from models.LogisticRegression import lr_param_selector
from models.RandomForestClassifier import rf_param_selector
from models.RandomForestRegressor import rfr_param_selector
from models.GradientBoostingClassifier import gbc_param_selector
from models.GradientBoostingRegressor import gbr_param_selector
from models.AdaBoost import ada_param_selector
from models.SupportVectorRegressor import svr_param_selector
from models.DecisionTreeClassifier import dt_param_selector
from models.LinearRegression import lr_param_selector



# Mock data for testing
X_train = np.random.rand(100, 5)
Y_train_reg = np.random.rand(100)
Y_train_clf = np.random.randint(0, 2, 100)

try:
    # Test regression models
    model, duration = dtr_param_selector(X_train, Y_train_reg)
    
    model, duration = rfr_param_selector(X_train, Y_train_reg)
    
    model, duration = gbr_param_selector(X_train, Y_train_reg)
    
    model, duration = svr_param_selector(X_train, Y_train_reg)
    
    model, duration = lr_param_selector(X_train, Y_train_reg)
    
    # Test classification models
    model, duration = lr_param_selector(X_train, Y_train_clf)
    
    model, duration = rf_param_selector(X_train, Y_train_clf)

    model, duration = gbc_param_selector(X_train, Y_train_clf)
    
    model, duration = ada_param_selector(X_train, Y_train_clf)

    model, duration = dt_param_selector(X_train, Y_train_clf)
    
except Exception as e:
    raise AutoMLException(e, sys)
    
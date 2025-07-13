import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from sklearn.svm import SVR

def svr_param_selector(X_train, Y_train):
    try:
        logging.info("Configuring SupportVectorRegressor parameters")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        gamma = st.selectbox("Gamma", ["scale", "auto"])
        C = st.number_input("C (Regularization Parameter)", min_value = 0.01, max_value = 10.0, value = 1.0, step = 0.1)
        epsilon = st.number_input("Epsilon", min_value = 0.01, max_value = 1.0, value = 0.1, step = 0.01)
        
        params = {
            "kernel": kernel,
            "gamma": gamma,
            "C": C,
            "epsilon": epsilon
        }
        
        logging.info(f"Training SupportVectorRegressor with params: {params}")
        model = SVR(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"SupportVectorRegressor trained in {duration:.3f} seconds")
        return model, duration
    
    except Exception as e:
        raise AutoMLException(e, sys)
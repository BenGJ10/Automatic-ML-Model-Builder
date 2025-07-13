import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from sklearn.linear_model import LinearRegression

def lr_param_selector(X_train, Y_train):
    try:
        logging.info("Configuring LinearRegression parameters")
        fit_intercept = st.checkbox("Fit Intercept", value=True)
        params = {
            "fit_intercept": fit_intercept
        }
        
        logging.info(f"Training LinearRegression with params: {params}")
        model = LinearRegression(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"LinearRegression trained in {duration:.3f} seconds")
        return model, duration
    
    except Exception as e:
        raise AutoMLException(e, sys)
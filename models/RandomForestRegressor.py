import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logging.logger import logging
from src.exception.exception import AutoMLException
from sklearn.ensemble import RandomForestRegressor

def rfr_param_selector(X_train, Y_train):
    try:
        logging.info("Configuring RandomForestRegressor parameters")
        n_estimators = st.number_input("Number of Estimators", min_value = 10, max_value = 200, value = 100, step = 10)
        max_depth = st.number_input("Max Depth", min_value=1, max_value = 50, value = 10, step = 1)
        min_samples_split = st.number_input("Min Samples Split", min_value = 2, max_value = 20, value = 2, step = 1)
        max_features = st.selectbox("Max Features", ["sqrt", "log2", None])
        
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features
        }
        
        logging.info(f"Training RandomForestRegressor with params: {params}")
        model = RandomForestRegressor(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"RandomForestRegressor trained in {duration:.3f} seconds")
        return model, duration

    except AutoMLException as e:
        raise AutoMLException(e, sys)

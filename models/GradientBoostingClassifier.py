import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from sklearn.ensemble import GradientBoostingClassifier

def gbc_param_selector(X_train, Y_train):
    try:
        logging.info("Configuring GradientBoostingClassifier parameters")
        loss = st.selectbox("Loss Function", ["log_loss", "exponential"])
        criterion = st.selectbox("Criterion", ["friedman_mse", "squared_error", "mae"])
        n_estimators = st.number_input("Number of Estimators", min_value = 10, max_value = 200, value = 100, step = 10)
        learning_rate = st.number_input("Learning Rate", min_value = 0.01, max_value = 1.0, value = 0.1, step = 0.01)
        random_state = st.number_input("Random State", min_value = 0, max_value = 1000, value = 0, step = 1, key = "xgboost")
        max_depth = st.number_input("Max Depth", min_value = 1, max_value = 50, value = 3, step = 1)
        max_features_option = st.selectbox("Max Features", ["None (use all)", "sqrt", "log2"])
        max_features = None if max_features_option.startswith("None") else max_features_option


        params = {
            "loss": loss,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "random_state": random_state,
            "max_features": max_features
        }
        
        logging.info(f"Training GradientBoostingClassifier with params: {params}")
        model = GradientBoostingClassifier(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"GradientBoostingClassifier trained in {duration:.3f} seconds")
        return model, duration
    
    except Exception as e:
        raise AutoMLException(e, sys)
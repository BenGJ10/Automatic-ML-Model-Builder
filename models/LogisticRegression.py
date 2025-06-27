import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logging.logger import logging
from src.exception.exception import AutoMLException
from sklearn.linear_model import LogisticRegression

def lr_param_selector(X_train, Y_train):
    try:
        logging.info("Configuring LogisticRegression parameters")

        penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
        C = st.number_input("C (Inverse of Regularization Strength)", min_value = 0.01, max_value = 10.0, value = 1.0, step = 0.1)
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        
        params = {
            "penalty": penalty,
            "C": C,
            "solver": solver
        }
        
        logging.info(f"Training LogisticRegression with params: {params}")
        model = LogisticRegression(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"LogisticRegression trained in {duration:.3f} seconds")
        return model, duration
    
    except AutoMLException as e:
        raise AutoMLException(e, sys)
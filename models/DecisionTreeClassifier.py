import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from sklearn.tree import DecisionTreeClassifier

def dt_param_selector(X_train, Y_train):
    try:
        logging.info("Configuring DecisionTreeClassifier parameters")
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.number_input("Max Depth", min_value = 1, max_value = 50, value = 5, step = 1)
        min_samples_split = st.number_input("Min Samples Split", min_value = 2, max_value = 20, value = 2, step= 1)
        max_features = st.selectbox("Max Features", [None, "sqrt", "log2"])
        
        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features
        }
        
        logging.info(f"Training DecisionTreeClassifier with params: {params}")
        model = DecisionTreeClassifier(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"DecisionTreeClassifier trained in {duration:.3f} seconds")
        return model, duration
    
    except Exception as e:
        raise AutoMLException(e, sys)
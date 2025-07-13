import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import streamlit as st
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def ada_param_selector(X_train, Y_train):

    try:
        logging.info("Configuring AdaBoost parameters")
        base_estimator = st.selectbox("base_estimator", ["Decision Tree Classifier"])
        base_estimator1 = DecisionTreeClassifier(max_depth = 1)
        
        if base_estimator == "Decision Tree Classifier":
            criterion = st.selectbox("Criterion for " + base_estimator, ["gini", "entropy"])
            max_depth = st.number_input("Max Depth for " + base_estimator, 1, 50, 5, 1)
            min_samples_split = st.number_input(
                "Min Samples split for " + base_estimator, 1, 20, 2, 1
            )
            max_features_option = st.selectbox("Max Features", ["None (all features)", "sqrt", "log2"])
            max_features = None if max_features_option.startswith("None") else max_features_option

            paramsDT = {
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
            }
            base_estimator1 = DecisionTreeClassifier(**paramsDT)

        base_estimator = base_estimator1
        n_estimators = st.number_input("Number of Estimators", 1, 1000, 100, 10)
        learning_rate = st.number_input("Learning Rate", 0.0, 10.0, 0.1, 0.1)
        random_state = st.number_input("Random State", 0, 1000, 0, 1, key = "ada")

        params = {
            "estimator": base_estimator1,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }

        logging.info(f"Training AdaBoost with params: {params}")
        model = AdaBoostClassifier(**params)
        t0 = time.time()
        model.fit(X_train, Y_train)
        duration = time.time() - t0
        logging.info(f"AdaBoost trained in {duration:.3f} seconds")
        return model, duration
    
    except Exception as e:
        raise AutoMLException(e, sys)
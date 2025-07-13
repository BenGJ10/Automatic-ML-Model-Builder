import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import streamlit as st
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.utils import model_imports, model_urls, model_infos

from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, root_mean_squared_error
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
from src.functions import get_model_url, get_model_tips, generate_snippet, plot_metrics, save_model_history

def introduction():
    try:
        st.title("**Welcome to AutoML Studio**")
        st.subheader("Train machine learning models right from your browser")
        st.markdown("""
        - 🗂️ Upload a **pre-processed** CSV dataset
        - ⚙️ Select a model and configure its hyperparameters
        - 📉 Train and evaluate performance on train/test data
        - 🩺 Diagnose overfitting and experiment with settings
        -----
        """)
        logging.info("Rendered introduction")
    except Exception as e:
        logging.error(f"Introduction error: {str(e)}")
        raise AutoMLException(f"Introduction error: {str(e)}", sys)
    
def dataset_upload():
    try:
        with st.sidebar.expander("Upload a Dataset", expanded = True):
            dataset = st.file_uploader("Upload Dataset", type = ["csv"])
            if dataset is not None:
                dfname = dataset.name
                df = pd.read_csv(dataset)
                problem_type = st.selectbox("Type of Problem", ("Regression", "Classification"))
                dependent_column = st.text_input("Enter the Dependent Variable")
                if dependent_column:
                    y = df[dependent_column]
                    X = df.drop(dependent_column, axis=1)
                    logging.info(f"Dataset loaded: {dfname}, shape: {df.shape}")
                    return [problem_type, dependent_column, df, X, y, dfname]
                else:
                    st.error("Please enter a valid dependent variable name.")
                    logging.warning("Dependent variable not specified")
            else:
                st.info("Please upload a CSV file.")
                logging.info("No dataset uploaded")
        return None
    except Exception as e:
        logging.error(f"Dataset upload error: {str(e)}")
        raise AutoMLException(f"Dataset upload error: {str(e)}", sys)
    

def split_data(result):
    try:
        with st.sidebar.expander("Data Splitting", expanded=True):
            train_size_in_percent = st.number_input("Train Size in %", 0, 100, 80, 1)
            test_size = 1 - (float(train_size_in_percent) / 100)
            random_state = st.number_input("random_state", 0, 1000, 42, 1)
            X_train, X_test, y_train, y_test = train_test_split(
                result[3], result[4], test_size=test_size, random_state=random_state
            )
            st.write(f"Shape of X train: {X_train.shape}")
            st.write(f"Shape of X test: {X_test.shape}")
            logging.info(f"Data split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test, test_size, random_state
    except Exception as e:
        logging.error(f"Data splitting error: {str(e)}")
        raise AutoMLException(f"Data splitting error: {str(e)}", sys)

def scale_data(result, X_train, X_test):
    try:
        with st.sidebar.expander("Data Scaling", expanded=True):
            st.write("Select Scaling Method")
            standard_scaler = st.checkbox("StandardScaler")
            minmax_scaler = st.checkbox("MinMaxScaler")
            no_scaling = st.checkbox("None")
            columns = st.text_input("Enter columns to scale (comma-separated)")
            column_list = [col.strip() for col in columns.split(",") if col.strip()]
            if not columns or no_scaling:
                logging.info("No scaling applied")
                return X_train, X_test
            if standard_scaler and minmax_scaler:
                st.error("Please select only one scaling method")
                return X_train, X_test
            if standard_scaler:
                for col in column_list:
                    scaler = StandardScaler()
                    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
                st.write("Scaled X_test preview:", X_test.head())
                logging.info(f"Applied StandardScaler to columns: {column_list}")
                return X_train, X_test
            if minmax_scaler:
                for col in column_list:
                    scaler = MinMaxScaler()
                    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
                st.write("Scaled X_test preview:", X_test.head())
                logging.info(f"Applied MinMaxScaler to columns: {column_list}")
                return X_train, X_test
    except Exception as e:
        logging.error(f"Data scaling error: {str(e)}")
        raise AutoMLException(f"Data scaling error: {str(e)}", sys)

def model_selector(problem_type, X_train, y_train):
    try:
        with st.sidebar.expander("Train a Model", expanded=True):
            model_map = {
                "Classification": {
                    "LogisticRegression": lr_param_selector,
                    "RandomForestClassifier": rf_param_selector,
                    "GradientBoostingClassifier": gbc_param_selector,
                    "AdaBoostClassifier": ada_param_selector,
                    "DecisionTreeClassifier": dt_param_selector
                },
                "Regression": {
                    "DecisionTreeRegressor": dtr_param_selector,
                    "RandomForestRegressor": rfr_param_selector,
                    "GradientBoostingRegressor": gbr_param_selector,
                    "SupportVectorRegressor": svr_param_selector,
                    "LinearRegression": lr_param_selector
                }
            }
            available_models = model_map.get(problem_type, {})
            if not available_models:
                raise AutoMLException(f"No models available for {problem_type}", sys)
            model_type = st.selectbox("Choose a model", list(available_models.keys()))
            st.markdown(f"**Model Info**: {model_infos[model_type]}")
            st.markdown(get_model_url(model_type), unsafe_allow_html=True)
            if st.button("Train Model"):
                logging.info(f"Training {model_type}")
                model_func = available_models[model_type]
                model, duration = model_func(X_train, y_train)
                return model_type, model, duration, problem_type
            return None, None, None, problem_type
    except AutoMLException as e:
        logging.error(f"Model selection error: {str(e)}")
        st.error(str(e))
        return None, None, None, problem_type

def evaluate_model(model, X_train, y_train, X_test, y_test, duration, problem_type):
    try:
        logging.info(f"Evaluating {problem_type} model")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        if problem_type == "Classification":
            train_accuracy = round(accuracy_score(y_train, y_train_pred), 3)
            train_f1 = round(f1_score(y_train, y_train_pred, average="weighted"), 3)
            test_accuracy = round(accuracy_score(y_test, y_test_pred), 3)
            test_f1 = round(f1_score(y_test, y_test_pred, average="weighted"), 3)
            logging.info(f"Classification metrics: train_accuracy={train_accuracy}, test_accuracy={test_accuracy}")
            return model, train_accuracy, train_f1, test_accuracy, test_f1
        elif problem_type == "Regression":
            train_mse = round(mean_squared_error(y_train, y_train_pred), 3)
            train_rmse = round(root_mean_squared_error(y_train, y_train_pred), 3)
            test_mse = round(mean_squared_error(y_test, y_test_pred), 3)
            test_rmse = round(root_mean_squared_error(y_test, y_test_pred), 3)
            return model, train_mse, train_rmse, test_mse, test_rmse
    except Exception as e:
        logging.error(f"Model evaluation error: {str(e)}")
        raise AutoMLException(f"Model evaluation error: {str(e)}", sys)

def footer():
    try:
        st.sidebar.markdown("""
        [<img src='https://github.com/favicon.ico' class='img-fluid' width=30 height=30>](https://github.com/BenGJ10/Automatic-ML-Model-Builder) <small> AutoML Studio | July 2025</small>
        """, unsafe_allow_html=True)
        logging.info("Rendered footer")
    except Exception as e:
        logging.error(f"Footer error: {str(e)}")
        raise AutoMLException(f"Footer error: {str(e)}", sys)
    
    
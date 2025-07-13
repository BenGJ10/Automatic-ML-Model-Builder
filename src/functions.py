import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from models.utils import model_imports, model_urls, model_infos

def load_css(file_name):
    """
    Load a CSS file and apply its styles to the Streamlit app.
    """
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        logging.info(f"CSS loaded from {file_name}")
    except Exception as e:
        logging.error(f"Failed to load CSS: {str(e)}")
        raise AutoMLException(f"Failed to load CSS: {str(e)}", sys)
    
def get_model_url(model_type):
    """ 
    Retrieve the URL for the documentation of a specific model type.
    """
    try:
        url = model_urls[model_type]
        text = f"**Link to scikit-learn documentation [here]({url}) 💻**"
        logging.info(f"Retrieved URL for {model_type}")
        return text
    except Exception as e:
        logging.error(f"Failed to get model URL: {str(e)}")
        raise AutoMLException(f"Failed to get model URL: {str(e)}", sys)

def get_model_tips(model_type):
    """
    Retrieve tips for a specific model type.
    """
    try:
        tips = model_infos[model_type]
        logging.info(f"Retrieved tips for {model_type}")
        return tips
    except Exception as e:
        logging.error(f"Failed to get model tips: {str(e)}")
        raise AutoMLException(f"Failed to get model tips: {str(e)}", sys)
    
def plot_metrics(metrics, problem_type):
    """
    Plot the evaluation metrics for the model based on the problem type.
    
    """
    try:
        logging.info(f"Plotting metrics for {problem_type}")
        if problem_type == "Classification":
            fig = make_subplots( 
                rows = 2, cols = 1, specs = [[{"type": "indicator"}], [{"type": "indicator"}]], 
                row_heights = [0.7, 0.3]
            )
            fig.add_trace( 
                go.Indicator(
                    mode = "gauge+number+delta", value = metrics["test_accuracy"],
                    title = {"text": "Accuracy (test)"}, gauge = {"axis": {"range": [0, 1]}},
                    delta = {"reference": metrics["train_accuracy"]}
                ),
                row = 1, col = 1
            )
            fig.add_trace( 
                go.Indicator(
                    mode = "gauge+number+delta", value = metrics["test_f1"],
                    title = {"text": "F1 Score (test)"}, gauge = {"axis": {"range": [0, 1]}},
                    delta = {"reference": metrics["train_f1"]}
                ),
                row = 2, col = 1
            )
            fig.update_layout(height = 700)
            return fig
        
        elif problem_type == "Regression":
            fig = make_subplots(
                rows = 2, cols = 1, specs = [[{"type": "indicator"}], [{"type": "indicator"}]],
                row_heights = [0.7, 0.3]
            )
            fig.add_trace(
                go.Indicator(
                    mode = "number+delta", value = metrics["test_mse"],
                    title = {"text": "MSE (test)"}, delta = {"reference": metrics["train_mse"], "increasing.color": "red"}
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Indicator(
                    mode = "number+delta", value = metrics["test_rmse"],
                    title = {"text": "RMSE (test)"}, delta = {"reference": metrics["train_rmse"], "increasing.color": "red"}
                ),
                row = 2, col = 1
            )
            fig.update_layout(height = 700)
            return fig
    except Exception as e:
        logging.error(f"Failed to plot metrics: {str(e)}")
        raise AutoMLException(f"Failed to plot metrics: {str(e)}", sys)
    
def generate_snippet(model, model_type, dataset_name, test_size, random_state, dependent_column, problem_type):
    """ 
    Generate a code snippet for training and evaluating the model.
    This function creates a code snippet based on the model type, dataset, and problem type.
    """

    try:
        logging.info(f"Generating code snippet for {model_type}")
        model_text_rep = repr(model)
        model_import = model_imports[model_type]
        dataset_import = f"df = pd.read_csv('{dataset_name}')"
        if problem_type == "Classification":
            snippet = f"""
{model_import}
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

{dataset_import}
dependent_column = '{dependent_column}'
y = df[dependent_column]
X = df.drop(dependent_column, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={round(test_size, 2)}, random_state={random_state}
)
model = {model_text_rep}
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            """
            return snippet
        elif problem_type == "Regression":
            snippet = f"""
{model_import}
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

{dataset_import}
dependent_column = '{dependent_column}'
y = df[dependent_column]
X = df.drop(dependent_column, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={round(test_size, 2)}, random_state={random_state}
)
model = {model_text_rep}
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
            """
            return snippet
    except Exception as e:
        logging.error(f"Failed to generate snippet: {str(e)}")
        raise AutoMLException(f"Failed to generate snippet: {str(e)}", sys)
    

def save_model_history(model, model_type, metrics, dataset_name):
    """ Save the model training history to a file."""
    try:
        from datetime import datetime
        with open("data/model_history.txt", "a") as f:
            f.write(f"\n\nTrained at: {datetime.now()}\n")
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write(f"Model: {model_type}\n")
            f.write(f"Parameters: {str(model)}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("--" * 40 + "\n")
        logging.info(f"Saved model history for {model_type}")
    except Exception as e:
        logging.error(f"Failed to save model history: {str(e)}")
        raise AutoMLException(f"Failed to save model history: {str(e)}", sys)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from src.logging.logger import logging
from src.exception.exception import AutoMLException

class DataPreprocessor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.target_column = None
        self.problem_type = None

    def upload_dataset(self):
        try:
            logging.info("Initiating dataset upload")
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file)
                logging.info(f"Dataset loaded: {uploaded_file.name}, shape: {self.df.shape}")
                st.write("Dataset Preview:")
                st.dataframe(self.df.head())
                return True
            return False
        except Exception as e:
            raise AutoMLException(f"Failed to upload dataset: {str(e)}", sys)
    
    def select_target_and_problem_type(self):
        try:
            if self.df is None:
                raise AutoMLException("No dataset loaded", sys)
            logging.info("Selecting target column and problem type")
            self.target_column = st.selectbox("Select Target Column", self.df.columns)
            self.problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])
            logging.info(f"Selected target: {self.target_column}, problem type: {self.problem_type}")
        except Exception as e:
            raise AutoMLException(f"Failed to select target or problem type: {str(e)}", sys)
    
    def split_data(self):
        try:
            if self.df is None or self.target_column is None:
                raise AutoMLException("Dataset or target column not set", sys)
            logging.info("Splitting dataset into train and test sets")
            test_size = st.slider("Test Size (%)", min_value = 10, max_value = 50, value = 20, step = 5) / 100
            random_state = st.number_input("Random State", min_value = 0, max_value = 100, value = 42)
            
            X = self.df.drop(columns=[self.target_column])
            Y = self.df[self.target_column]
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                X, Y, test_size = test_size, random_state = random_state
            )

            logging.info(f"Data split: X_train shape: {self.X_train.shape}, X_test shape: {self.X_test.shape}")
            st.write(f"Training set shape: {self.X_train.shape}")
            st.write(f"Test set shape: {self.X_test.shape}")
        except Exception as e:
            logging.error(f"Data splitting failed: {str(e)}")
            raise AutoMLException(f"Failed to split dataset: {str(e)}", sys)
    
    def get_data(self):
        if self.X_train is None or self.Y_train is None:
            raise AutoMLException("Data not split", sys)
        return self.X_train, self.X_test, self.Y_train, self.Y_test, self.problem_type
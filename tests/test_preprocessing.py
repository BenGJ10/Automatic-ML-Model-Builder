import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
from src.logger.custom_logger import logging
from src.exception.exception import AutoMLException
from src.preprocessing import DataPreprocessor


def test_preprocessing():
    try:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        logging.info("Starting preprocessing test")

        # Create mock dataset
        df = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.randint(0, 2, 100)
        })
        df.to_csv("data/test.csv", index = False)
        
        # Simulate preprocessing
        preprocessor = DataPreprocessor()
        preprocessor.df = pd.read_csv("data/test.csv")
        preprocessor.target_column = "target"
        preprocessor.problem_type = "Classification"
        preprocessor.split_data()
        X_train, X_test, Y_train, _test, problem_type = preprocessor.get_data()
        
        logging.info(f"Preprocessing test successful: X_train shape: {X_train.shape}, problem_type: {problem_type}")
    
    except AutoMLException as e:
        logging.error(f"Preprocessing test failed: {e}")
        raise

if __name__ == "__main__":
    test_preprocessing()
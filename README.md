# AutoML Studio

**AutoML Studio** is a Streamlit-based web application that automates the machine learning pipeline for supervised learning tasks (classification and regression). It allows users to upload datasets, preprocess data, select and configure machine learning models, train them, evaluate performance, and visualize results interactively. Built with Python, Scikit-learn, Streamlit, and Plotly, this project.

## Features

- **Model Support**: Train and evaluate 10 supervised learning models from Scikit-learn:
  - Classification: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier
  - Regression: DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, SupportVectorRegressor, LinearRegression
- **Interactive Interface**: Streamlit-based UI with sidebar controls for dataset upload, data splitting, feature scaling, model selection, and hyperparameter tuning.
- **Data Preprocessing**: Upload CSV datasets, select target variables, split data into train/test sets, and apply StandardScaler or MinMaxScaler to specified columns.
- **Model Evaluation**: Compute and visualize performance metrics (accuracy, F1-score for classification; MSE, RMSE for regression) using Plotly charts.
- **Code Snippet Generation**: Generate Python code to replicate trained models outside the application.
- **Model History**: Save and view model training history, including parameters and metrics.
- **Robust Logging**: Timestamped log files for tracking application events and errors.
- **Custom Exception Handling**: Comprehensive error management for reliable operation.
- **Documentation**: Detailed model metadata with Scikit-learn documentation links and usage tips.

---

## Project Structure

```
AutoML-Studio/
├── data/                      # Store datasets and model history
│
├── models/                    # Model scripts and metadata
│   ├── DecisionTreeRegressor.py
│   ├── LogisticRegression.py
│   ├── RandomForestClassifier.py
│   ├── RandomForestRegressor.py
│   ├── GradientBoostingClassifier.py
│   ├── GradientBoostingRegressor.py
│   ├── AdaBoost.py
│   ├── SupportVectorRegressor.py
│   ├── DecisionTreeClassifier.py
│   ├── LinearRegression.py
│   └── utils.py
│
├── images/                    # Optional images (e.g., GitHub logo)
│
├── css/                       # Styling for Streamlit UI
│   └── style.css
│
├── src/                       # Core application logic
│   ├── exception/
│   │   ├── __init__.py
│   │   └── exception.py
│   │
│   ├── logger/
│   │   ├── __init__.py
│   │   └── custom_logger.py
│   │
│   ├── ui.py
│   ├── functions.py
│   ├── preprocessing.py
│   └── main.py
│
├── logs/                      # Timestamped log files
│
├── tests/                     # Test scripts
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── setup.py                   # Setup script
```

---

## Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/BenGJ10/Automatic-ML-Model-Builder.git

cd AutoML-Studio
   ```

2. **Install Dependencies**:

```
pip install -r requirements.txt
```

3. **Run the Application**:
```
streamlit run src/main.py
```

---

## Usage

1. Launch the App: Run streamlit run src/main.py and open http://localhost:8501 in a web browser.


2. Upload a Dataset:

    - Use the sidebar to upload a pre-processed CSV file.

    - Select the problem type (Classification or Regression) and the target column.


3. Preprocess Data:

    - Specify train/test split ratios and random state.

    - Optionally apply StandardScaler or MinMaxScaler to selected columns.


4. Train a Model:

    - Choose a model from the available options based on the problem type.

    - Configure hyperparameters using interactive inputs.

    - Click "Train Model" to fit the model to the training data.


5. Evaluate Performance:

    - View performance metrics (accuracy, F1-score for classification; MSE, RMSE for regression) in Plotly visualizations.

    - Check training duration and model tips.



6. Export and Review:

    - Generate Python code snippets to replicate the model training.

    - Save model history to data/model_history.txt and view past models.



7. Check Logs:

    - Review timestamped log files in the logs/ directory for detailed application events.
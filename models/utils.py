model_imports = {
    "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor",
    "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor"
}

model_urls = {
    "DecisionTreeRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    "LogisticRegression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "RandomForestClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "RandomForestRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    "GradientBoostingClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "GradientBoostingRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html"
}

model_infos = {
    "DecisionTreeRegressor": """
    DecisionTreeRegressor is a non-linear regression model that splits the data into subsets based on feature values. 
    It is useful for capturing complex relationships in the data.""",

    "LogisticRegression": """"
    LogisticRegression is a linear model for binary classification. It estimates the probability of a class label based
    on a linear combination of input features. It is effective for linearly separable data.""",
    
    "RandomForestClassifier": """
    RandomForestClassifier is an ensemble method that builds multiple decision trees and merges them to improve accuracy
    and control overfitting. It is robust to noise and can handle large datasets.""",
    
    "RandomForestRegressor": """
    RandomForestRegressor is similar to RandomForestClassifier but used for regression tasks. It averages the predictions
    from multiple decision trees to provide a more accurate and stable prediction.""",
    
    "GradientBoostingClassifier": """
    GradientBoostingClassifier is an ensemble method that builds trees sequentially, where each tree correct
    the errors of the previous ones. It is effective for both classification and regression tasks, especially with
    complex datasets.""",

    "GradientBoostingRegressor": """
    GradientBoostingRegressor is similar to GradientBoostingClassifier but used for regression tasks.
    It builds trees sequentially to minimize the residuals of the previous trees, providing a powerful
    method for regression problems with complex relationships."""
}
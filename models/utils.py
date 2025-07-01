model_imports = {
    "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor",
    "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
    "AdaBoostClassifier": "from sklearn.ensemble import AdaBoostClassifier",
    "SupportVectorRegressor": "from sklearn.svm import SVR",
    "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
    "LinearRegression": "from sklearn.linear_model import LinearRegression"
}

model_urls = {
    "DecisionTreeRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    "LogisticRegression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "RandomForestClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "RandomForestRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    "GradientBoostingClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "GradientBoostingRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
    "AdaBoostClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
    "SupportVectorRegressor": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
    "DecisionTreeClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "LinearRegression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
}

model_infos = {
    "DecisionTreeRegressor": """
    DecisionTreeRegressor is a non-linear regression model that splits the data into subsets based on feature values. 
    It is useful for capturing complex relationships in the data.""",

    "LogisticRegression": """
    LogisticRegression is a linear model for binary classification. It estimates the probability of a class label based
    on a linear combination of input features. It is effective for linearly separable data.""",

    "RandomForestClassifier": """
    RandomForestClassifier is an ensemble method that builds multiple decision trees and merges them to improve accuracy
    and control overfitting. It is robust to noise and can handle large datasets.""",

    "RandomForestRegressor": """
    RandomForestRegressor is similar to RandomForestClassifier but used for regression tasks. It averages the predictions
    from multiple decision trees to provide a more accurate and stable prediction.""",

    "GradientBoostingClassifier": """
    GradientBoostingClassifier is an ensemble method that builds trees sequentially, where each tree corrects
    the errors of the previous ones. It is effective for both classification and regression tasks, especially with
    complex datasets.""",

    "GradientBoostingRegressor": """
    GradientBoostingRegressor is similar to GradientBoostingClassifier but used for regression tasks.
    It builds trees sequentially to minimize the residuals of the previous trees, providing a powerful
    method for regression problems with complex relationships.""",

    "AdaBoostClassifier": """
    AdaBoostClassifier is an ensemble boosting method that combines multiple weak learners (typically decision trees) to 
    create a strong classifier. It focuses on misclassified instances, improving accuracy but can be sensitive to noise.""",

    "SupportVectorRegressor": """
    SupportVectorRegressor is a regression model based on Support Vector Machines. It finds a function that predicts 
    continuous values within a margin of tolerance (epsilon). It performs well on small datasets with scaled features.""",

    "DecisionTreeClassifier": """
    DecisionTreeClassifier is a non-linear classification model that splits data into subsets based on feature values. 
    It is interpretable but prone to overfitting unless parameters like max_depth are tuned.""",
    
    "LinearRegression": """
    LinearRegression is a simple linear model for regression tasks. It fits a linear equation to the data, assuming a 
    linear relationship between features and target. It is effective for simple datasets with linear patterns."""
}
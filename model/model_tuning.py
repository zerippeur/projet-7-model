from typing import List, Tuple, Literal, Optional, Union, Any, Dict, Callable, Iterable, TypeVar, Generic, Type, cast, no_type_check

import numpy as np
import pandas as pd
from random import randint
from scipy.stats import uniform, loguniform, randint as sp_randint

import time
from contextlib import contextmanager

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

import mlflow

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title: str):
    """
    A context manager that measures the time taken to execute a block of code.

    Args:
        title (str): The title of the code block.

    Yields:
        None

    Prints:
        The title of the code block and the time taken to execute it.

    Example:
        >>> with timer("Function execution"):
        ...     # code block
        Function execution - done in 5s
    """
    start_time = time.time()
    yield
    print(f"{title} - done in {time.time() - start_time:.0f}s")

@contextmanager
def mlflow_run(title: str):
    """
    A context manager that starts and ends an MLflow run.

    Args:
        title (str): The title of the MLflow run.

    Yields:
        None

    Prints:
        The title of the MLflow run.

    Example:
        >>> with mlflow_run("MLflow experiment"):
        ...     # code block
        MLflow experiment - run started
        ...
        MLflow experiment - run ended
    """
    try:
        mlflow.start_run()
        print(f"{title} - run started")
        yield
    finally:
        mlflow.end_run()
        print(f"{title} - run ended")

# Define the custom scoring function based on the cost
def custom_cost_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cost of the predicted labels compared to the true labels.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        float: The total cost calculated based on the false positive and false negative costs.
    """
    false_positive_cost = 1
    false_negative_cost = 10

    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    fn = cm[1, 0]

    total_cost = fp * false_positive_cost + fn * false_negative_cost
    return total_cost

class CustomThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier: object, threshold: float = 0.5) -> None:
        """
        Initializes a new instance of the class.

        Args:
            base_classifier: The base classifier object.
            threshold: The threshold value.
        """
        self.base_classifier = base_classifier
        # self.base_classifier = base_classifier(**base_params) if base_params else base_classifier
        self.threshold = threshold

    def fit(self, X: List, y: List) -> Type['CustomThresholdClassifier']:
        """Fits the classifier to the training data.

        Args:
            X (List): The input features.
            y (List): The target values.

        Returns:
            Classifier: The fitted classifier.
        """
        self.base_classifier.fit(X, y)
        return self

    def predict_proba(self, X: List[Any]) -> List[List[float]]:
        """
        Predicts the probabilities of the target classes for the given input data.

        Args:
            X (List[Any]): The input data to be classified.

        Returns:
            List[List[float]]: A list of lists representing the predicted probabilities of the target classes.
        """
        return self.base_classifier.predict_proba(X)

    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> Union[np.ndarray, List[int]]:
        """
        Custom prediction based on the adjusted threshold.

        Parameters:
            X: The input data for prediction. It can be either a numpy array or a list of lists of floats.

        Returns:
            The predicted values based on the adjusted threshold. It can be either a numpy array or a list of integers.
        """
        return (self.base_classifier.predict_proba(X)[:, 1] > self.threshold).astype(int)
    
    def classes_(self):
        """
        Returns the class labels.
        """
        return np.array([0,1])

def balance_classes(X: pd.DataFrame, y: pd.Series, method: Literal['smote', 'randomundersampler']='smote')-> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance classes in the dataset using SMOTE or RandomUnderSampler.

    Args:
        X: Features. Accepts either a pandas DataFrame or a numpy array.
        y: Target variable. Accepts either a pandas Series or a numpy array.
        method: Method to use for balancing. Options: 'smote' or 'randomundersampler'.
            Defaults to 'smote'.

    Returns:
        Balanced feature set and target variable. Returns a tuple containing the balanced feature set and target variable,
        which can be either a pandas DataFrame or a numpy array depending on the input types.
    """
    sampler_dict = {'smote': SMOTE(sampling_strategy='auto', random_state=42),
                    'randomundersampler': RandomUnderSampler(sampling_strategy='auto', random_state=42)}
    sampler = sampler_dict.get(method.lower())
    
    if sampler is None:
        raise ValueError("Invalid method. Choose 'smote' or 'randomundersampler'.")
    
    pipeline = Pipeline([('sampler', sampler)])
    return pipeline.fit_resample(X, y)

def generate_param_grid(model_name: Literal['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier'], search_method: Literal['grid', 'random']) -> dict:
    """
    Generate the parameter grid for the specified model.

    Args:
        model_name: The machine learning model name being tuned.
        search_method: The search method to use ('grid' or 'random').
    Returns:
        param_grid: The parameter grid for the specified model.
    """

    if model_name == 'LogisticRegression':
        param_grid = {
            'base_classifier__C': [0.1, 1, 10] if search_method == 'grid' else uniform(0.1, 9.9),
            'base_classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'base_classifier__max_iter': [100, 500, 1000] if search_method == 'grid' else sp_randint(100, 900),
            # 'base_classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            # 'base_classifier__multi_class': ['ovr', 'multinomial', 'auto'],
            'base_classifier__fit_intercept': [True, False],
            'base_classifier__class_weight': [None, 'balanced'],
            'threshold': [0.3, 0.4, 0.5, 0.6, 0.7] if search_method == 'grid' else uniform(0.3, 0.4)
        }
    elif model_name == 'RandomForestClassifier':
        param_grid = {
            'base_classifier__n_estimators': [100, 200, 300] if search_method == 'grid' else sp_randint(100, 300),
            'base_classifier__max_depth': [None, 5, 10, 20] if search_method == 'grid' else sp_randint(5, 20),
            'base_classifier__min_samples_split': [2, 5, 10] if search_method == 'grid' else sp_randint(2, 10),
            'base_classifier__min_samples_leaf': [1, 2, 4] if search_method == 'grid' else sp_randint(1, 4),
            # 'base_classifier__max_features': ['auto', 'sqrt', 'log2'],
            'base_classifier__bootstrap': [True, False],
            # 'base_classifier__criterion': ['gini', 'entropy'],
            'threshold': [0.3, 0.4, 0.5, 0.6, 0.7] if search_method == 'grid' else uniform(0.3, 0.4)
        }
    elif model_name == 'XGBClassifier':
        param_grid = {
            'base_classifier__n_estimators': [100, 200, 300] if search_method == 'grid' else sp_randint(100, 300),
            'base_classifier__max_depth': [3, 4, 5] if search_method == 'grid' else sp_randint(3, 5),
            'base_classifier__learning_rate': [0.1, 0.2, 0.3] if search_method == 'grid' else uniform(0.1, 0.2),
            'base_classifier__subsample': [0.8, 1.0] if search_method == 'grid' else uniform(0.8, 0.2),
            'base_classifier__colsample_bytree': [0.8, 1.0] if search_method == 'grid' else uniform(0.8, 0.2),
            'base_classifier__gamma': [0, 0.1, 0.2] if search_method == 'grid' else uniform(0, 0.2),
            'base_classifier__reg_alpha': [0, 0.1, 0.2] if search_method == 'grid' else uniform(0, 0.2),
            'base_classifier__reg_lambda': [1, 1.1, 1.2] if search_method == 'grid' else uniform(1, 0.2),
            'threshold': [0.3, 0.4, 0.5, 0.6, 0.7] if search_method == 'grid' else uniform(0.3, 0.4)
        }
    elif model_name == 'LGBMClassifier':
        param_grid = {
            'base_classifier__n_estimators': [100, 200, 300] if search_method == 'grid' else sp_randint(100, 300),
            'base_classifier__max_depth': [3, 4, 5] if search_method == 'grid' else sp_randint(3, 5),
            'base_classifier__learning_rate': [0.1, 0.2, 0.3] if search_method == 'grid' else uniform(0.1, 0.2),
            'base_classifier__subsample': [0.8, 1.0] if search_method == 'grid' else uniform(0.8, 0.2),
            'base_classifier__colsample_bytree': [0.8, 1.0] if search_method == 'grid' else uniform(0.8, 0.2),
            'base_classifier__gamma': [0, 0.1, 0.2] if search_method == 'grid' else uniform(0, 0.2),
            'base_classifier__reg_alpha': [0, 0.1, 0.2] if search_method == 'grid' else uniform(0, 0.2),
            'base_classifier__reg_lambda': [1, 1.1, 1.2] if search_method == 'grid' else uniform(1, 0.2),
            'threshold': [0.3, 0.4, 0.5, 0.6, 0.7] if search_method == 'grid' else uniform(0.3, 0.4)
        }

    return param_grid

def tune_model(model: BaseEstimator, X_train: Union[list, np.array], y_train: Union[list, np.array], X_test: Union[list, np.array], y_test: Union[list, np.array], search_method: Literal['grid', 'random'], balance_method: Literal['smote', 'randomundersampler'], debug: bool = False):
    """
    Tune the specified model using grid search or random search based on the specified search method.
    
    Parameters:
        - model: The machine learning model to be tuned.
        - X: The input features for training the model.
        - y: The target variable for training the model.
        - param_grid: The parameter grid defining the hyperparameter search space.
        - search_method: The search method to be used for hyperparameter tuning. Choose 'grid' or 'random'.
        
    Returns:
        None
        
    Raises:
        ValueError: If an invalid search method is specified.
    
    This function tunes the specified model using either grid search or random search based on the specified search method. 
    It performs a hyperparameter search over the provided parameter grid and fits the model to the training data.
    The function calculates and logs various evaluation metrics to MLflow, including AUC and accuracy.
    The best model found during the hyperparameter search is also logged to MLflow.
    """

    model_name = model.__class__.__name__
    run_type = 'debug' if debug else 'tuning'

    if balance_method == 'smote':
        X_train_resampled, y_train_resampled = balance_classes(X_train, y_train, method=balance_method)
    elif balance_method == 'randomundersampler':
        X_train_resampled, y_train_resampled = balance_classes(X_train, y_train, method=balance_method)
    else:
        raise ValueError("Invalid balance method. Choose 'smote' or 'randomundersampler'.")
    
    param_grid = generate_param_grid(model_name, search_method)
    custom_threshold_model = CustomThresholdClassifier(model)

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_{search_method}_search_{balance_method}_balancing_{run_type}_run"):
    
        try:
            # Define the custom scorer using the custom cost function
            custom_scorer = make_scorer(custom_cost_function, greater_is_better=False)
            
            # Perform grid search or random search based on specified search method
            if search_method == "grid":
                search = GridSearchCV(custom_threshold_model, param_grid, scoring=custom_scorer, cv=5)
            elif search_method == "random":
                search = RandomizedSearchCV(custom_threshold_model, param_grid, scoring=custom_scorer, cv=5, n_iter=20)
            else:
                raise ValueError("Invalid search method. Choose 'grid' or 'random'.")
            
            search.fit(X_train_resampled, y_train_resampled)

            # # Reconstruct and fit the best model
            best_params = search.get_params()

            # Extract only the parameters related to your base classifier in CustomThresholdClassifier
            base_classifier_params = {key.replace('base_estimator__', ''): value for key, value in best_params.items() if key.replace('estimator__', '') in custom_threshold_model.get_params().keys()}
            best_model = CustomThresholdClassifier(model)
            best_model.set_params(**base_classifier_params)
            best_model.fit(X_train_resampled, y_train_resampled)
            
            # # Calculate and log AUC and accuracy
            # y_pred_proba = best_model.predict_proba(X)[:, 1]  # Assuming binary classification
            # y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
            # y_pred = best_model.predict(X)
            # y_pred_test = best_model.predict(X_test)

            y_pred_proba = search.predict_proba(X_train_resampled)[:, 1]  # Assuming binary classification
            y_pred_proba_test = search.predict_proba(X_test)[:, 1]
            y_pred = search.predict(X_train_resampled)
            y_pred_test = search.predict(X_test)

            auc_train = roc_auc_score(y_train_resampled, y_pred_proba)
            accuracy_train = accuracy_score(y_train_resampled, y_pred)
            auc_test = roc_auc_score(y_test, y_pred_proba_test)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            
            # Log parameters, metrics to MLflow
            mlflow.log_params(best_params)  # Log hyperparameters
            # mlflow.log_metric("best_score", search.best_score_)  # Log the best score
            metrics = {
                "auc_train": auc_train,
                "accuracy_train": accuracy_train,
                "auc_test": auc_test,
                "accuracy_test": accuracy_test
            }
            mlflow.log_metrics(metrics)  # Log the metrics

            # Log best model
            mlflow.sklearn.log_model(best_model, "best_model")
        finally:
            # End MLflow run
            mlflow.end_run()

def main(model: Union[LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier] = LogisticRegression(),
         search_method: Literal['grid', 'random'] = 'random',
         balance_method: Literal['smote', 'randomundersampler'] = 'randomundersampler',
         debug: bool = True):
    
    with timer("Data loading"):
        train_data = pd.read_csv('./train_df_test.csv') if debug else pd.read_csv('./train_df.csv')
        X_train = train_data.drop(columns=['SK_ID_CURR', 'index', 'TARGET'])
        y_train = train_data['TARGET']
        test_data = pd.read_csv('./test_df_test.csv') if debug else pd.read_csv('./test_df.csv')
        X_test = test_data.drop(columns=['SK_ID_CURR', 'index', 'TARGET'])
        y_test = test_data['SK_ID_CURR']

    with timer("Tuning model"):
        tune_model(model, X_train, y_train, X_test, y_test, search_method, balance_method, debug)

main(model=LogisticRegression(n_jobs=-1), search_method='random', balance_method='randomundersampler', debug=True)
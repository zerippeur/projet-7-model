from typing import List, Tuple, Literal, Optional, Union, Any, Dict, Callable, Iterable, TypeVar, Generic, Type, cast, no_type_check

import numpy as np
import pandas as pd
from random import randint
from scipy.stats import uniform, loguniform, randint as sp_randint
from skopt.space import Real, Integer, Categorical

import time
from contextlib import contextmanager

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, make_scorer, fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from skopt import BayesSearchCV
from skopt.plots import plot_convergence

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import matplotlib.pyplot as plt
import seaborn as sns

from lime import lime_tabular
import shap

import io
import os

from mlflow.store.artifact.utils.models import get_model_name_and_version

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

def get_folder_latest_version(path, folder_prefix):
    # Get a list of directories in the specified path
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Filter directories with the specified prefix
    matching_dirs = [d for d in dirs if d.startswith(folder_prefix)]

    # If matching directories exist, find the latest version
    if matching_dirs:
        latest_version = max([int(d.split('-')[-1]) for d in matching_dirs])
        return latest_version
    else:
        return None  # No matching directories found

def filter_params(params, param_grid):
    return {key: value for key, value in param_grid.items() if key in params}

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

# Define the fbeta_score scorer
def custom_fbeta_score(y_true, y_pred, beta: float=np.sqrt(10)):
    """
    Calculates the custom f-beta score.

    Parameters:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
        beta (float, optional): The beta value for the f-beta score calculation. Default is np.sqrt(10).

    Returns:
        float: The custom f-beta score.
    """
    return fbeta_score(y_true, y_pred, beta)  # Change beta as needed

def threshold_cut(y_hat: np.ndarray, threshold=0.5):
    return y_hat >= threshold

def fbeta_curve(y_true: np.ndarray, y_hat: np.ndarray, beta: float=np.sqrt(10), show_classification: bool=True) -> Tuple[np.ndarray, List[np.ndarray], pd.DataFrame]:
    """
    Calculate the F-beta curve.

    Args:
        y_true (np.ndarray): The true labels.
        y_hat (np.ndarray): The predicted labels or probabilities.
        beta (float): Beta value for F-beta score calculation.
        show_classification (bool): Flag to include classification metrics.

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]: A tuple containing the threshold values, the corresponding F-beta scores, and the classification metrics if specified.
    """
    thresholds = np.linspace(0, 1, num=100)  # Adjust the number of thresholds as needed
    fbeta_scores = [fbeta_score(y_true, threshold_cut(y_hat, threshold=t), beta=beta) for t in thresholds]

    if show_classification:
        precision = [precision_score(y_true, threshold_cut(y_hat, threshold=t), zero_division=1) for t in thresholds]
        recall = [recall_score(y_true, threshold_cut(y_hat, threshold=t), zero_division=1) for t in thresholds]
        cost = np.array([custom_cost_function(y_true, threshold_cut(y_hat, threshold=t)) for t in thresholds])
        cost = cost/max(cost)
        accuracy = [accuracy_score(y_true, threshold_cut(y_hat, threshold=t)) for t in thresholds]

        scores_df = pd.DataFrame({
            'threshold': thresholds,
            'fbeta_score': fbeta_scores,
            'precision': precision,
            'recall': recall,
            'cost': cost,
            'accuracy': accuracy
        })

        return thresholds, fbeta_scores, scores_df

    return thresholds, fbeta_scores, None

def plot_f_beta_curve(y_true: np.ndarray, y_hat: np.ndarray, beta: float = np.sqrt(10), model_description: str = '', show_classification: bool=True, dpi: int = 300) -> np.array:
    """
    Plot the F-beta curve.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        image_array: The image array of the plot.
    """
    f_beta_curve = fbeta_curve(y_true, y_hat, beta, show_classification=show_classification)
    plt.figure(figsize=(10, 10))
    plt.plot(f_beta_curve[0], f_beta_curve[1], color='Gold')

    colors = ['Gold', 'Darkturquoise', 'Paleturquoise', 'Coral', 'Midnightblue']
    descriptors = ['f_beta', 'recall', 'precision', 'cost', 'accuracy']
    linestyles = ['-', '-', '-', '-', '-']
    classification_df = pd.DataFrame()
    classification_df['colors'] = colors
    classification_df['descriptors'] = descriptors
    classification_df['linestyles'] = linestyles

    if show_classification:
        for string in descriptors[1:]:
            plt.plot(
                f_beta_curve[0], f_beta_curve[2][string],
                color=classification_df.loc[classification_df.descriptors == string, 'colors'].values[0],
                linestyle=classification_df.loc[classification_df.descriptors == string, 'linestyles'].values[0]
            )
        
        # Find the index where the F-beta score is highest
        max_fbeta_index = np.argmax(f_beta_curve[1])
        max_fbeta_threshold = f_beta_curve[0][max_fbeta_index]
        max_fbeta_score = f_beta_curve[1][max_fbeta_index]
        max_precision = f_beta_curve[2].loc[max_fbeta_index, 'precision']
        max_recall = f_beta_curve[2].loc[max_fbeta_index, 'recall']
        max_cost = f_beta_curve[2].loc[max_fbeta_index, 'cost']
        max_accuracy = f_beta_curve[2].loc[max_fbeta_index, 'accuracy']

        # Plot marker at the point of highest F-beta score
        plt.plot(max_fbeta_threshold, max_fbeta_score, marker='o', markersize=8, color='red')

        # Draw lines corresponding to the maximum F-beta score
        plt.axvline(x=max_fbeta_threshold, linestyle='--', color='gray', ymax=max_fbeta_score, linewidth=0.8)
        plt.axhline(y=max_fbeta_score, linestyle='--', color='gray', xmax=max_fbeta_threshold, linewidth=0.8)

        # Annotate the point with threshold and F-beta score
        plt.annotate(
            f'Threshold: {max_fbeta_threshold:.3f}\n'
            f'F_beta:    {max_fbeta_score:.3f}\n'
            f'Precision: {max_precision:.3f}\n'
            f'Recall:    {max_recall:.3f}\n'
            f'Accuracy:  {max_accuracy:.3f}\n'
            f'Cost:      {max_cost:.3f}\n',
            xy=(max_fbeta_threshold, max_fbeta_score),
            xytext=(max_fbeta_threshold + 0.1, max_fbeta_score - 0.3),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            fontfamily='Courier New', fontsize=10, fontweight='bold'
        )

    plt.ylabel('F_beta_score')
    plt.title(f'F_beta Curve\nModel: {model_description}')
    plt.xlabel('Threshold')
    plt.legend(descriptors, loc='upper right')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi)
    buffer.seek(0)
    image_array = plt.imread(buffer)

    return image_array

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
    X_resampled, y_resampled = pipeline.named_steps['sampler'].fit_resample(X, y)
    # return pipeline.fit_resample(X, y)
    return X_resampled, y_resampled

def generate_param_grid(model_name: Literal['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier'], search_method: Literal['grid', 'random', 'bayes']) -> dict:
    """
    Generate the parameter grid for the specified model.

    Args:
        model_name: The machine learning model name being tuned.
        search_method: The search method to use ('grid', 'random', or 'bayes').
    Returns:
        param_grid: The parameter grid for the specified model.
    """
    if model_name == 'LogisticRegression':
        if search_method == 'grid':
            param_grid = {
                'C': [0.1, 1, 10],
                'class_weight': [None, 'balanced'],
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            }
        elif search_method == 'random':
            param_grid = {
                'C': uniform(0.1, 9.9),
                'class_weight': [None, 'balanced'],
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
            }
        elif search_method == 'bayes':
            param_grid = {
                'C': Real(0.1, 10.0),
                'class_weight': Categorical([None, 'balanced']),
                'fit_intercept': Categorical([True, False]),
                'solver': Categorical(['newton-cg', 'lbfgs', 'sag', 'saga'])
            }
        else:
            raise ValueError("Invalid search method")
    elif model_name in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
        if search_method == 'grid':
            param_grid = {
                'bootstrap': [True, False],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'learning_rate': [0.1, 0.2, 0.3],
                'max_depth': [None, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [100, 200, 300],
                'reg_alpha': [0, 0.1, 0.2],
                'reg_lambda': [1, 1.1, 1.2],
                'subsample': [0.7, 1.0]
            }
        elif search_method == 'random':
            param_grid = {
                'bootstrap': [True, False],
                'colsample_bytree': uniform(0.8, 0.2),
                'gamma': uniform(0, 0.2),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': sp_randint(3, 21),
                'min_samples_leaf': sp_randint(1, 5),
                'min_samples_split': sp_randint(2, 11),
                'n_estimators': sp_randint(100, 301),
                'reg_alpha': uniform(0, 0.2),
                'reg_lambda': uniform(1, 0.2),
                'subsample': uniform(0.7, 0.2)
            }
        elif search_method == 'bayes':
            param_grid = {
                'bootstrap': Categorical([True, False]),
                'colsample_bytree': Real(0.1, 1),
                'gamma': Real(0, 0.5),
                'learning_rate': Real(0.0001, 0.2),
                'max_depth': Integer(2, 8),
                'max_features': Categorical([1, 5, 10, 15, 20, 30]),
                'max_leaf_nodes': Categorical([5, 8, 10, 12, 15, 20, 30, 40, 50, 60]),
                'min_child_samples': Integer(1, 20),
                'min_samples_leaf': Integer(1, 20),
                'min_samples_split': Integer(2, 20),
                'n_estimators': Integer(100, 300),
                'num_leaves': Integer(10, 30),
                'reg_alpha': Real(0, 2),
                'reg_lambda': Real(1, 30),
                'subsample': Real(0.01, 1),
            }
        else:
            raise ValueError("Invalid search method")
    else:
        raise ValueError("Invalid model name")

    return param_grid

def tune_model(model: BaseEstimator, X_train: Union[list, np.ndarray], y_train: Union[list, np.array], X_test: Union[list, np.ndarray], y_test: Union[list, np.array], search_method: Literal['grid', 'random', 'bayes'], balance_method: Literal['smote', 'randomundersampler'], debug: bool = False):
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
    filtered_grid = filter_params(model.get_params(), param_grid)

    # Start MLflow run
    path = f"{model_name}_model__{search_method}_search__{balance_method}_balancing__{run_type}_run"

    # Set tracking server uri for logging (start mlflow server first: mlflow server --host 127.0.0.1 --port 8080 in git bash)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create new MLflow experiment
    mlflow.set_experiment(path)

    with mlflow.start_run(run_name=path):
    
        try:            
            # Perform grid search or random search based on specified search method
            if search_method == "grid":
                search = GridSearchCV(model, filtered_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
            elif search_method == "random":
                search = RandomizedSearchCV(model, filtered_grid, cv=5, n_iter=10 if debug else 100, scoring='roc_auc', verbose=1, n_jobs=-1)
            elif search_method == "bayes":
                search = BayesSearchCV(model, filtered_grid, cv=5, n_iter=10 if debug else 200, scoring='roc_auc', verbose=1, n_jobs=-1)
            else:
                raise ValueError("Invalid search method. Choose 'grid', 'random' or 'bayes.")
            
            search.fit(X_train_resampled, y_train_resampled)

            # # Reconstruct and fit the best model
            best_params = search.best_params_
            best_model = search.best_estimator_      

            results = search.cv_results_
            best_index = search.best_index_
            auc_cv = results['mean_test_score'][best_index]

            y_hat = search.predict_proba(X_train_resampled)[:, 1]  # Assuming binary classification
            y_hat_test = search.predict_proba(X_test)[:, 1]
            y_pred = search.predict(X_train_resampled)
            y_pred_test = search.predict(X_test)

            # Infer the model signature
            signature = infer_signature(X_train_resampled, y_hat)

            auc_train = roc_auc_score(y_train_resampled, y_hat)
            accuracy_train = accuracy_score(y_train_resampled, y_pred)
            auc_test = roc_auc_score(y_test, y_hat_test)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            
            # Log parameters, metrics to MLflow
            mlflow.log_params(best_params)  # Log hyperparameters
            # mlflow.log_metric("best_score", search.best_score_)  # Log the best score
            metrics = {
                "auc_train": auc_train,
                "auc_test": auc_test,
                "auc_test_cv": auc_cv,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
            }
            mlflow.log_metrics(metrics)  # Log the metrics

            # Log best model
            mlflow.sklearn.log_model(best_model, artifact_path=path, signature=signature, input_example=X_train_resampled, registered_model_name=model_name)

            model_path = f'mlruns\models\{model_name}'
            # get the latest version of the model
            latest_version = get_folder_latest_version(model_path, folder_prefix='version')
            mlflow.log_image(plot_f_beta_curve(y_train_resampled, y_hat, model_description=model_name, show_classification=True, dpi=300), artifact_file=f"{path}_f_beta_version_{latest_version}.png")

        finally:
            # End MLflow run
            mlflow.end_run()

def main(model: Union[LogisticRegression, RandomForestClassifier, XGBClassifier, LGBMClassifier] = LogisticRegression(),
         search_method: Literal['grid', 'random', 'bayes'] = 'bayes',
         balance_method: Literal['smote', 'randomundersampler'] = 'randomundersampler',
         debug: bool = True):
    
    with timer("Data loading"):
        train_data = pd.read_csv('features/train_df_debug.csv') if debug else pd.read_csv('features/train_df.csv')
        X_train = train_data.drop(columns=['SK_ID_CURR', 'index', 'TARGET'])
        y_train = train_data['TARGET']
        test_data = pd.read_csv('features/test_df_debug.csv') if debug else pd.read_csv('features/test_df.csv')
        X_test = test_data.drop(columns=['SK_ID_CURR', 'index', 'TARGET'])
        y_test = test_data['TARGET']

    with timer("Tuning model"):
        tune_model(model, X_train, y_train, X_test, y_test, search_method, balance_method, debug)

parameters_sets = [
    {
        'model': LogisticRegression(n_jobs=-1, random_state=42),
        'search_method': 'bayes',
        'balance_method': 'randomundersampler'
        'debug': True
    },
    {
        'model': RandomForestClassifier(n_jobs=-1, random_state=42),
        'search_method': 'bayes',
        'balance_method': 'randomundersampler'
        'debug': True
    },
    {
        'model': XGBClassifier(n_jobs=-1, random_state=42),
        'search_method': 'bayes',
        'balance_method': 'randomundersampler'
        'debug': True
    },
    {
        'model': LGBMClassifier(n_jobs=-1, random_state=42),
        'search_method': 'bayes',
        'balance_method': 'randomundersampler'
        'debug': True
    }
]
for params in parameters_sets:
    main(**params)
main(model=RandomForestClassifier(n_jobs=-1, random_state=42), search_method='bayes', balance_method='randomundersampler', debug=True)
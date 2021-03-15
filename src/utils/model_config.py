# Import models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Set params range
DEFAULT_PARAMS = {"random_state": 100,
                  "n_jobs": -1}
CONFIGS = {
    'lightgbm': {
        'params_range': [
            {'name': 'num_leaves', 'type': 'range', 'bounds': [10, 40]},
            {'name': 'max_depth', 'type': 'range', 'bounds': [3, 31]},
            {'name': 'learning_rate', 'type': 'range', 'bounds': [0.001, 0.3]},
            {'name': 'n_estimators', 'type': 'range', 'bounds': [1, 4]},
            {'name': 'reg_alpha', 'type': 'range', 'bounds': [0.001, 10.0]},
            {'name': 'reg_lambda', 'type': 'range', 'bounds': [0.001, 10.0]},
            {'name': 'class_weight', 'type': 'choice', 'values': ["balanced", "balanced_subsample"]}           
        ],
        'model_instance': LGBMClassifier(random_state=100)
    },
    'catboost': {
        'params_range': [
            {'name': 'depth', 'type': 'choice', 'values': [
                3, 1, 2, 6, 4, 5, 7, 8, 9, 10]},
            {'name': 'iterations', 'type': 'choice',
                'values': [250, 100, 500, 1000]},
            {'name': 'border_count', 'type': 'choice',
                'values': [32, 5, 10, 20, 50, 100, 200]},
            {'name': 'learning_rate', 'type': 'choice', 'values': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]}],
        'model_instance': CatBoostClassifier(random_state=100)
    },
    'xgboost': {
        'params_range': [
            {'name': 'learning_rate', 'type': 'choice',
                'values': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]},
            {'name': 'max_depth', 'type': 'choice',
                'values': [3, 4, 5, 6, 8, 10, 12, 15]},
            {'name': 'min_child_weight',
                'type': 'choice', 'values': [1, 3, 5, 7]},
            {'name': 'gamma', 'type': 'choice',
                'values': [0.0, 0.1, 0.2, 0.3, 0.4]},
            {'name': 'colsample_bytree', 'type': 'choice',
                'values': [0.3, 0.4, 0.5, 0.7]},
        ],
        'model_instance': XGBClassifier(random_state=100)
    },
    'rf': {
        'params_range': [
            {'name': 'n_estimators', 'type': 'range', 'bounds': [10, 100]},
            {'name': 'max_depth', 'type': 'choice',
                'values': [3, 4, 5, 6, 8, 10, 12, 15]},
            {'name': 'bootstrap', 'type': 'choice', 'values': [True, False]},
            {'name': 'criterion', 'type': 'choice',
                'values': ["gini", "entropy"]},
            {'name': 'max_features', 'type': 'choice',
                'values': ["auto", "sqrt", "log2"]},
            {'name': 'class_weight', 'type': 'choice',
                'values': ["balanced", "balanced_subsample"]},

        ],
        'model_instance': RandomForestClassifier(**DEFAULT_PARAMS)
    },
    'log_reg': {
        'params_range': [
            {'name': 'penalty', 'type': 'choice', 'values': [
                'l1', 'l2', 'elasticnet', 'none']},
            {'name': 'C', 'type': 'range', 'bounds': [1.0, 25.0]},
            {'name': 'solver', 'type': 'choice', 'values': [
                'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
            {'name': 'max_iter', 'type': 'range', 'bounds': [20, 120]},
            {'name': 'class_weight', 'type': 'choice', 'values': ['balanced', 'None']},            
        ],
        'model_instance': LogisticRegression(**DEFAULT_PARAMS)
    },
    'mlp': {
        'params_range': [
            {'name': 'hidden_layer_sizes',
                'type': 'range', 'bounds': [20, 100]},
            {'name': 'activation', 'type': 'choice', 'values': [
                'identity', 'logistic', 'tanh', 'relu']},
            {'name': 'solver', 'type': 'choice', 'values': ['sgd', 'adam']},
            # {'name': 'alpha', 'type': 'range', 'bounds': [0.0001, 1.0]},
            {'name': 'learning_rate', 'type': 'choice',
                'values': ['constant', 'invscaling', 'adaptive']},
        ],
        'model_instance': MLPClassifier(random_state=100)
    }

}

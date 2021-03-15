import streamlit as st
from sklearn.model_selection import StratifiedKFold

from .optimizer import ModelBuilder
from .utils.model_config import CONFIGS


def cv_score(clf, params, X, y, n_folds=10):
    kfold = StratifiedKFold(n_folds)
    scores = []
    for itrain, ival in kfold.split(X, y):
        partitions = {}
        partitions['train'] = X.iloc[itrain], y.iloc[itrain]
        partitions['val'] = X.iloc[ival], y.iloc[ival]
        eval_method = ModelBuilder(partitions).build(clf)
        scores.append(eval_method(params))
    return scores


def compute_scores(training_history):
    optmizer_results = training_history['all_results']
    x_train, y_train = training_history['partitions']['train']
    scores = {}
    for clf_name, config in optmizer_results.items():
        print(f'RUNNING for {clf_name}')
        estimator = CONFIGS[clf_name]["model_instance"]
        score = cv_score(estimator, config['best_params'], x_train, y_train)
        scores[clf_name] = score
    return scores

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.optimizer import Optimizer
from src.utils.data_handler import fetch_credit_data, save_model

from .db.database import RedisDB
from .evaluation import compute_scores


def process_data(credit_data):
    logging.info("Processing the data")
    num_features = ['months_duration', 'credit_amount', 'install_rate', 'present_residence',
                    'age', 'n_bank_credits', 'dependents']
    cat_features = ['checking_account', 'credit_history', 'credit_purpose', 'savings_status',
                    'present_employ', 'personal_status', 'other_debtors', 'property', 'installment_plans',
                    'housing_status', 'job', 'has_phone', 'is_foreign_worker']

    credit_data[num_features] = credit_data[num_features].astype(float)
    credit_data[cat_features] = credit_data[cat_features].astype('category')

    X, y = credit_data.drop('label', axis=1), credit_data['label']
    # credit_data.to_csv('bin/data.csv')
    return X, y


def train_model():
    data = fetch_credit_data()
    X, y = process_data(data)

    # Split into train and validation
    logging.info('Training process started')
    partitions = {}
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    partitions["train"] = x_train, y_train
    partitions["val"] = x_test, y_test

    # Optimize models
    logging.info('Optmising models')
    opt = Optimizer(partitions)
    best_model, model_instance = opt.run()

    response = {'clf_name': best_model,
                'model_instance': model_instance,
                'all_results': opt.results,
                'partitions': partitions}
    response.update({'cv-scores': compute_scores(response)})

    # Save the results from this trial
    save_model(response)
    return response

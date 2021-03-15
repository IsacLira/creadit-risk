import logging
import pickle

import pandas as pd
from datapackage import Package

from ..db.database import RedisDB

FEATURE_NAMES = [
    "checking_account", "months_duration", "credit_history", "credit_purpose",
    "credit_amount", "savings_status", "present_employ", "install_rate", "personal_status",
    "other_debtors", "present_residence", "property", "age", "installment_plans", "housing_status",
    "n_bank_credits", "job", "dependents", "has_phone", "is_foreign_worker", "label"
]
MODEL_KEY = 'best_model1'
db = RedisDB()


def save_model(model, key=MODEL_KEY):
    try:
        db.set_value(key, pickle.dumps(model))
    except Exception as e:
        logging.exception(e)
        return False
    return True


def fetch_model(model_key=MODEL_KEY):
    try:
        bin_model = db.get_value(model_key)
        return pickle.loads(bin_model)
    except:
        with open('bin/model_start.pkl', 'rb') as f:
            model_start = pickle.load(f)
            save_model(model_start)
        return model_start


def fetch_credit_data(data_key='credict_data'):
    data = db.get_value(data_key)
    if not data:
        logging.info("Fetching credit data..")
        package = Package(
            'https://datahub.io/machine-learning/credit-g/datapackage.json')

        for resource in package.resources:
            if resource.descriptor['datahub']['type'] == 'derived/csv':
                data = resource.read()
        db.set_value(data_key, pickle.dumps(data))
    else:
        data = pickle.loads(data)

    credit_data = pd.DataFrame(data, columns=FEATURE_NAMES)
    return credit_data

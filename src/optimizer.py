import logging
from copy import copy

from ax import optimize
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils.model_config import CONFIGS


class ModelBuilder:
    def __init__(self, partitions):
        self.x_train, self.y_train = partitions['train']
        self.x_val, self.y_val = partitions['val']

        self.cat_features = self.x_train.select_dtypes('category').columns
        self.num_features = self.x_train.select_dtypes('number').columns

        ct = ColumnTransformer([
            ('encode', OneHotEncoder(), self.cat_features),
            ('normalizer', StandardScaler(), self.num_features)]
        )
        self.steps = [('transformers', ct)]
        fitted_models = {}

    def fit_model(self, estimator, params):
        estimator.set_params(**params)
        if isinstance(estimator, CatBoostClassifier):
            pipe = estimator.fit(X=self.x_train,
                                 y=self.y_train,
                                 cat_features=self.cat_features)
        else:
            pipe = Pipeline(self.steps + [("estimator", estimator)])
            pipe.fit(X=self.x_train, y=self.y_train)

        return pipe

    def build(self, model, save_model=False):
        def model_eval(parameterization, verbose=False):

            estimator = copy(model)

            try:
                # Set params on model and create a pipeline
                pipe = self.fit_model(estimator, parameterization)

                # Compute accuracy on validation set
                val_pred = pipe.predict(self.x_val)
                acc_score = accuracy_score(self.y_val, val_pred)
                return acc_score
            except Exception as e:
                logging.error(e)
                print('ERROR', e)
                return 0.0

        return model_eval


class Optimizer:
    def __init__(self, partitions):
        self.partitions = partitions
        self.model_builder = ModelBuilder(self.partitions)

    def run(self, configs=CONFIGS):

        self.results = {}
        for clf_name, config in configs.items():
            logging.info(f'Optmising model {clf_name}')
            eval_method = self.model_builder.build(config["model_instance"])

            best_params, best_values, *_ = optimize(
                parameters=config["params_range"],
                evaluation_function=eval_method,
                minimize=False,
                total_trials=20,
                # random_seed=100
            )
            self.results[clf_name] = {
                'best_params': best_params,
                'best_score': best_values[0]['objective']}

        # Select the best model
        def get_score(r): return r[1]['best_score']
        best_model_d = sorted(self.results.items(), key=get_score)[-1]

        best_clf_name = best_model_d[0]
        best_clf = CONFIGS[best_clf_name]['model_instance']
        fitted_model = self.model_builder.fit_model(
            estimator=best_clf,
            params=best_model_d[1]['best_params']
        )
        return best_model_d, fitted_model

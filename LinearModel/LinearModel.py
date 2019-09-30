# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
np.random.seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class OHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__columns = None
        self.__missing = None
        self.__categories = None
        self.__lab_encoder = None
        self.__ohe_encoder = None

    def fit(self, X, y=None):
        feature, label = X.copy(deep=True), y.copy(deep=True)
        del X, y
        gc.collect()

        self.__columns = list()
        self.__missing = dict()
        self.__categories = dict()
        self.__lab_encoder = dict()

        for column in feature.columns:
            num_unique = feature[column].nunique()

            if num_unique == 1:
                continue
            else:

                self.__columns.append(column)

                if feature[column].isna().sum():
                    self.__missing[column] = "missing"
                    self.__categories[column] = feature[column].unique()
                else:
                    self.__missing[column] = feature[column].value_counts(ascending=True).index[0]
                    self.__categories[column] = feature[column].unique()

                encoder = LabelEncoder()
                encoder.fit(feature[column])
                feature[column] = encoder.transform(feature[column])
                self.__lab_encoder[column] = encoder

        feature = feature[self.__columns].copy(deep=True)

        self.__ohe_encoder = OneHotEncoder(categories="auto", sparse=True)  # drop="first" bad
        self.__ohe_encoder.fit(feature)

    def transform(self, X):
        feature = X.copy(deep=True)
        del X
        gc.collect()

        feature = feature[self.__columns].copy(deep=True)

        for column in feature.columns:
            feature[column] = feature[column].fillna(self.__missing[column])
            feature[column] = feature[column].apply(
                lambda element: element if element in self.__categories[column] else self.__missing[column])
            feature[column] = self.__lab_encoder[column].transform(feature[column])

        return self.__ohe_encoder.transform(feature)

    def fit_transform(self, X, y=None, **fit_params):
        feature, label = X.copy(deep=True), y.copy(deep=True)
        del X, y
        gc.collect()

        self.fit(feature, label)
        return self.transform(feature)


class ROHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__columns = None
        self.__missing = None
        self.__categories = None
        self.__lab_encoder = None
        self.__tar_encoder = None
        self.__ohe_encoder = None

    def fit(self, X, y=None):
        feature, label = X.copy(deep=True), y.copy(deep=True)
        del X, y
        gc.collect()

        self.__columns = list()
        self.__missing = dict()
        self.__categories = dict()
        self.__lab_encoder = dict()

        for column in feature.columns:
            num_unique = feature[column].nunique()

            if num_unique == 1:
                continue
            else:

                self.__columns.append(column)

                if feature[column].isna().sum():
                    self.__missing[column] = "missing"
                    self.__categories[column] = feature[column].unique()
                else:
                    self.__missing[column] = feature[column].value_counts(ascending=True).index[0]
                    self.__categories[column] = feature[column].unique()

                encoder = LabelEncoder()
                encoder.fit(feature[column])
                feature[column] = encoder.transform(feature[column])
                self.__lab_encoder[column] = encoder

        feature = feature[self.__columns].copy(deep=True)

        self.__tar_encoder = TargetEncoder()
        self.__tar_encoder.fit(feature.astype(str), label)

        self.__ohe_encoder = OneHotEncoder(categories="auto", sparse=True)  # drop="first" bad
        self.__ohe_encoder.fit(self.__tar_encoder.transform(feature.astype(str)))

    def transform(self, X):
        feature = X.copy(deep=True)
        del X
        gc.collect()

        feature = feature[self.__columns].copy(deep=True)

        for column in feature.columns:
            feature[column] = feature[column].fillna(self.__missing[column])
            feature[column] = feature[column].apply(
                lambda element: element if element in self.__categories[column] else self.__missing[column])
            feature[column] = self.__lab_encoder[column].transform(feature[column])

        return self.__ohe_encoder.transform(self.__tar_encoder.transform(feature.astype(str)))

    def fit_transform(self, X, y=None, **fit_params):
        feature, label = X.copy(deep=True), y.copy(deep=True)
        del X, y
        gc.collect()

        self.fit(feature, label)
        return self.transform(feature)


class LinearModel(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]  # test_index dataframe

        # model fit predict
        self.__model = None
        self.__folds = None
        self.__oof_preds = None
        self.__sub_preds = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 1:-1].copy(deep=True), self.__train.iloc[:, -1].copy(deep=True))
        self.__test_feature, self.__test_index = (
            self.__test.iloc[:, 1:].copy(deep=True), self.__test.iloc[:, [0]].copy(deep=True))
        del self.__train, self.__test
        gc.collect()

    def model_fit_predict(self):
        # optimize
        def linear_model_crossval(C):
            estimator = Pipeline([
                ("OHE", OHE()),
                ("CLF", LogisticRegression(
                    C=C, solver="lbfgs", max_iter=1000, random_state=7))
            ])

            cval = cross_val_score(
                estimator,
                self.__train_feature,
                self.__train_label,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            )

            return cval.mean()

        optimizer = BayesianOptimization(
            f=linear_model_crossval,
            pbounds={"C": (0.1, 0.15)},
            random_state=7,
            verbose=2
        )
        optimizer.maximize(init_points=10, n_iter=50)

        # stacking
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            trn_x, trn_y = self.__train_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
            val_x, _ = self.__train_feature.iloc[val_idx], self.__train_label.iloc[val_idx]

            self.__model = Pipeline([
                ("OHE", OHE()),
                ("CLF", LogisticRegression(
                    C=0.1036, solver="lbfgs", max_iter=1000, random_state=7))  # optimizer.max["params"]["C"]
            ])
            self.__model.fit(trn_x, trn_y)

            pred_val = self.__model.predict_proba(val_x)[:, 1]
            self.__oof_preds[val_idx] = logit(pred_val)

            del trn_x, trn_y, val_x, _
            gc.collect()

        # model fit predict
        self.__model = Pipeline([
            ("OHE", OHE()),
            ("CLF", LogisticRegression(
                C=0.1036, solver="lbfgs", max_iter=1000, random_state=7))  # optimizer.max["params"]["C"]
        ])
        self.__model.fit(self.__train_feature, self.__train_label)

        pred_test = self.__model.predict_proba(self.__test_feature)[:, 1]
        self.__sub_preds = logit(pred_test)
        self.__test_index["target"] = pred_test

    def data_write(self):
        print("Fold all prediction trn auc: %.5f" % (
            roc_auc_score(self.__train_label, expit(self.__oof_preds))))

        pd.Series(self.__oof_preds).to_frame("oof_raw_score").to_csv(
            os.path.join(self.__output_path, "oof_raw_score.csv"), index=False)
        pd.Series(self.__sub_preds).to_frame("sub_raw_score").to_csv(
            os.path.join(self.__output_path, "sub_raw_score.csv"), index=False)
        self.__test_index.to_csv(os.path.join(self.__output_path, "sample_submission_linear_model.csv"), index=False)


if __name__ == "__main__":
    lm = LinearModel(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    lm.data_read()
    lm.data_prepare()
    lm.model_fit_predict()
    lm.data_write()

# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from bayes_opt import BayesianOptimization
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
        feature, _ = X.copy(deep=True), y.copy(deep=True)
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


class LinearModelXGBLeaf(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        self.__train, self.__test = [None for _ in range(2)]
        self.__oof_leafs, self.__sub_leafs = [None for _ in range(2)]

        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]

        self.__model = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

        self.__oof_leafs = sparse.load_npz(os.path.join(self.__input_path, "oof_leafs.npz"))
        self.__sub_leafs = sparse.load_npz(os.path.join(self.__input_path, "sub_leafs.npz"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 1:-1].copy(deep=True), self.__train.iloc[:, -1].copy(deep=True))
        self.__test_feature, self.__test_index = (
            self.__test.iloc[:, 1:].copy(deep=True), self.__test.iloc[:, [0]].copy(deep=True))
        del self.__train, self.__test

        self.__train_feature = self.__train_feature.drop(
            ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "day", "month"], axis=1)
        self.__test_feature = self.__test_feature.drop(
            ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "day", "month"], axis=1)

        encoder = OHE()
        encoder.fit(self.__train_feature, self.__train_label)
        self.__train_feature = encoder.transform(self.__train_feature)
        self.__test_feature = encoder.transform(self.__test_feature)

    def model_fit_predict(self):
        # optimize
        # def linear_model_crossval(C):
        #     estimator = LogisticRegression(
        #         C=C, solver="lbfgs", max_iter=10000, random_state=7)
        #
        #     cval = cross_val_score(
        #         estimator,
        #         sparse.hstack([self.__train_feature, self.__oof_leafs]),
        #         self.__train_label,
        #         scoring="roc_auc",
        #         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        #     )
        #
        #     return cval.mean()
        #
        # optimizer = BayesianOptimization(
        #     f=linear_model_crossval,
        #     pbounds={"C": (0.05, 0.15)},
        #     random_state=7,
        #     verbose=2
        # )
        # optimizer.maximize(init_points=25, n_iter=125)

        # model fit predict
        self.__model = LogisticRegression(
            C=0.1179, solver="lbfgs", max_iter=10000, random_state=7)  # C=optimizer.max["params"]["C"]
        self.__model.fit(sparse.hstack([self.__train_feature, self.__oof_leafs]), self.__train_label)
        self.__test_index["target"] = self.__model.predict_proba(
            sparse.hstack([self.__test_feature, self.__sub_leafs]))[:, 1]

    def data_write(self):
        self.__test_index.to_csv(
            os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    lmxgbleaf = LinearModelXGBLeaf(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    lmxgbleaf.data_read()
    lmxgbleaf.data_prepare()
    lmxgbleaf.model_fit_predict()
    lmxgbleaf.data_write()
# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
np.random.seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class XGBLeaf(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        self.__train, self.__test = [None for _ in range(2)]
        self.__oof_leafs, self.__sub_leafs = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]

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
        ord_encoder_columns = ["ord_3", "ord_4", "ord_5"]
        tar_encoder_columns = ["ord_0", "ord_1", "ord_2", "day", "month"]

        # optimize
        def gbm_model_crossval(learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda):
            estimator = Pipeline([
                ("ENCODER", ColumnTransformer([
                    ("ORD_ENCODER", OrdinalEncoder(categories="auto"), ord_encoder_columns),
                    ("TAR_ENCODER", TargetEncoder(cols=tar_encoder_columns), tar_encoder_columns),
                ], remainder="drop")),
                ("LGBMCLF", LGBMClassifier(
                    max_depth=1,
                    learning_rate=learning_rate,
                    n_estimators=np.int(np.round(n_estimators)),
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=7,
                    n_jobs=-1))
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
            f=gbm_model_crossval,
            pbounds={
                "learning_rate": (0.01, 0.1),
                "n_estimators": (100, 500),
                "subsample": (0.5, 1),
                "colsample_bytree": (0.5, 1),
                "reg_alpha": (0, 10),
                "reg_lambda": (0, 10)},
            random_state=7,
            verbose=2
        )
        optimizer.maximize(init_points=10, n_iter=50)

        encoder = ColumnTransformer([
            ("ORD_ENCODER", OrdinalEncoder(), ord_encoder_columns),
            ("TAR_ENCODER", TargetEncoder(cols=tar_encoder_columns), tar_encoder_columns),
        ], remainder="drop")
        lgbmclf = LGBMClassifier(
            max_depth=1,
            learning_rate=optimizer.max["params"]["learning_rate"],
            n_estimators=np.int(np.round(optimizer.max["params"]["n_estimators"])),
            subsample=optimizer.max["params"]["subsample"],
            colsample_bytree=optimizer.max["params"]["colsample_bytree"],
            reg_alpha=optimizer.max["params"]["reg_alpha"],
            reg_lambda=optimizer.max["params"]["reg_lambda"],
            random_state=7,
            n_jobs=-1)

        encoder.fit(self.__train_feature, self.__train_label)
        self.__train_feature = encoder.transform(self.__train_feature)
        self.__test_feature = encoder.transform(self.__test_feature)

        lgbmclf.fit(self.__train_feature, self.__train_label)
        self.__oof_leafs = lgbmclf.predict(self.__train_feature, pred_leaf=True)
        self.__sub_leafs = lgbmclf.predict(self.__test_feature, pred_leaf=True)

    def data_write(self):
        self.__oof_leafs = sparse.csr_matrix(self.__oof_leafs)
        self.__sub_leafs = sparse.csr_matrix(self.__sub_leafs)

        sparse.save_npz(os.path.join(self.__output_path, "oof_leafs.npz"), self.__oof_leafs)
        sparse.save_npz(os.path.join(self.__output_path, "sub_leafs.npz"), self.__sub_leafs)


if __name__ == "__main__":
    xgbleaf = XGBLeaf(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    xgbleaf.data_read()
    xgbleaf.data_prepare()
    xgbleaf.model_fit_predict()
    xgbleaf.data_write()

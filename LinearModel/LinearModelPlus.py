# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
from scipy.special import expit
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class LinearModelPlus(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]

        self.__oof_raw_score, self.__sub_raw_score = [None for _ in range(2)]
        self.__model = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

        self.__oof_raw_score = pd.read_csv(os.path.join(self.__input_path, "oof_raw_score.csv"))
        self.__sub_raw_score = pd.read_csv(os.path.join(self.__input_path, "sub_raw_score.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 1:-1].copy(deep=True), self.__train.iloc[:, -1].copy(deep=True))
        self.__test_feature, self.__test_index = (
            self.__test.iloc[:, 1:].copy(deep=True), self.__test.iloc[:, [0]].copy(deep=True))
        del self.__train, self.__test
        gc.collect()

        self.__oof_raw_score = self.__oof_raw_score.squeeze()
        self.__sub_raw_score = self.__sub_raw_score.squeeze()

    def model_fit_predict(self):
        ord_encoder_columns = [
            "bin_3", "bin_4",
            "ord_3", "ord_4", "ord_5"]
        tar_encoder_columns = [
            "nom_0", "nom_1", "nom_2", "nom_3", "nom_4",
            "nom_5", "nom_6", "nom_7", "nom_8", "nom_9",
            "ord_0", "ord_1", "ord_2", "day", "month"]

        # optimize
        def gbm_model_crossval(learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda):
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            cvals = np.zeros(shape=(folds.n_splits, ))

            for n_fold, (trn_idx, val_idx) in enumerate(
                    folds.split(X=self.__train_feature, y=self.__train_label)):

                trn_init_score = self.__oof_raw_score.iloc[trn_idx].copy(deep=True).to_numpy()
                val_init_score = self.__oof_raw_score.iloc[val_idx].copy(deep=True).to_numpy()

                trn_x = self.__train_feature.iloc[trn_idx].copy(deep=True)
                val_x = self.__train_feature.iloc[val_idx].copy(deep=True)

                trn_y = self.__train_label.iloc[trn_idx].copy(deep=True)
                val_y = self.__train_label.iloc[val_idx].copy(deep=True)

                encoder_validation = ColumnTransformer([
                    ("ORD_ENCODER", OrdinalEncoder(), ord_encoder_columns),
                    ("TAR_ENCODER", TargetEncoder(cols=tar_encoder_columns), tar_encoder_columns),
                ], remainder="drop")
                encoder_validation.fit(trn_x, trn_y)
                trn_x = encoder_validation.transform(trn_x)
                val_x = encoder_validation.transform(val_x)

                clf = LGBMClassifier(
                    max_depth=1,
                    learning_rate=learning_rate,
                    n_estimators=np.int(np.round(n_estimators)),
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=7,
                    n_jobs=-1
                )

                clf.fit(trn_x, trn_y, init_score=trn_init_score)
                cvals[n_fold] = roc_auc_score(val_y, expit(clf.predict(val_x, raw_score=True) + val_init_score))

            return cvals.mean()

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

        # model fit predict
        encoder = ColumnTransformer([
            ("ORD_ENCODER", OrdinalEncoder(), ord_encoder_columns),
            ("TAR_ENCODER", TargetEncoder(cols=tar_encoder_columns), tar_encoder_columns),
        ], remainder="drop")
        encoder.fit(self.__train_feature, self.__train_label)
        self.__train_feature = encoder.transform(self.__train_feature)
        self.__test_feature = encoder.transform(self.__test_feature)

        self.__model = LGBMClassifier(
            max_depth=1,
            learning_rate=optimizer.max["params"]["learning_rate"],
            n_estimators=np.int(np.round(optimizer.max["params"]["n_estimators"])),
            subsample=optimizer.max["params"]["subsample"],
            colsample_bytree=optimizer.max["params"]["colsample_bytree"],
            reg_alpha=optimizer.max["params"]["reg_alpha"],
            reg_lambda=optimizer.max["params"]["reg_lambda"],
            random_state=7,
            n_jobs=-1
        )
        self.__model.fit(self.__train_feature, self.__train_label, init_score=self.__oof_raw_score.to_numpy())
        self.__test_index["target"] = (
            expit(self.__model.predict(self.__test_feature, raw_score=True) + self.__sub_raw_score.to_numpy()))

    def data_write(self):
        self.__test_index.to_csv(
            os.path.join(self.__output_path, "sample_submission_linear_model_plus.csv"), index=False)


if __name__ == "__main__":
    lmp = LinearModelPlus(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    lmp.data_read()
    lmp.data_prepare()
    lmp.model_fit_predict()
    lmp.data_write()
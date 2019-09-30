# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
np.random.seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class FeatureStability(object):
    def __init__(self, *, path):
        self.__path = path

        self.__train = None
        self.__train_feature, self.__train_label = [None for _ in range(2)]

        self.__folds = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__path, "train.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 1:-1].copy(deep=True), self.__train.iloc[:, -1].copy(deep=True))
        del self.__train
        gc.collect()

    def model_fit_predict(self):
        # blending
        self.__folds = StratifiedKFold(
            n_splits=50, shuffle=True, random_state=7)

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
                X=self.__train_feature, y=self.__train_label)):

            # prepare data
            trn = self.__train_feature.iloc[trn_idx].copy(deep=True)
            val = self.__train_feature.iloc[val_idx].copy(deep=True)

            numerator = len(set(val["nom_9"]).difference(set(trn["nom_9"])))
            denominator = len(set(val["nom_9"]))

            print(str(n_fold) + ":" + str(numerator / denominator))
            # trn["target"] = 0
            # val["target"] = 1
            #
            # trn_model, val_model = train_test_split(pd.concat([trn, val]), test_size=0.25)
            #
            # for column in trn_model.columns:
            #     trn_model[column], indexer = trn_model[column].factorize()
            #     val_model[column] = indexer.get_indexer(val_model[column])
            #
            # # model train
            # trn_model = lgb.Dataset(trn_model.drop(["target"], axis=1), trn_model["target"])
            # val_model = lgb.Dataset(val_model.drop(["target"], axis=1), val_model["target"])
            #
            # print("*" * 36 + str(n_fold) + "*" * 36)
            # param = {
            #     "metric": "auc"
            # }
            # clf = lgb.train(
            #     param,
            #     trn_model,
            #     num_boost_round=500,
            #     valid_sets=[val_model],
            #     early_stopping_rounds=25,
            #     verbose_eval=True
            # )
            # print(pd.DataFrame({
            #     "feature_name": trn_model.feature_name,
            #     "feature_importance": clf.feature_importance()}
            # ).sort_values(by="feature_importance", ascending=False).head())


if __name__ == "__main__":
    fs = FeatureStability(path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge")
    fs.data_read()
    fs.data_prepare()
    fs.model_fit_predict()
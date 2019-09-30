# coding:utf-8

import os
import gc
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
np.random.seed(7)
tf.random.set_seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


def auc_score(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


class EntityEmbeddingNeuralNet(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]  # test_index dataframe

        self.__columns = list()
        self.__columns_counts = dict()  # each fold clear

        # model fit predict
        self.__folds = None
        self.__oof_preds = None
        self.__sub_preds = None

        self.__neural_net_util = importlib.import_module("NeuralNetUtil")
        self.__net, self.__early_stopping = [None for _ in range(2)]

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

        self.__train_feature = self.__train_feature[
            ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "day", "month"]].copy(deep=True)
        self.__test_feature = self.__test_feature[
            ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "day", "month"]].copy(deep=True)
        self.__columns = self.__train_feature.columns.tolist()

    def model_fit_predict(self):
        # blending
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        self.__oof_preds = np.zeros(shape=(self.__train_feature.shape[0],))
        self.__sub_preds = np.zeros(shape=(self.__test_feature.shape[0],))

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
                X=self.__train_feature, y=self.__train_label)):

            trn_x = self.__train_feature.iloc[trn_idx].copy(deep=True)
            val_x = self.__train_feature.iloc[val_idx].copy(deep=True)
            tes_x = self.__test_feature.copy(deep=True)

            trn_y = self.__train_label.iloc[trn_idx].copy(deep=True)
            val_y = self.__train_label.iloc[val_idx].copy(deep=True)

            for col in tqdm(self.__columns):
                num_unique = trn_x[col].nunique()

                if num_unique == 1:
                    trn_x = trn_x.drop([col], axis=1)
                    val_x = val_x.drop([col], axis=1)
                    tes_x = tes_x.drop([col], axis=1)
                else:

                    if trn_x[col].isna().sum():  # train exist np.nan
                        trn_x[col] = trn_x[col].fillna("missing")
                        categories = trn_x[col].unique()

                        val_x[col] = val_x[col].fillna("missing")
                        val_x[col] = val_x[col].apply(lambda x: x if x in categories else "missing")
                        tes_x[col] = tes_x[col].fillna("missing")
                        tes_x[col] = tes_x[col].apply(lambda x: x if x in categories else "missing")

                    else:  # train not exist np.nan
                        mode = trn_x[col].value_counts(ascending=True).index[0]
                        categories = trn_x[col].unique()

                        val_x[col] = val_x[col].fillna(mode)
                        val_x[col] = val_x[col].apply(lambda x: x if x in categories else mode)
                        tes_x[col] = tes_x[col].fillna(mode)
                        tes_x[col] = tes_x[col].apply(lambda x: x if x in categories else mode)

                    trn_x[col] = trn_x[col].astype(str)
                    val_x[col] = val_x[col].astype(str)
                    tes_x[col] = tes_x[col].astype(str)

                    encoder = LabelEncoder()
                    encoder.fit(trn_x[col])
                    trn_x[col] = encoder.transform(trn_x[col])
                    val_x[col] = encoder.transform(val_x[col])
                    tes_x[col] = encoder.transform(tes_x[col])

                    self.__columns_counts[col] = len(encoder.classes_)

            trn_feature_for_model = []
            val_feature_for_model = []
            tes_feature_for_model = []

            for col in self.__columns_counts.keys():
                trn_feature_for_model.append(trn_x[col].values)
                val_feature_for_model.append(val_x[col].values)
                tes_feature_for_model.append(tes_x[col].values)

            self.__net = self.__neural_net_util.network(
                columns_counts=self.__columns_counts,
                output_layer_bias=trn_y.mean()
            )
            self.__net.compile(
                loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=[auc_score])

            self.__net.fit(
                x=trn_feature_for_model,
                y=trn_y.values,
                epochs=75,
                batch_size=512,
                verbose=2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True
                    )],
                validation_data=(val_feature_for_model, val_y.values)
            )

            self.__neural_net_util.network_preformance(
                n_fold=n_fold,
                net=self.__net,
                trn_feature=trn_feature_for_model,
                val_feature=val_feature_for_model,
                trn_label=trn_y,
                val_label=val_y
            )

            pred_vals = self.__net.predict(val_feature_for_model).reshape((-1,))  # 2D shape -> 1D shape
            self.__oof_preds[val_idx] += logit(pred_vals)

            pred_test = self.__net.predict(tes_feature_for_model).reshape((-1,))
            self.__sub_preds += logit(pred_test) / self.__folds.n_splits

            self.__columns_counts.clear()
            del trn_x, val_x, tes_x, trn_y, val_y
            gc.collect()

    def data_write(self):
        print("Fold all prediction trn auc: %.5f" % (
            roc_auc_score(self.__train_label, expit(self.__oof_preds))))

        self.__test_index["target"] = expit(self.__sub_preds.reshape((-1,)))
        self.__test_index.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    eenn = EntityEmbeddingNeuralNet(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    eenn.data_read()
    eenn.data_prepare()
    eenn.model_fit_predict()
    eenn.data_write()
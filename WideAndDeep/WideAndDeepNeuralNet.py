# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)
tf.random.set_seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class WideAndDeepNeuralNet(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]  # test_index dataframe

        self.__columns = list()
        self.__columns_counts = dict()

        # blending
        self.__folds = None
        self.__val_preds = None
        self.__sub_preds = None

        # model
        self.__wide_columns = []
        self.__deep_columns = []

        self.__model = None

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

        self.__columns = self.__train_feature.columns.tolist()

    def model_fit_predict(self):
        # blending
        self.__folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        self.__val_preds = np.zeros(shape=(self.__train_feature.shape[0],))
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
                    trn_x[col] = encoder.transform(trn_x[col]).astype(str)
                    val_x[col] = encoder.transform(val_x[col]).astype(str)
                    tes_x[col] = encoder.transform(tes_x[col]).astype(str)

                    self.__columns_counts[col] = encoder.classes_.tolist()

            trn_input = tf.compat.v1.estimator.inputs.pandas_input_fn(
                x=trn_x,
                y=trn_y,
                batch_size=512,
                num_epochs=1,
                shuffle=True
            )
            val_input = tf.compat.v1.estimator.inputs.pandas_input_fn(
                x=val_x,
                batch_size=val_x.shape[0],
                shuffle=False
            )
            tes_input = tf.compat.v1.estimator.inputs.pandas_input_fn(
                x=tes_x,
                batch_size=tes_x.shape[0],
                shuffle=False
            )

            for column, count in self.__columns_counts.items():
                categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                    key=column, vocabulary_list=count)

                self.__wide_columns.append(tf.feature_column.indicator_column(categorical_column))
                self.__deep_columns.append(tf.feature_column.embedding_column(
                    categorical_column, dimension=4 * int(np.round(np.log2(len(count))))))

            self.__model = tf.estimator.DNNLinearCombinedClassifier(
                linear_feature_columns=self.__wide_columns,
                dnn_feature_columns=self.__deep_columns,
                dnn_hidden_units=[1],
                dnn_activation_fn=tf.nn.sigmoid
            )
            self.__model.train(input_fn=trn_input)

            pred_val, pred_tes = [np.array([]) for _ in range(2)]
            for element in self.__model.predict(input_fn=val_input):
                pred_val = np.append(pred_val, element["logits"])

            print("Fold %i prediction trn auc: %.5f" % (n_fold, roc_auc_score(val_y.tolist(), expit(pred_val.tolist()))))

            for element in self.__model.predict(input_fn=tes_input):
                pred_tes = np.append(pred_tes, element["logits"])

            self.__val_preds[val_idx] += pred_val
            self.__sub_preds += pred_tes / self.__folds.n_splits

            self.__columns_counts.clear()
            del trn_x, val_x, tes_x, trn_y, val_y, self.__model
            gc.collect()

    def data_write(self):
        print("Fold all prediction trn auc: %.5f" % (roc_auc_score(self.__train_label, expit(self.__val_preds))))

        self.__test_index["target"] = expit(self.__sub_preds.reshape((-1,)))
        self.__test_index.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    wadnn = WideAndDeepNeuralNet(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    wadnn.data_read()
    wadnn.data_prepare()
    wadnn.model_fit_predict()
    wadnn.data_write()
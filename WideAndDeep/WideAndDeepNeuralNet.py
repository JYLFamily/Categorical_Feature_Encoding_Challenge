# coding:utf-8

import os
import gc
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.metrics import AUC
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
np.random.seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class FitGenerator(Sequence):
    def __init__(self, feature, label, encoder, shuffle, batch_size):
        self.__index = np.arange(feature.shape[0])
        self.__feature, self.__label = feature, label
        self.__encoder, self.__shuffle = encoder, shuffle
        self.__batch_size = batch_size

    def __len__(self):
        return self.__feature.shape[0] // self.__batch_size

    def __getitem__(self, idx):
        index = self.__index[idx * self.__batch_size: (idx + 1) * self.__batch_size]

        batch_feature, batch_label = [self.__encoder.transform(self.__feature[index, :15])], self.__label[index]
        for col in range(15, 23):
            batch_feature.append(self.__feature[index, col])

        return batch_feature, batch_label

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__index)


class PredictGenerator(Sequence):
    def __init__(self, feature, encoder):
        self.__index = np.arange(feature.shape[0])
        self.__feature = feature
        self.__encoder = encoder

    def __len__(self):
        return self.__feature.shape[0]

    def __getitem__(self, idx):
        index = self.__index[idx: (idx + 1)]

        batch_feature = [self.__encoder.transform(self.__feature[index, :15])]
        for col in range(15, 23):
            batch_feature.append(self.__feature[index, col])

        return batch_feature


class WideAndDeepNeuralNet(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]  # test_index dataframe

        self.__columns = list()
        self.__columns_counts = OrderedDict()  # each fold clear

        # model fit predict
        self.__folds = None
        self.__val_preds = None
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

        self.__columns = self.__train_feature.columns.tolist()

    def model_fit_predict(self):
        # blending
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
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
                    trn_x[col] = encoder.transform(trn_x[col])
                    val_x[col] = encoder.transform(val_x[col])
                    tes_x[col] = encoder.transform(tes_x[col])

                    self.__columns_counts[col] = len(encoder.classes_)

            # data
            trn_x, val_x, tes_x = trn_x.to_numpy(), val_x.to_numpy(), tes_x.to_numpy()
            trn_y, val_y = trn_y.to_numpy(), val_y.to_numpy()
            encoder = OneHotEncoder(categories="auto", sparse=False)
            encoder.fit(trn_x[:, :15])

            # neural net
            self.__net = self.__neural_net_util.network(
                columns_counts=self.__columns_counts,
                output_layer_bias=trn_y.mean()
            )
            self.__net.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=[AUC()])
            self.__net.fit_generator(
                generator=FitGenerator(trn_x, trn_y, encoder, True, 32),
                steps_per_epoch=trn_x.shape[0] // 32,
                epochs=75,
                verbose=2,
                callbacks=[
                    EarlyStopping(
                        patience=5,
                        restore_best_weights=True
                    )],
                validation_data=FitGenerator(val_x, val_y, encoder, False, 1),
                validation_steps=val_x.shape[0],
                workers=1,
                use_multiprocessing=False
            )

            pred_trns = self.__net.predict_generator(
                generator=PredictGenerator(trn_x, encoder),
                steps=trn_x.shape[0],
                workers=1,
                use_multiprocessing=False).reshape((-1,))
            pred_vals = self.__net.predict_generator(
                generator=PredictGenerator(val_x, encoder),
                steps=val_x.shape[0],
                workers=1,
                use_multiprocessing=False).reshape((-1,))
            pred_test = self.__net.predict_generator(
                generator=PredictGenerator(tes_x, encoder),
                steps=tes_x.shape[0],
                workers=1,
                use_multiprocessing=False).reshape((-1,))

            self.__neural_net_util.network_preformance(
                n_fold=n_fold,
                pred_trn=pred_trns,
                pred_val=pred_vals,
                trn_label=trn_y,
                val_label=val_y
            )

            self.__val_preds[val_idx] += logit(pred_vals)
            self.__sub_preds += logit(pred_test) / self.__folds.n_splits

            self.__columns_counts.clear()
            del trn_x, val_x, tes_x, trn_y, val_y
            gc.collect()

    def data_write(self):
        print("Fold all prediction trn auc: %.5f" % (roc_auc_score(self.__train_label, expit(self.__val_preds))))

        self.__test_index["target"] = expit(self.__sub_preds.reshape((-1,)))
        self.__test_index.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    wadn = WideAndDeepNeuralNet(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    wadn.data_read()
    wadn.data_prepare()
    wadn.model_fit_predict()
    wadn.data_write()
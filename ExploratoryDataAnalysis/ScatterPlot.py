# coding:utf-8

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
np.random.seed(7)
pd.set_option("max_row", None)
pd.set_option("max_columns", None)


class ScatterPlot(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        self.__train = None
        self.__train_feature, self.__train_label = [None for _ in range(2)]

        self.__encoder = None
        self.__pca, self.__t_sne = [None for _ in range(2)]

    def data_read(self):
        self.__train = pd.read_csv(
            os.path.join(self.__input_path, "train.csv"))
        self.__train = self.__train.drop(["id"], axis=1)
        self.__train_feature, self.__train_label = (
            self.__train.drop(["target"], axis=1).copy(deep=True), self.__train["target"].copy(deep=True))
        self.__train_feature = self.__train_feature.astype(str)

    def data_prepare(self):
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature, self.__train_label)
        self.__train_feature = self.__encoder.transform(self.__train_feature)

        self.__pca = PCA(n_components=2, random_state=7)
        self.__train_feature = self.__pca.fit_transform(self.__train_feature)
        self.__train_feature = pd.DataFrame(self.__train_feature, columns=["col_1", "col_2"])

        # self.__t_sne = TSNE(verbose=True, random_state=7)
        # self.__train_feature = self.__t_sne.fit_transform(self.__train_feature)
        # self.__train_feature = pd.DataFrame(self.__train_feature, columns=["col_1", "col_2"])

    def scatter_plot(self):
        _, ax = plt.subplots(figsize=(16, 9))
        ax = sns.scatterplot(
            x="col_1",
            y="col_2",
            hue=self.__train_label,
            data=self.__train_feature,
            ax=ax
        )
        ax.get_figure().savefig(os.path.join(self.__output_path, "PCA.png"))


if __name__ == "__main__":
    sp = ScatterPlot(
        input_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge",
        output_path="E:\\Kaggle\\Categorical_Feature_Encoding_Challenge"
    )
    sp.data_read()
    sp.data_prepare()
    sp.scatter_plot()

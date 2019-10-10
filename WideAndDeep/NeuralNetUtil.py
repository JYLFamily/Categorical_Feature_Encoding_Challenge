# coding:utf-8


import numpy as np
from operator import *
from scipy.special import logit
from keras.models import Model
from keras.initializers import constant, he_normal, TruncatedNormal
from keras.layers import Input, Embedding, Reshape, Dense, Activation, Concatenate
from sklearn.metrics import roc_auc_score
np.random.seed(7)


def network(columns_counts, output_layer_bias):
    input_layers = list()

    # wide part
    input_wide_layer = Input(
        shape=(sum(list(columns_counts.values())[:15]),),
        name="wide_input")
    input_layers.append(input_wide_layer)

    # deep part
    embedding_layers = list()
    for col, num in list(columns_counts.items())[15:]:
        # input deep part
        input_deep_layer = Input(shape=(1,), name=col + "_deep_input")
        input_layers.append(input_deep_layer)

        # embedding deep part
        embedding_layer = Embedding(
            input_dim=int(num),  # must int
            output_dim=mul(int(np.round(np.log2(num))), 4),  # must int
            embeddings_initializer=TruncatedNormal(mean=0, stddev=1/np.sqrt(num), seed=7),
            input_length=1,
            name=col + "_embedding")(input_deep_layer)
        embedding_layer = (
            Reshape(target_shape=(
                mul(int(np.round(np.log2(num))), 4),), name=col + "_reshape")(embedding_layer))
        embedding_layer = Activation(activation="sigmoid", name=col + "_activation")(embedding_layer)
        embedding_layers.append(embedding_layer)

    # output
    output_layer = Dense(
        units=1,
        kernel_initializer=he_normal(seed=7),
        bias_initializer=constant(logit(output_layer_bias)),
        activation="sigmoid", name="output_layer")(
        Concatenate()([
            Dense(
                units=1,
                kernel_initializer=he_normal(seed=7),
                activation="sigmoid", name="dense_wide")(input_wide_layer),
            Dense(
                units=1,
                kernel_initializer=he_normal(seed=7),
                activation="sigmoid", name="dense_deep")(Concatenate()(embedding_layers))
        ]))

    return Model(input_layers, output_layer)


def network_preformance(n_fold, pred_trn, pred_val, trn_label, val_label):
    trn_auc = roc_auc_score(trn_label.reshape((-1,)), pred_trn)
    val_auc = roc_auc_score(val_label.reshape((-1,)), pred_val)

    print("Fold %i prediction trn auc: %.5f" % (n_fold, trn_auc))
    print("Fold %i prediction val auc: %.5f" % (n_fold, val_auc))

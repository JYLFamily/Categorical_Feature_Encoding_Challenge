# coding:utf-8

import numpy as np
import tensorflow as tf
from operator import *
from scipy.special import logit
from sklearn.metrics import roc_auc_score
np.random.seed(7)
tf.random.set_seed(7)


def network(columns_counts, output_layer_bias):
    input_layers = list()
    embedding_layers = list()

    for col, num in columns_counts.items():
        input_layer = tf.keras.layers.Input(shape=(), name=col + "_input")
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=int(num),  # must int
            output_dim=mul(int(np.round(np.log2(num))), 4),  # must int
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=1/np.sqrt(num), seed=7),
            input_length=1,
            name=col + "_embedding")(input_layer)
        embedding_layer = (
            tf.keras.layers.Reshape(target_shape=(
                mul(int(np.round(np.log2(num))), 4),), name=col + "_reshape")(embedding_layer))
        input_layers.append(input_layer)
        embedding_layers.append(embedding_layer)

    hidden_layer_1 = tf.keras.layers.Dense(
        units=16,
        kernel_initializer=tf.keras.initializers.he_normal(),
        activation="relu",
        name="dense_1")(
        tf.keras.layers.Concatenate()(embedding_layers))

    hidden_layer_2 = tf.keras.layers.Dense(
        units=16,
        kernel_initializer=tf.keras.initializers.he_normal(),
        activation="relu",
        name="dense_2")(
        hidden_layer_1)

    hidden_layer_3 = tf.keras.layers.Dense(
        units=16,
        kernel_initializer=tf.keras.initializers.he_normal(),
        activation="relu",
        name="dense_3")(
        tf.keras.layers.Add()([hidden_layer_1, hidden_layer_2]))

    output_layer = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=tf.keras.initializers.he_normal(),
        bias_initializer=tf.keras.initializers.constant(logit(output_layer_bias)),
        activation="sigmoid", name="output_layer")(
        tf.keras.layers.Add()([hidden_layer_1, hidden_layer_3]))

    return tf.keras.Model(input_layers, output_layer)


def network_preformance(n_fold, net, trn_feature, val_feature, trn_label, val_label):
    pred_trn = net.predict(trn_feature).reshape((-1,))
    trn_auc = roc_auc_score(trn_label.values.reshape((-1,)), pred_trn)

    pred_val = net.predict(val_feature).reshape((-1,))
    val_auc = roc_auc_score(val_label.values.reshape((-1,)), pred_val)

    print("Fold %i prediction trn auc: %.5f" % (n_fold, trn_auc))
    print("Fold %i prediction val auc: %.5f" % (n_fold, val_auc))




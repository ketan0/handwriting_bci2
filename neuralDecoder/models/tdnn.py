#!/usr/bin/env python3
from tensorflow.keras import Model
from tensorflow import expand_dims
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, BatchNormalization, Dropout, ReLU
import tensorflow as tf

# TODO: add activations, dropout?
# following pytorch implementation

class TDNNBlock(Model):
    def __init__(self,
                 output_dim=31,
                 context_size=10,
                 stride=1,
                 dilation=1,
                 batch_norm=True,
                 dropout_p=0.0,
                 grouped=1,
                 pad="SAME"):
        super().__init__()
        self.context_size = context_size
        self.stride = stride
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.grouped = grouped
        self.pad = pad

        self.kernel = Conv1D(self.output_dim, self.context_size, dilation_rate=self.dilation, padding=self.pad, strides=1)
        if self.batch_norm:
            self.bn = BatchNormalization(axis=-1, dtype='float64')
        if self.dropout_p:
            self.dropout = Dropout(rate=self.dropout_p, dtype='float64')
        self.activation = ReLU()

    def call(self, inputs):
        # x = tf.transpose(inputs, perm=[0, 2, 1])
        print(inputs.shape)
        x = self.kernel(inputs)
        print(x.shape)

        if self.dropout_p:
            x = self.dropout(x)
        if self.batch_norm:
            x = self.bn(x)

        x = self.activation(x)
        return x

class TDNN(Model):
    def __init__(self,
                 num_classes=31, #output dimsensions
                 dropout_p=0.0,
                 batch_norm=True,

                 grouped=1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.TDNN1 = TDNNBlock(output_dim=256, context_size=2, stride=1, dilation=1, dropout_p=self.dropout_p, batch_norm=self.batch_norm)
        self.TDNN2 = TDNNBlock(output_dim=512, context_size=4, stride=1, dilation=2, dropout_p=self.dropout_p, batch_norm=self.batch_norm)
        self.TDNN3 = TDNNBlock(output_dim=256, context_size=8, stride=1, dilation=4, dropout_p=self.dropout_p, batch_norm=self.batch_norm)
        self.TDNN4 = TDNNBlock(output_dim=128, context_size=16, stride=1, dilation=4, dropout_p=self.dropout_p, batch_norm=self.batch_norm)
        self.linear1 = Dense(128, activation="relu")
        self.linear2 = Dense(self.num_classes, activation="softmax")

    def call(self, x):
        x = self.TDNN1(x)
        x = self.TDNN2(x)
        x = self.TDNN3(x)
        x = self.TDNN4(x)
        x = self.linear1(x)
        outs = self.linear2(x)
        return outs









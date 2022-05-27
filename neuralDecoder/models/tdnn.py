#!/usr/bin/env python3
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D

# TODO: add activations, dropout?
class TDNN(Model):
    def __init__(self, dropout_p: int, n_classes: int, use_bias=False):
        super().__init__()
        self.conv1 = Conv1D(filters=32, kernel_size=12,  padding="causal", dilation_rate=2, strides=1)
        self.pool1 = MaxPool1D(2, 1, padding="same")
        self.conv2 = Conv1D(filters=64, kernel_size=32, padding="causal", dilation_rate=2, strides=1)
        self.pool2 = MaxPool1D(2, 1, padding="same")
        self.conv3 = Conv1D(filters=128, kernel_size=64, padding="causal", dilation_rate=2, strides=1)
        self.pool3 = MaxPool1D(2, 1, padding="same")
        self.conv4 = Conv1D(filters=256, kernel_size=64,  padding="causal", dilation_rate=2, strides=1)
        self.pool4 = MaxPool1D(2, 1, padding="same")
        self.lin1 = Dense(512, use_bias=use_bias)
        self.lin2 = Dense(256, use_bias=use_bias)
        self.lin3 = Dense(n_classes, use_bias=use_bias)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.lin1(x)
        x = self.lin2(x)
        outputs = self.lin3(x)
        return outputs

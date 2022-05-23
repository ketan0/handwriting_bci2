import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, Dense

from conformer_tf import ConformerBlock
from tensorflow.python.keras.layers.core import Dropout

class GRU(Model):
    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0):
        super(GRU, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        #actReg = tf.keras.regularizers.L2(actReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional

        if bidirectional:
            self.initStates = [
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
            ]
        else:
            self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)))

        self.rnn1 = tf.keras.layers.GRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        kernel_regularizer=weightReg,
                                        activity_regularizer=actReg,
                                        recurrent_initializer=recurrent_init,
                                        kernel_initializer=kernel_init,
                                        dropout=dropout)
        self.rnn2 = tf.keras.layers.GRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        kernel_regularizer=weightReg,
                                        activity_regularizer=actReg,
                                        recurrent_initializer=recurrent_init,
                                        kernel_initializer=kernel_init,
                                        dropout=dropout)
        if bidirectional:
            self.rnn1 = tf.keras.layers.Bidirectional(self.rnn1)
            self.rnn2 = tf.keras.layers.Bidirectional(self.rnn2)
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, state=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        if state is None:
            if self.bidirectional:
                initState1 = [tf.tile(s, [batchSize, 1]) for s in self.initStates]
            else:
                initState1 = tf.tile(self.initStates, [batchSize, 1])
            initState2 = None
        else:
            initState1 = state[0]
            initState2 = state[1]

        x, s1 = self.rnn1(x, training=training, initial_state=initState1)
        if self.subsampleFactor > 1:
            x = x[:, ::self.subsampleFactor, :]
        x, s2 = self.rnn2(x, training=training, initial_state=initState2)
        x = self.dense(x, training=training)

        if returnState:
            return x, [s1, s2]
        else:
            return x

    def getIntermediateLayerOutput(self, x):
        x, _ = self.rnn1(x)
        return x


class Conv2dSubsampling(Layer):
    def __init__(self, out_channels: int, subsampling_factor: int):
        super().__init__()
        self.sequential = Sequential([
            Conv2D(out_channels, kernel_size=3, strides=(subsampling_factor // 2), activation='relu'),
            Conv2D(out_channels, kernel_size=3, strides=(subsampling_factor // 2), activation='relu'),
        ])
        self.subsampling_factor = subsampling_factor
    def call(self, inputs):
        outputs = self.sequential(tf.expand_dims(inputs, -1))
        batch_size, subsampled_lengths, subsampled_dim, channels = outputs.get_shape().as_list()
        print('SHAPE:', batch_size, subsampled_lengths, subsampled_dim, channels)
        outputs = tf.transpose(outputs, perm=[0, 1, 3, 2])
        outputs = tf.reshape(outputs, [batch_size, subsampled_lengths, channels * subsampled_dim])

        # outputs = tf.transpose(outputs, perm=[0, 3, 2, 1])
        # outputs = tf.reshape(outputs, [batch_size, channels * subsampled_dim, subsampled_lengths])

        return outputs


class ConformerEncoder(Layer):
    def __init__(
            self,
            encoder_dim: int,
            num_encoder_layers: int,
            num_attention_heads: int,
            feed_forward_expansion_factor: int,
            conv_expansion_factor: int,
            input_dropout_p: float,
            feed_forward_dropout_p: float,
            attention_dropout_p: float,
            conv_dropout_p: float,
            conv_kernel_size: int,
            subsampling_factor: int,
    ):
        super().__init__()
        self.conv_subsample = Conv2dSubsampling(out_channels=encoder_dim, subsampling_factor=subsampling_factor)
        self.input_projection = Sequential([
            Dense(encoder_dim),
            Dropout(input_dropout_p),
        ])
        self.conformer_blocks = Sequential([
            ConformerBlock(dim=encoder_dim, dim_head=encoder_dim, heads=num_attention_heads,
                           ff_mult=feed_forward_expansion_factor, # TODO: half_step_residual
                           conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size,
                           attn_dropout=attention_dropout_p, ff_dropout=feed_forward_dropout_p,
                           conv_dropout=conv_dropout_p)
            for _ in range(num_encoder_layers)
        ])

    def call(self, inputs):
        outputs = self.conv_subsample(inputs)
        # print('AFTER SUBSAMPLE:', outputs)
        outputs = self.input_projection(outputs)
        outputs = self.conformer_blocks(outputs)
        return outputs


class Conformer(Model):
    def __init__(
            self,
            num_classes: int,
            encoder_dim: int,
            num_encoder_layers: int,
            num_attention_heads: int,
            feed_forward_expansion_factor: int,
            conv_expansion_factor: int,
            input_dropout_p: float,
            feed_forward_dropout_p: float,
            attention_dropout_p: float,
            conv_dropout_p: float,
            conv_kernel_size: int,
            subsampling_factor: int,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            encoder_dim,
            num_encoder_layers,
            num_attention_heads,
            feed_forward_expansion_factor,
            conv_expansion_factor,
            input_dropout_p,
            feed_forward_dropout_p,
            attention_dropout_p,
            conv_dropout_p,
            conv_kernel_size,
            subsampling_factor,
        )
        self.fc = Dense(num_classes, use_bias=False)

    def call(self, inputs):
        encoder_outputs = self.encoder(inputs)
        outputs = self.fc(encoder_outputs)
        return outputs

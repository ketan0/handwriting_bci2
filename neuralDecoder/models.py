import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, Dense, Conv1D, MaxPool1D
from tensorflow.keras.activations import relu, elu

# from tensorflow.keras.layers import Layer, Conv2D, Dense

# from conformer_tf import ConformerBlock
# from tensorflow.python.keras.layers.core import Dropout

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


# class Conv2dSubsampling(Layer):
#     def __init__(self, out_channels: int, subsampling_factor: int):
#         super().__init__()
#         self.sequential = Sequential([
#             Conv2D(out_channels, kernel_size=3, strides=(subsampling_factor // 2), activation='relu'),
#             Conv2D(out_channels, kernel_size=3, strides=(subsampling_factor // 2), activation='relu'),
#         ])
#         self.subsampling_factor = subsampling_factor
#     def call(self, inputs):
#         outputs = self.sequential(tf.expand_dims(inputs, -1))
#         batch_size, subsampled_lengths, subsampled_dim, channels = outputs.get_shape().as_list()
#         print('SHAPE:', batch_size, subsampled_lengths, subsampled_dim, channels)
#         outputs = tf.transpose(outputs, perm=[0, 1, 3, 2])
#         outputs = tf.reshape(outputs, [batch_size, subsampled_lengths, channels * subsampled_dim])
#         # TODO: generate random matrix + output

#         # outputs = tf.transpose(outputs, perm=[0, 3, 2, 1])
#         # outputs = tf.reshape(outputs, [batch_size, channels * subsampled_dim, subsampled_lengths])

#         return outputs

# TODO: convert this to yaml file (omitting irrelevant stuff, like speech_config)
CONFORMER_CONFIG = {
    "speech_config": {
        "sample_rate": 16000,
        "frame_ms": 25,
        "stride_ms": 10,
        "num_feature_bins": 80,
        "feature_type": "log_mel_spectrogram",
        "preemphasis": 0.97,
        "normalize_signal": True,
        "normalize_feature": True,
        "normalize_per_frame": False,
    },
    "decoder_config": {
        "vocabulary": None,
        "target_vocab_size": 1000,
        "max_subword_length": 10,
        "blank_at_zero": True,
        "beam_width": 0,
        "norm_score": True,
        "corpus_files": None,
    },
    "model_config": {
        "name": "conformer",
        "encoder_subsampling": {
            "type": "conv2d",
            "filters": 144,
            "kernel_size": 3,
            "strides": 2,
        },
        "encoder_positional_encoding": "sinusoid_concat",
        "encoder_dmodel": 144,
        "encoder_num_blocks": 16,
        "encoder_head_size": 36,
        "encoder_num_heads": 4,
        "encoder_mha_type": "relmha",
        "encoder_kernel_size": 32,
        "encoder_fc_factor": 0.5,
        "encoder_dropout": 0.1,
        "prediction_embed_dim": 320,
        "prediction_embed_dropout": 0,
        "prediction_num_rnns": 1,
        "prediction_rnn_units": 320,
        "prediction_rnn_type": "lstm",
        "prediction_rnn_implementation": 2,
        "prediction_layer_norm": True,
        "prediction_projection_units": 0,
        "joint_dim": 320,
        "prejoint_linear": True,
        "joint_activation": "tanh",
        "joint_mode": "add",
    },
    "learning_config": {
        "train_dataset_config": {
            "use_tf": True,
            "augmentation_config": {
                "feature_augment": {
                    "time_masking": {
                        "num_masks": 10,
                        "mask_factor": 100,
                        "p_upperbound": 0.05,
                    },
                    "freq_masking": {"num_masks": 1, "mask_factor": 27},
                }
            },
            "data_paths": [
                "/mnt/h/ML/Datasets/ASR/Raw/LibriSpeech/train-clean-100/transcripts.tsv"
            ],
            "tfrecords_dir": None,
            "shuffle": True,
            "cache": True,
            "buffer_size": 100,
            "drop_remainder": True,
            "stage": "train",
        },
        "eval_dataset_config": {
            "use_tf": True,
            "data_paths": None,
            "tfrecords_dir": None,
            "shuffle": False,
            "cache": True,
            "buffer_size": 100,
            "drop_remainder": True,
            "stage": "eval",
        },
        "test_dataset_config": {
            "use_tf": True,
            "data_paths": None,
            "tfrecords_dir": None,
            "shuffle": False,
            "cache": True,
            "buffer_size": 100,
            "drop_remainder": True,
            "stage": "test",
        },
        "optimizer_config": {
            "warmup_steps": 40000,
            "beta_1": 0.9,
            "beta_2": 0.98,
            "epsilon": 1e-09,
        },
        "running_config": {
            "batch_size": 2,
            "num_epochs": 50,
            "checkpoint": {
                "filepath": "/mnt/e/Models/local/conformer/checkpoints/{epoch:02d}.h5",
                "save_best_only": False,
                "save_weights_only": True,
                "save_freq": "epoch",
            },
            "states_dir": "/mnt/e/Models/local/conformer/states",
            "tensorboard": {
                "log_dir": "/mnt/e/Models/local/conformer/tensorboard",
                "histogram_freq": 1,
                "write_graph": True,
                "write_images": True,
                "update_freq": "epoch",
                "profile_batch": 2,
            },
        },
    },
}

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

# def create_conformer(args):

# class ConformerEncoder(Layer):
#     def __init__(
#             self,
#             encoder_dim: int,
#             num_encoder_layers: int,
#             num_attention_heads: int,
#             feed_forward_expansion_factor: int,
#             conv_expansion_factor: int,
#             input_dropout_p: float,
#             feed_forward_dropout_p: float,
#             attention_dropout_p: float,
#             conv_dropout_p: float,
#             conv_kernel_size: int,
#             subsampling_factor: int,
#     ):
#         super().__init__()
#         self.conv_subsample = Conv2dSubsampling(out_channels=encoder_dim, subsampling_factor=subsampling_factor)
#         self.input_projection = Sequential([
#             Dense(encoder_dim),
#             Dropout(input_dropout_p),
#         ])
#         self.conformer_blocks = Sequential([
#             ConformerBlock(dim=encoder_dim, dim_head=encoder_dim, heads=num_attention_heads,
#                            ff_mult=feed_forward_expansion_factor, # TODO: half_step_residual
#                            conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size,
#                            attn_dropout=attention_dropout_p, ff_dropout=feed_forward_dropout_p,
#                            conv_dropout=conv_dropout_p)
#             for _ in range(num_encoder_layers)
#         ])

#     def call(self, inputs):
#         outputs = self.conv_subsample(inputs)
#         # print('AFTER SUBSAMPLE:', outputs)
#         outputs = self.input_projection(outputs)
#         outputs = self.conformer_blocks(outputs)
#         return outputs


# class Conformer(Model):
#     def __init__(
#             self,
#             num_classes: int,
#             encoder_dim: int,
#             num_encoder_layers: int,
#             num_attention_heads: int,
#             feed_forward_expansion_factor: int,
#             conv_expansion_factor: int,
#             input_dropout_p: float,
#             feed_forward_dropout_p: float,
#             attention_dropout_p: float,
#             conv_dropout_p: float,
#             conv_kernel_size: int,
#             subsampling_factor: int,
#     ):
#         super().__init__()
#         self.encoder = ConformerEncoder(
#             encoder_dim,
#             num_encoder_layers,
#             num_attention_heads,
#             feed_forward_expansion_factor,
#             conv_expansion_factor,
#             input_dropout_p,
#             feed_forward_dropout_p,
#             attention_dropout_p,
#             conv_dropout_p,
#             conv_kernel_size,
#             subsampling_factor,
#         )
#         self.fc = Dense(num_classes, use_bias=False)

#     def call(self, inputs):
#         encoder_outputs = self.encoder(inputs)
#         outputs = self.fc(encoder_outputs)
#         return outputs

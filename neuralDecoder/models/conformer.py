#!/usr/bin/env python3

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
        "vocabulary": '/Users/ketanagrawal/CS224s/handwriting_bci2/neuralDecoder/datasets/char_def.txt',
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
            #**TODO: bump back up**
            # "filters": 144,
            "filters": 4,
            "kernel_size": 3,
            "strides": 2,
        },
        "encoder_positional_encoding": "sinusoid_concat",

        #TODO: bump back up**
        # "encoder_dmodel": 144,
        "encoder_dmodel": 4,

        #**TODO: bump back up**
        # "encoder_num_blocks": 16,
        "encoder_num_blocks": 2,

        "encoder_head_size": 36,
        "encoder_num_heads": 4,
        "encoder_mha_type": "relmha",
        "encoder_kernel_size": 32,
        "encoder_fc_factor": 0.5,

        #TODO: change back**
        # "encoder_dropout": 0.1,
        "encoder_dropout": 0,

        #**TODO: bump back up**
        # "prediction_embed_dim": 320,
        "prediction_embed_dim": 4,

        "prediction_embed_dropout": 0,
        "prediction_num_rnns": 1,

        #**TODO: bump back up**
        # "prediction_rnn_units": 320,
        "prediction_rnn_units": 4,

        "prediction_rnn_type": "lstm",
        "prediction_rnn_implementation": 2,
        "prediction_layer_norm": True,
        "prediction_projection_units": 0,

        #TODO: bump back up**
        # "joint_dim": 320,
        "joint_dim": 16,

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

import tensorflow as tf
from text.symbols import symbols

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=5000,
        iters_per_checkpoint=500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,

        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False, # if true, 1st element in the filelist should be mel
        mel_data_type='numpy', # 'numpy' or 'torch'
        training_files='filelists/ljspeech_wav_train.txt',
        validation_files='filelists/ljspeech_wav_test.txt',
        text_cleaners=['english_cleaners'], # english_cleaners, korean_cleaners

        ################################
        # Emotion Embedding Parameters #
        ################################
        include_emo_emb=False,  # check filelist and ensure include emo if True
        load_emo_from_disk=True,  # currently only support True (ignored if include_emo_emb is False)
        emo_emb_dim=64,

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256, # number audio of frames between stft colmns, default win_length/4
        win_length=1024, # win_length int <= n_ftt: fft window size (frequency domain), defaults to win_length = n_fft
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=11025.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols = len(symbols), # set 80 for korean_cleaners. set 65 for english_cleaners
        symbols_embedding_dim=512,

        # Transcript encoder parameters
        encoder_kernel_size = 5,
        encoder_n_convolutions = 3,
        encoder_embedding_dim = 512,

        # Speaker embedding parameters
        n_speakers = 1,
        speaker_embedding_dim=16,

        # ---------------------------------------- #
        # emotion 
        n_emotions = 4, # number of emotion labels
        emotion_embedding_dim=16,
        vae_input_type='mel',  # mel (default) or emo

        # reference encoder
        E = 512,
        ref_enc_filters = [32, 32, 64, 64, 128, 128],
        ref_enc_size = [3, 3],
        ref_enc_strides = [2, 2],
        ref_enc_pad = [1, 1],
        ref_enc_gru_size = 512 // 2,

        z_latent_dim = 32,
        anneal_function = 'logistic',
        anneal_k = 0.0025,
        anneal_x0 = 10000,
        anneal_upper = 0.2,
        anneal_lag = 50000,
        anneal_constant = 0.001,

        # Prosody embedding parameters
        prosody_n_convolutions = 6,
        prosody_conv_dim_in = [1, 32, 32, 64, 64, 128],
        prosody_conv_dim_out = [32, 32, 64, 64, 128, 128],
        prosody_conv_kernel = 3,
        prosody_conv_stride = 2,
        prosody_embedding_dim = 128,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams

def hparams_debug_string(hparams):
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)

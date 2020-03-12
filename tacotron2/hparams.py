import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=200,
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
        load_mel_from_disk=False,
        training_files='filelists/train.csv',
        validation_files='filelists/val.csv',
        test_files='filelists/test.csv',
        text_cleaners=['persian_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        #Audio parameters are the most important parameters to tune when using this work on your personal data. Below are the beginner steps to adapt
        #this work to your personal data:
        #	1- Determine my data sample rate: First you need to determine your audio sample_rate (how many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
        #		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, so there are plenty of examples to refer to)
        #	2- set sample_rate parameter to your data correct sample rate
        #	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms window_size, and 12.5ms frame_shift(hop_size))
        #		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
        #		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
        #	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
        #		a- n_fft can be either equal to win_size or the first power of 2 that comes after win_size. I usually recommend using the latter
        #			to be more consistent with signal processing friends. No big difference to be seen however. For the tuto example: n_fft = 2048 = 2**11
        #		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 1 = 1025.
        #		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto example: upsample_scales=[15, 20] where 15 * 20 = 300
        #			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
        #			so the training segments should be long enough (2.8~3x upsample_scales[0] * hop_size or longer) so that the first kernel size can see the middle 
        #			of the samples efficiently. The length of WaveNet training segments is under the parameter "max_time_steps".
        #	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying preprocessing (or part of it, ctrl-C to stop), then use the
        #		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That will first give you some idea about your above parameters, and
        #		it will also give you an idea about trimming. If silences persist, try reducing trim_top_db slowly. If samples are trimmed mid words, try increasing it.
        #	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are showing black silent regions on top), then restart from step 2.

        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.4,
        p_decoder_dropout=0.4,

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
        learning_rate=1.5e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=56,
        # old value 52
        # 56 for 16
        # 40 for 11
        
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams

# Create speech datasets as used in the experiment in the paper,
# Towards Empirical Sandwich Bounds on the Rate-Distortion Function, ICLR 2022, Yibo Yang, Stephan Mandt
# https://arxiv.org/abs/2111.12166

# Compute STFT feature on small time windows of the audios; specifically, using the log
# of the magnitude of the STFT output (which are complex numbers).
# Inspired by https://www.tensorflow.org/tutorials/audio/simple_audio
# Usage:
# 1. Clone the FSDD repo from git@github.com:Jakobovski/free-spoken-digit-dataset.git
# 2. Update the 'data_dir' below to point to the cloned repo
# 3. Run this script, like `python create_data.py`
# Yibo Yang, 2021

import numpy as np
import tensorflow as tf


def decode_audio(audio_path):
    audio_binary = tf.io.read_file(audio_path)
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, sample_rate = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_stft_npy(wav_paths, frame_length=255, frame_step=1, concat_time=True):
    # Given a list of .wav files, compute STFT features for each, and output a single npy array
    # concatenated across time
    # Using frame_step 1 to only slide the window by 1 each time, extracting STFT from all
    # possible windows over the waveform.

    out_npy = []
    for wav_path in wav_paths:
        wave = decode_audio(wav_path)
        # https://www.tensorflow.org/api_docs/python/tf/signal/stft
        spec = tf.signal.stft(wave, frame_length=frame_length, frame_step=frame_step)  # [time, freq]
        spec = tf.abs(spec).numpy()
        spec = np.log(spec + np.finfo(float).eps)
        if not concat_time:
            spec = spec[None, ...]  # create a separate batch dimension
        out_npy.append(spec)
    out_npy = np.concatenate(out_npy)
    return out_npy


if __name__ == '__main__':

    import glob
    import os

    data_dir = '/path/to/free-spoken-digit-dataset' # replace with your own
    speaker = 'theo'
    marginal_coords = [30, 10]  # for creating 2D marginals; these have the least abs correlation on the training data
    for split in ['train', 'test']:
        save_name = f'stft-split={split}.npy'

        if split == 'train':  # Using the original train/test split; https://github.com/Jakobovski/free-spoken-digit-dataset/#usage
            id_range = range(5, 50)
        else:
            id_range = range(5)

        wav_paths = []
        for i in id_range:
            glob_pat = f'{data_dir}/recordings/*_{speaker}_{i}.wav'
            wav_paths += glob.glob(glob_pat)

        arr = get_stft_npy(wav_paths, frame_length=63, frame_step=1)
        np.save(save_name, arr)
        print('saved to', save_name)

        save_name = 'n=2-' + save_name
        np.save(save_name, arr[:, marginal_coords])
        print('saved to', save_name)

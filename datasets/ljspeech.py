from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import torch
import os
import pickle as pkl

from utils import load_wav_to_torch
import layers
from hparams_ljspeech import create_hparams
hparams = create_hparams()


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.
    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar
    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  metafile = os.path.join(in_dir, 'metadata.csv')
  lines = open(metafile, encoding='utf-8').readlines()
  nlines = len(lines)
  batchsize = int(nlines/10/1000)*1000
  for i, line in enumerate(lines):
    if (i+1) % batchsize == 0:
      print('%d/%d lines submitted ...' % (i+1, nlines))
    parts = line.strip().split('|')
    wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
    text = parts[2]  # normalized text
    #futures.append(_process_utterance(out_dir, wav_path, text))
    futures.append(executor.submit(partial(_process_utterance, out_dir, wav_path, text)))
  print('All lines submitted!')

  return [future.result() for future in tqdm(futures)]

def _process_utterance(out_dir, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.
  This writes the mel feature to disk and returns a tuple to write
  to the mels.txt file.
  Args:
    out_dir: The directory to write the spectrograms into
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file
  Returns:
    A (melspec, n_frames, text) tuple to write to mels.txt
  '''

  fid = os.path.splitext(os.path.basename(wav_path))[0]

  # case if mel already exist
  if hparams.mel_data_type == 'numpy':
    mel_path = os.path.join(out_dir, '{}.npy'.format(fid))
    if os.path.isfile(mel_path):
      melspec = torch.from_numpy(np.load(mel_path))
      return (mel_path, melspec.shape[1], text)
  elif hparams.mel_data_type == 'torch':
    mel_path = os.path.join(out_dir, '{}.pt'.format(fid))
    if os.path.isfile(mel_path):
      #melspec = torch.load(mel_path) # pkl is faster than torch here
      with open(mel_path, 'rb') as f:
        melspec = pkl.load(f)
      return (mel_path, melspec.shape[1], text)

  # case if mel has not been generated
  audio, sampling_rate = load_wav_to_torch(wav_path)
  if sampling_rate != hparams.sampling_rate:
    raise ValueError("{}: {} SR doesn't match target {} SR".format(
      wav_path, sampling_rate, hparams.sampling_rate))
  audio_norm = audio / hparams.max_wav_value # dim: #samples
  audio_norm = audio_norm.unsqueeze(0) # dim: 1 X #samples
  audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
  stft = layers.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)
  melspec = stft.mel_spectrogram(audio_norm)
  melspec = torch.squeeze(melspec, 0)

  if hparams.mel_data_type == 'numpy':
    np.save(mel_path, melspec.numpy(), allow_pickle=False)
  elif hparams.mel_data_type == 'torch':
    #torch.save(melspec, mel_path) # pkl is faster than torch here
    with open(mel_path, 'wb') as f:
      pkl.dump(melspec, f, protocol=pkl.HIGHEST_PROTOCOL)

  # Return a tuple describing this training example:
  return (mel_path, melspec.shape[1], text)


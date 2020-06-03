import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams_soe import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from train import load_model

from text import text_to_sequence

from utils import load_wav_to_torch
from scipy.io.wavfile import write
import os
import time

from sklearn.manifold import TSNE
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pylab as plt
#import IPython.display as ipd
from tqdm import tqdm

# additional packages
import glob
import random

# prepare data for scatter plot
colors = 'r','b','g','y'
labels = 'neutral','happy','angry','sad'

hparams = create_hparams()
hparams.sampling_rate = 22050
hparams.max_decoder_steps = 1000

stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

def load_mel(path):
  audio, sampling_rate = load_wav_to_torch(path)
  if sampling_rate != hparams.sampling_rate:
    raise ValueError("{} SR doesn't match target {} SR".format(
      sampling_rate, stft.sampling_rate))
  audio_norm = audio / hparams.max_wav_value
  audio_norm = audio_norm.unsqueeze(0)
  audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
  melspec = stft.mel_spectrogram(audio_norm)
  #melspec = torch.squeeze(melspec, 0) # zge: add back
  melspec = melspec.cuda() # original, but why? -> this makes no difference
  return melspec

# load waveglow model
waveglow_name = "waveglow_model_628000"
waveglow_path = "/home/zge/Work/Projects/evs/experiments/SpeechSynthesis/Tacotron2/synth_models"
waveglow_path = os.path.join(waveglow_path, waveglow_name)
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda()

# load filepaths
path = './filelists/soe/soe_wav_test.txt'
with open(path, encoding='utf-8') as f:
  filepaths_and_text = [line.strip().split("|") for line in f]

filepaths_per_cat = [[] for _ in range(len(labels))]
for i, cat in enumerate(list(labels)):
  print('cat {}: {}'.format(i, cat))
  for line in filepaths_and_text:
    emotion_idx = int(line[-1])
    filepaths_per_cat[emotion_idx].append(line)

for i, filepaths in enumerate(filepaths_per_cat):
  print('cat {} ({}): {}'.format(i, labels[i], len(filepaths)))

# remove digits and letters from neutral (causing confusion in clustering)
filepaths_neutral = [line for line in filepaths_per_cat[0] if 'normal' in line[0]]

# get filepaths from neutral cat and the #files to be select as subset
filepaths_neutral_sub = filepaths_neutral[:]
length_max = max([len(filepaths_per_cat[i]) for i in range(1,4)])
print('select {} out of {} in neutral'.format(length_max, len(filepaths_neutral)))

# randomly select a subset from neutral cat with similar #files, compared to other cats
random.seed(0)
random.shuffle(filepaths_neutral_sub)
filepaths_neutral_sub = filepaths_neutral_sub[:length_max]

# combine and get new filepaths and text
filepaths_and_text = filepaths_neutral_sub + filepaths_per_cat[1] + \
                     filepaths_per_cat[2] + filepaths_per_cat[3]

# get checkpoint paths
checkpoint_paths = sorted(glob.glob('outdir/soe/train7/checkpoint_*'))
checkpoint_paths = [ckpt_path for ckpt_path in checkpoint_paths if '.png' not in ckpt_path]

for checkpoint_path in checkpoint_paths:

  print('ckpt: {} ...'.format(checkpoint_path))
  tag = '_'.join(checkpoint_path.split('_')[:-1])
  figname_scatter = '_'.join([tag, 'scatter', 'test.png'])
  figname_tsne = '_'.join([tag, 'tsne', 'test.png'])

  if os.path.isfile(figname_scatter) and os.path.isfile(figname_tsne):
    print('skip ckpt: {} ...'.format(checkpoint_path))
  else:
    print('processing ckpt: {} ...'.format(checkpoint_path))

    #checkpoint_path = 'outdir/soe/train7/checkpoint_90000_0.2760'
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.eval()

    model.eval()
    prosody_outputs = []
    emotions = []
    mus = []
    zs = []

    for audio_path, _, _, emotion in tqdm(filepaths_and_text):
      melspec = load_mel(audio_path)
      prosody, mu, _, z = model.vae_gst(melspec)
      prosody_outputs.append(prosody.squeeze(1).cpu().data)
      mus.append(mu.cpu().data)
      zs.append(z.cpu().data)
      emotions.append(int(emotion))

    prosody_outputs = torch.cat(prosody_outputs, dim=0)
    emotions = np.array(emotions)  # list이면 안됨 -> ndarray
    mus = torch.cat(mus, dim=0)
    zs = torch.cat(zs, dim=0)

    data_x = mus.data.numpy()
    data_y = emotions

    idx = sorted(np.argsort(np.std(data_x, 0))[::-1][:2])
    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(data_x[data_y==i,idx[0]], data_x[data_y==i,idx[1]], c=c,
                    label=label, alpha=0.5)
    plt.xlabel('dim {}'.format(idx[0]))
    plt.ylabel('dim {}'.format(idx[1]))
    plt.title('scatter plot of mus with emotion color labels, ' +
            'dim: {}, show {}, {}'.format(mus.shape[1], idx[0], idx[1]))

    axes = plt.gca()
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(figname_scatter)
    plt.close()

    data_x = mus.data.numpy()
    data_y = emotions

    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10, 10))
    for i, (c, label) in enumerate(zip(colors, labels)):
      plt.scatter(tsne_all_data[tsne_all_y_data == i, 0], tsne_all_data[tsne_all_y_data == i, 1], c=c, label=label,
                  alpha=0.5)
    plt.xlabel('dim 0'), plt.ylabel('dim 1')
    plt.title('t-SNE plot of mus with emotion color labels, dim: {}'.format(data_x.shape[1]))

    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(figname_tsne)
    plt.close()

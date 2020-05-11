# prepare emotion features
#
# example:
# python prep_emotion.py \
#   --input-dir /data/evs/SOE/renamed \
#   --model-file  models/emr_datatang_4emo_trimmed/best_model.pkl \
#   --stats-emr models/emr_datatang_4emo_trimmed/stats_emr \
#   --cfg-file emr/htk.cfg \
#   --out-dim 64 \
#   --gpu-device 1

import os, sys
import argparse
import torch
import numpy as np
import glob

# include HTK libraries (assume in '~/bin')
os.environ['HTK_ROOT'] = os.path.expanduser('~/bin')
os.environ['PATH'] = os.environ['HTK_ROOT'] + ':' + os.environ['PATH']

# include audio emotion recognition directory into search path
emrpath = os.path.join(os.getcwd(), 'emr')
sys.path.extend([emrpath])
from models_embed import Layered_RNN

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', type=str, required=True,
                      help='input directory for audio wav files')
  parser.add_argument('--model-file', type=str, required=True)
  parser.add_argument('--stats-emr', type=str, required=True)
  parser.add_argument('--cfg-file', type=str, default='emr/htk.cfg')
  parser.add_argument('--out-dim', type=int, default=64)
  parser.add_argument('--gpu-device', type=int, default=0)
  return parser.parse_args()

def main():

  # runtime mode
  args = parse_args()

  # # interactive mode
  # args = argparse.ArgumentParser()
  # args.input_dir = '/data/evs/SOE/renamed'
  # args.model_file = 'models/emr_datatang_4emo_trimmed/best_model.pkl'
  # args.stats_emr = 'models/emr_datatang_4emo_trimmed/stats_emr'
  # args.cfg_file = 'emr/htk.cfg'
  # args.out_dim = 64
  # args.gpu_device = 1

  print('input dir: {}'.format(args.input_dir))
  print('model file: {}'.format(args.model_file))
  print('stats emr: {}'.format(args.stats_emr))
  print('config file: {}'.format(args.cfg_file))
  print('output dim: {}'.format(args.out_dim))
  print('gpu device: {}'.format(args.gpu_device))

  # set current GPU device
  torch.cuda.set_device(args.gpu_device)
  print('current GPU: {}'.format(torch.cuda.current_device()))

  # load audio emotion recognition model
  model = Layered_RNN(out_dim=args.out_dim, stats_emr=args.stats_emr).type(
    torch.cuda.FloatTensor)
  model.load_state_dict(torch.load(args.model_file))
  model.eval()

  # get wav list
  wav_paths = os.path.join(args.input_dir, '**', '*.wav')
  wavs = sorted(glob.glob(wav_paths, recursive=True))

  nwavs = len(wavs)
  for i, wav in enumerate(wavs):

    if os.path.isfile(wav):

      # check if emotion features already generated
      file = {'preds': wav.replace('.wav', '_prob.npy'),
              'embeds': wav.replace('.wav', '_embed.npy'),
              'preds_g': wav.replace('.wav', '_prob_g.npy'),
              'embeds_g': wav.replace('.wav', '_embed_g.npy')}
      flags = [os.path.isfile(file['preds']),
               os.path.isfile(file['embeds']),
               os.path.isfile(file['preds_g']),
               os.path.isfile(file['embeds_g'])]
      if all(flags):
        print('{}: emotion feature generated, skip!'.format(os.path.basename(wav)))
        continue

      # preds, preds_g: #frames X #ems, 1 X #ems
      # embeds, embeds_g: #frames X out_dim, 1 X out_dim
      outputs = model.get_embed(wav, cfg_file=args.cfg_file)
      preds, embeds, preds_g, embeds_g = outputs

      # save emotion features
      np.save(file['preds'], preds)
      np.save(file['embeds'], embeds)
      np.save(file['preds_g'], preds_g)
      np.save(file['embeds_g'], embeds_g)
      print("({}/{}) {}: generated emb prob {}, embed {}".format(
        i+1, nwavs, os.path.basename(wav), preds.shape, embeds.shape))

    else:
      raise Exception('{} does not exist!'.format(wav))

if __name__ == '__main__':
  main()
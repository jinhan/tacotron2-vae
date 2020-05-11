import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import ljspeech, soe
from hparams import create_hparams

hparams = create_hparams()


def preprocess_ljspeech(args):
  in_dir = os.path.join(args.base_dir, 'LJSpeech-1.1')
  out_dir = os.path.join(in_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_soe(args):
  in_dir = os.path.join(args.base_dir, 'SOE', 'renamed')
  out_dir = os.path.join(in_dir, args.output).rstrip(os.sep)
  os.makedirs(out_dir, exist_ok=True)
  metadata = soe.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'mels.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[1] for m in metadata])
  #hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  hours = frames * hparams.hop_length / hparams.sampling_rate / 3600
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length (#words in normalized text):  %d' %
        max(len(m[2]) for m in metadata))
  print('Max output length: %d (#frames in spectrogram)' % max(m[1] for m in metadata))


def parse_args():
  usage = 'generate mel feature'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--base-dir', default=os.getcwd())
  parser.add_argument('--output', default='')
  parser.add_argument('--dataset', required=True, choices=['ljspeech', 'soe'])
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  return parser.parse_args()


def main():

  # runtime mode
  args = parse_args()

  # # interactive mode
  # args = argparse.ArgumentParser()
  # args.base_dir = '/data/evs'
  # args.output = '' # 'mels' (ljspeech), or '' (soe)
  # args.dataset = 'soe' # 'ljspeech', or 'soe'
  # args.num_workers = cpu_count()

  if args.dataset == 'ljspeech':
    preprocess_ljspeech(args)
  elif args.dataset == 'soe':
    preprocess_soe(args)


if __name__ == "__main__":
  main()
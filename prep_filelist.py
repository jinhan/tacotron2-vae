# Prepare file lists and save in the 'filelist' directory
#
# Zhenhao Ge, 2020-04-24

import argparse
import os
import glob
import random

def parse_args():

  usage = 'usage: prepare file lists'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-i', '--input-dir', type=str,
                     help='directory for input data')
  parser.add_argument('-o', '--output-dir',type=str,
                     help='directory for output file lists')
  parser.add_argument('-d', '--dataset', required=True,
                      choices=['ljspeech', 'soe'])
  parser.add_argument('-r', 'ratio', type=str, default='8:1:1',
                     help='train/valid/test ratios')
  parser.add_argument('--metafile', type=str, default='',
                      help='optional metadata file')
  parser.add_argument('--emo-labels', type=str, default=
                    'neutral:0,happy:1,angry:2,sad:3',
                     help='emotion labels to be attached')
  parser.add_argument('--seed', type=int, default=0,
                     help=('random seed to split filelist into train',
                           ' validation and test'))

  return parser.parse_args()

def read_meta_ljspeech(metafile):
  """read ljspeech metadata.csv to get filename to text mapping"""

  lines = open(metafile).readlines()
  filename2text = {}
  for i, line in enumerate(lines):
    filename, text, text_normed = line.rstrip().split('|')
    basename = os.path.splitext(os.path.basename(filename))[0]
    filename2text[basename] = text_normed
  return filename2text

def prep_ljspeech(args):
  """prepare lines in filelist for ljspeech"""

  wavfiles = glob.glob(os.path.join(args.input_dir,'*.wav'))
  spk_id, emo_id = 0, 0
  filename2text = read_meta_ljspeech(args.metafile)

  lines = ['' for _ in range(len(wavfiles))]
  for i, wavfile in enumerate(wavfiles):
    basename = os.path.splitext(os.path.basename(wavfile))[0]
    text = filename2text[basename]
    lines[i] = '|'.join([wavfile, text, str(spk_id), str(emo_id)])

  return lines

def split(lines, ratio='8:1:1', seed=0):
  """split lines by ratios for train/valid/test"""

  # get line indecies to split
  ratios = [float(r) for r in ratio.split(':')]
  percents = [sum(ratios[:i+1]) for i in range(len(ratios))]
  percents = [p/sum(ratios) for p in percents]
  nlines = len(lines)
  idxs = [0] + [int(p*nlines) for p in percents]

  # shuffle lines with fixed random seed
  random.seed(seed)
  random.shuffle(lines)

  cats = ['train', 'valid', 'test']
  flist = {cat:sorted(lines[idxs[i]:idxs[i + 1]]) for (cat,i)
           in zip(cats, range(len(cats)))}

  return flist

def write_flist(flist, dataset, output_dir, verbose=True):
  for cat in flist.keys():
    listfile = '{}_{}.txt'.format(dataset, cat)
    listpath = os.path.join(output_dir, listfile)
    open(listpath, 'w').write('\n'.join(flist[cat]))
    print('wrote list file: {}'.format(listpath))

def main():

  # runtime
  args = parse_args()

  # # debug
  # args = argparse.ArgumentParser()
  # args.input_dir = r'data/LJSpeech-1.1/wavs'
  # args.output_dir = r'filelists'
  # args.dataset = 'ljspeech'
  # args.metafile = r'data/LJSpeech-1.1/metadata.csv'
  # args.ratio = '8:1:1'
  # args.seed = 0

  if args.dataset == 'ljspeech':
    lines = prep_ljspeech(args)

  flist = split(lines, ratio=args.ratio, seed=args.seed)
  write_flist(flist, args.dataset, args.output_dir)

if __name__ == "__main__":
  main()

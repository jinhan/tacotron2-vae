# Prepare file lists and save in the 'filelist' directory
#
# Zhenhao Ge, 2020-04-24

import argparse
import os
import glob
import random
import wave
import contextlib

def convert_symbol(text, l1, l2, quote='"'):
  """convert symbol l1 to l2 if inside quote"""
  text2 = ''
  inside = False
  for c in text:
    if c == quote:
      inside = not inside
    elif c == l1:
      if inside:
        text2 += l2
      else:
        text2 += l1
    else:
       text2 += c
  return text2

def csv2dict(csvname, delimiter=','):
  """extract rows in csv file to a dictionary list"""
  lines = open(csvname, 'r').readlines()
  header = lines[0].rstrip().split(delimiter)
  lines = lines[1:]
  nlines = len(lines)

  dict_list = [{} for _ in range(nlines)]
  for i, line in enumerate(lines):
    line2 = convert_symbol(line.rstrip(), delimiter, '|')
    items = line2.split(delimiter)
    items2 = [s.replace('|', delimiter) for s in items]
    dict_list[i] = {k:items2[j] for j,k in enumerate(header)}

  return dict_list

def read_meta_ljspeech(metafile):
  """read ljspeech metadata.csv to get filename to text mapping"""

  lines = open(metafile).readlines()
  filename2text = {}
  for i, line in enumerate(lines):
    filename, text, text_normed = line.rstrip().split('|')
    basename = os.path.splitext(os.path.basename(filename))[0]
    filename2text[basename] = text_normed
  return filename2text

def read_meta_soe(metafile):
  """read soe metadata.csv to get filename to meta mapping"""
  wavfiles = csv2dict(metafile)
  return {f['fid']:{k:v for (k,v) in f.items() if k!='fid'} for f in wavfiles}

def parse_emo_label(emo_labels):
  """parse emotion labels to emotion dictionary"""
  emo_dict = {}
  for label in emo_labels.split(','):
    k,v = label.split(':')
    emo_dict[k] = int(v)
  return emo_dict

def prep_ljspeech(args, verbose=False):
  """prepare lines in filelist for ljspeech"""

  wavfiles = glob.glob(os.path.join(args.input_dir, '*.wav'))
  wavfiles = sorted(wavfiles, key=lambda f: os.path.basename(f))
  nwavfiles = len(wavfiles)
  if verbose:
    print('# of wavs in {}: {}'.format(args.input_dir, nwavfiles))

  spk_id, emo_id = 0, 0
  filename2text = read_meta_ljspeech(args.metafile)

  filename2dur = {}
  batchsize = 1000
  for i, wavfile in enumerate(wavfiles):
    if verbose and (i%batchsize == 0):
      print('getting duration for file {} ~ {}'.format(
        i, min(i+batchsize, nwavfiles)))
    basename = os.path.splitext(os.path.basename(wavfile))[0]
    filename2dur[basename] = wav_duration(wavfile)

  lines = ['' for _ in range(len(wavfiles))]
  for i, wavfile in enumerate(wavfiles):
    basename = os.path.splitext(os.path.basename(wavfile))[0]
    text, dur = filename2text[basename], filename2dur[basename]
    if args.include_emb:
      embfile = wavfile.replace('/wavs/', '/embs/')
      embfile = embfile.replace('.wav', '.npy')
      lines[i] = '|'.join([wavfile, embfile, text, str(dur), str(spk_id), str(emo_id)])
    else:
      lines[i] = '|'.join([wavfile, text, str(dur), str(spk_id), str(emo_id)])

  return lines

def prep_soe(args, verbose=False):
  """prepare lines in filelist for soe"""

  spk_dict = {'F1': 0, 'M1': 1}
  emo_dict = parse_emo_label(args.emo_labels)
  dur_range = (1, 11)

  # get wav files and sort by filename
  wavfiles = glob.glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True)
  wavfiles = sorted(wavfiles, key=lambda f:os.path.basename(f))
  nwavfiles = len(wavfiles)
  if verbose:
    print('# of wavs in {}: {}'.format(args.input_dir, nwavfiles))
  filename2meta = read_meta_soe(args.metafile)

  lines = []
  for wavfile in wavfiles:
    basename = os.path.splitext(os.path.basename(wavfile))[0]
    dur = float(filename2meta[basename]['dur'])
    if dur >= dur_range[0] and dur < dur_range[1]:
      text = filename2meta[basename]['text']
      spk_id = spk_dict[filename2meta[basename]['speaker']]
      emo_id = emo_dict[filename2meta[basename]['cat']]
      if args.include_emb:
        embfile = wavfile.replace('.wav', '_embed.npy')
        if args.emo_ver != None:
          embfile = embfile.replace('_embed.npy', '_v{}_embed.npy'.format(args.emo_ver))
        lines.append('|'.join([wavfile, embfile, text, str(dur), str(spk_id), str(emo_id)]))
      else:
        lines.append('|'.join([wavfile, text, str(dur), str(spk_id), str(emo_id)]))

  return lines

def split(lines, ratio='8:1:1', seed=0, ordered=True):
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
  if ordered:
    flist = {cat:sorted(lines[idxs[i]:idxs[i + 1]]) for (cat,i)
             in zip(cats, range(len(cats)))}
  else:
    flist = {cat:lines[idxs[i]:idxs[i + 1]] for (cat,i)
             in zip(cats, range(len(cats)))}
  return flist

def wav_duration(filename):
  """get wav file duration in seconds"""
  with contextlib.closing(wave.open(filename,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    #print(duration)
  return duration

def sort_by_dur(lines, reverse=True):
  """sort lines by duration (as one item in the line)"""
  lines_sorted = sorted(lines, key=lambda line:float(line.split('|')[-3]),
                        reverse=reverse)
  return lines_sorted

def write_flist(flist, dataset, output_dir, include_emb=False, emo_ver=None,
                verbose=True):

  # update output directory to include dataset name
  output_dir = os.path.join(output_dir, dataset)
  os.makedirs(output_dir, exist_ok=True)

  for cat in flist.keys():
    if include_emb:
      listfile = '{}_wav-emo_{}.txt'.format(dataset, cat)
      if emo_ver != None:
        listfile = listfile.replace('-emo', '-emo_v{}'.format(emo_ver))
    else:
      listfile = '{}_wav_{}.txt'.format(dataset, cat)
    listpath = os.path.join(output_dir, listfile)
    open(listpath, 'w').write('\n'.join(flist[cat]))
    if verbose:
      print('wrote list file: {}'.format(listpath))

def parse_args():

  usage = 'usage: prepare file lists'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-i', '--input-dir', type=str,
                      help='directory for input data')
  parser.add_argument('-o', '--output-dir',type=str,
                      help='directory for output file lists')
  parser.add_argument('-d', '--dataset', required=True,
                      choices=['ljspeech', 'soe'])
  parser.add_argument('-r', '--ratio', type=str, default='8:1:1',
                      help='train/valid/test ratios')
  parser.add_argument('--metafile', type=str, default='',
                      help='optional metadata file')
  parser.add_argument('--include-emb', action="store_true",
                      help='include emotion embedding file if enabled')
  parser.add_argument('--emo-labels', type=str, default=
                      'neutral:0,happy:1,angry:2,sad:3',
                      help='emotion labels to be attached')
  parser.add_argument('--emo-ver', type=int, default=None,
                      help='emotion embedding version')
  parser.add_argument('--seed', type=int, default=0,
                      help=('random seed to split filelist'
                            ' into training, validation and test'))
  parser.add_argument('--ordered', action='store_true',
                      help=('file list will be sorted after randomization'
                            ' if specified'))

  return parser.parse_args()

def main():

  # runtime mode
  args = parse_args()

  # # interactive mode
  # args = argparse.ArgumentParser()
  # args.ratio = '8:1:1'
  # args.seed = 0
  # args.ordered = False
  # args.output_dir = r'filelists'
  #
  # # ljspeech
  # args.input_dir = r'data/LJSpeech-1.1/wavs'
  # args.dataset = 'ljspeech'
  # args.metafile = r'data/LJSpeech-1.1/metadata.csv'
  # args.include_emb = False
  # args.emo_ver = None
  #
  # # soe
  # args.input_dir = r'data/SOE/renamed/F1'
  # args.dataset = 'soe'
  # args.metafile = r'data/SOE/renamed/metadata-F1-raw.csv'
  # args.include_emb = True
  # args.emo_labels = r'neutral:0,happy:1,angry:2,sad:3'
  # args.emo_ver = 0

  if args.dataset == 'ljspeech':
    lines = prep_ljspeech(args)
  elif args.dataset == 'soe':
    lines = prep_soe(args)
  else:
    raise Exception('{} is not supported'.format(args.dataset))

  # split lines into train/valid/test
  flist = split(lines, ratio=args.ratio, seed=args.seed, ordered=args.ordered)

  # sort each set by duration (for efficient batching)
  for cat in flist.keys():
    flist[cat] = sort_by_dur(flist[cat], reverse=True)

  # write out file lists
  write_flist(flist, args.dataset, args.output_dir, args.include_emb, args.emo_ver)

if __name__ == "__main__":
  main()

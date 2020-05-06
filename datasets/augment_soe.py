# Take in the original soe filelists and add weights to the files from the
# extreme sub-category
#
# Example:
#   python augment_soe.py \
#     --infile filelists/soe_train.txt \
#     --outfile filelists/soe_train_3x.txt \
#     --multi-factor 3 \
#     --verbose
#
# Zhenhao Ge, 2020-05-04

import argparse
import random

def multiply_keyword_line(lines, keyword, r, verbose=False):
  lines_with_keyword = [line for line in lines if keyword in line]
  lines_without_keyword = [line for line in lines if keyword not in line]
  lines = lines_with_keyword*r + lines_without_keyword
  if verbose:
    L1, L2 = len(lines_with_keyword), len(lines_without_keyword)
    print('#lines with/without keyword {}: {}/{}'.format(keyword, L1, L2))
    L = len(lines)
    print('Total #lines after {}x of lines with keyword: {}'.format(r, L))
  return lines

def parse_args():
  usage = ('take in the original SOE filelist and add weights to the files'
           ' from the extreme sub-category')
  parser = argparse.ArgumentParser(usage)
  parser.add_argument('-i', '--infile', type=str, help='input filelist')
  parser.add_argument('-o', '--outfile', type=str, help='output filelist')
  parser.add_argument('-k', '--keyword', type=str, default='extreme', help='keyword')
  parser.add_argument('-x', '--multi-factor', type=int, default=2,
                      help='multiply factor')
  parser.add_argument('-s', '--seed', type=int, default=0,
                      help='seed to randomize the filelist')
  parser.add_argument('-v', '--verbose', action='store_true', help='flag to show info')

  return parser.parse_args()

def main():

  # runtime mode
  args = parse_args()

  # # interactive mode
  # args = argparse.ArgumentParser()
  # args.infile = 'filelists/soe_test.txt'
  # args.outfile = 'filelists/soe_test_3x.txt'
  # args.keyword = 'extreme'
  # args.multi_factor = 3
  # args.seed = 0
  # args.verbose = True

  # print out input arguments
  print('input file list: {}'.format(args.infile))
  print('output file list: {}'.format(args.outfile))
  print('keyword: {}'.format(args.keyword))
  print('mutil-factor: {}'.format(args.multi_factor))
  print('seed: {}'.format(args.seed))
  print('verbose: {}'.format(args.verbose))

  # read lines from input file list
  lines = open(args.infile, 'r').readlines()
  lines = [line.rstrip() for line in lines]

  # multiply the lines with keyword
  lines = multiply_keyword_line(lines, args.keyword, args.multi_factor, args.verbose)

  # randomize lines with fix seed
  random.seed(args.seed)
  random.shuffle(lines)

  open(args.outfile, 'w').write('\n'.join(lines))
  if args.verbose:
    print('wrote list file: {}'.format(args.outfile))

if __name__ == '__main__':
  main()
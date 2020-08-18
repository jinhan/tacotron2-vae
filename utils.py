import numpy as np
from scipy.io.wavfile import read
import librosa
import torch
import os
import csv

max_wav_value=32768.0

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    #mask = (ids < lengths.unsqueeze(1)).byte() #deprecated
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def str2bool(v):
    return v.lower() in ('true', '1')


def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)


def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)


def get_kl_weight(af, lag, k, x0, upper, constant, nsteps=250000):
    kl_weights = [0 for _ in range(nsteps)]
    steps = list(range(nsteps))
    for i, step in enumerate(steps):
        if step >= lag:
            if af == 'logistic':
                kl_weights[i] = float(upper/(1+np.exp(-k*(step-x0))))
            elif af == 'linear':
                #kl_weights[i] = min(upper, step/x0)
                kl_weights[i] = min(upper, (step-lag)/x0)
            elif af == 'constant':
                kl_weights[i] = constant
        else:
          kl_weights[i] = 0
    return kl_weights


def get_text_padding_rate(input_lengths, top_n=3):
    batch_size = input_lengths.size(0)
    max_len = int(max(input_lengths))
    mean_len = float(sum(input_lengths)) / len(input_lengths)
    padding_rate = 1 - mean_len / max_len
    top_len = input_lengths[:min(batch_size, top_n)]
    top_len = '-'.join([str(int(l)) for l in top_len])
    return padding_rate, max_len, top_len


def get_mel_padding_rate(gate_padded, top_n=3):
    batch_size, max_len = gate_padded.shape
    padded_zeros = torch.sum(gate_padded, 1) - 1
    padding_rate = float(sum(padded_zeros) / gate_padded.numel())
    min_padded_zeros = sorted(padded_zeros)[:min(batch_size, top_n)]
    top_len = [max_len-i for i in min_padded_zeros]
    top_len = '-'.join([str(int(l)) for l in top_len])
    return padding_rate, max_len, top_len


def add_rand_noise(key_values, noise_range=(-0.5,0.5), seed=0):
    n = len(key_values)
    np.random.seed(seed)
    values = np.random.rand(n)
    lower, upper = noise_range
    noises = [v * (upper-lower) + lower for v in values]
    key_values_with_noise = [d+n for (d,n) in zip(key_values, noises)]
    return key_values_with_noise


def sort_with_noise(key_values, key_values_noisy, reverse=True):
    """order clean key values with the order sorted by noisy key values"""
    idx = [i[0] for i in sorted(enumerate(key_values_noisy),
                                key=lambda x:x[1], reverse=reverse)]
    key_values_resorted = [key_values[i] for i in idx]
    return key_values_resorted

def get_key_values(filelist, filelist_cols):
    if 'dur' in filelist_cols:
        key = 'dur'
        key_idx = filelist_cols.index(key)
        key_values = [float(line[key_idx]) for line in filelist]
    else:
        key = 'text'
        key_idx = filelist_cols.index(key)
        key_values = [len(line[key_idx]) for line in filelist]
    return key_values, key


def get_batch_sizes(filelist, filelist_cols, batch_size):
    key_values, key = get_key_values(filelist, filelist_cols)
    values_sorted = sorted(key_values, reverse=True)
    batch_len_max_mean = np.mean(values_sorted[:batch_size])
    batch_capacity = batch_size * batch_len_max_mean
    # get batches where each batch gets full capacity
    batch_sizes = []
    remaining = key_values[:]
    while len(remaining) > 0:
        bs = 1
        while np.max(remaining[:min(bs, len(remaining))]) * bs <= batch_capacity:
            bs += 1
        batch_size_current = min(bs-1, len(remaining))
        batch_sizes.append(batch_size_current)
        remaining = remaining[batch_size_current:]
    return batch_sizes


def permute_filelist(filelist, filelist_cols, seed=0, permute_opt='rand',
                     local_rand_factor=0.1):
    if permute_opt == 'rand':
        filelist_permuted = filelist[:]
        np.random.seed(seed)
        np.random.shuffle(filelist_permuted)
        key, noise_range = '', (0,0)
    elif permute_opt == 'semi-sort':
        key_values, key = get_key_values(filelist, filelist_cols)
        idx_value_sorted = sorted(enumerate(key_values), key=lambda x:x[1], reverse=True)
        idxs_sorted = [x[0] for x in idx_value_sorted]
        filelist_sorted = [filelist[i] for i in idxs_sorted]
        values_sorted = [x[1] for x in idx_value_sorted]
        values_range = np.floor(values_sorted[-1]), np.ceil(values_sorted[0])
        noise_upper = (values_range[1] - values_range[0]) * local_rand_factor
        noise_range = -noise_upper/2, noise_upper/2
        values_sorted_noisy = add_rand_noise(values_sorted, noise_range, seed=seed)
        filelist_permuted = sort_with_noise(filelist_sorted, values_sorted_noisy)
        # # plot to verify semi-sorted order
        # keys_permuted = [len(line[key_idx].split()) for line in filelist_permuted]
        # plt.plot(keys_permuted), plt.savefig('verify.png')
    return filelist_permuted, (key, noise_range)

# import matplotlib.pyplot as plt
# filelist_permuted = permute_filelist(filelist, filelist_cols, seed,
#     permute_opt='semi-sort', local_rand_factor=0.1)
# keys_permuted = [float(line[key_idx]) for line in filelist_permuted]
# fig = plt.figure()
# plt.plot(keys_permuted)
# plt.xlabel('file index'), plt.ylabel('duration')
# plt.title('semi-sorted durations with rand noise within +/-{}'.format(
#     noise_upper/2))
# fig.savefig('verify.png'); plt.close()


def batching(filelist, batch_size):
    if isinstance(batch_size, list):
        # loop over various batch sizes
        num_batch_size = len(batch_size)
        filelist_remaining = filelist[:]
        idx = 0
        filelist_batched = []
        while len(filelist_remaining) > batch_size[idx % num_batch_size]:
            batch_size_selected = batch_size[idx % num_batch_size]
            filelist_batched.append(filelist_remaining[:batch_size_selected])
            filelist_remaining = filelist_remaining[batch_size_selected:]
            idx += 1
        if len(filelist_remaining) > 0:
            filelist_batched.append(filelist_remaining)
    else:
        # use fixed batch size
        num_files = len(filelist)
        num_batches = int(num_files / batch_size)
        filelist_batched = [filelist[i * batch_size:(i + 1) * batch_size] for i in
                            range(num_batches)]
        filelist_last = filelist[num_batches * batch_size:]
        filelist_batched += filelist_last
    return filelist_batched

# for i, batch in enumerate(filelist_batched):
#     print('batch {}: size {}'.format(i+1, len(batch)))

def permute_batch_from_batch(filelist_batched, seed=0):
    """permute batch from batched filelist"""
    np.random.seed(seed)
    np.random.shuffle(filelist_batched)
    return filelist_batched

def permute_batch_from_filelist(filelist, batch_size, seed=0):
    """permute batch from filelist with fixed batch size"""
    filelist_batched = batching(filelist, batch_size)
    if len(filelist_batched[-1]) < batch_size:
        filelist_last = filelist_batched[-1]
        filelist_batched = filelist_batched[:-1]
    else:
        filelist_last = []
    np.random.seed(seed)
    np.random.shuffle(filelist_batched)
    filelist_shuffled = flatten_list(filelist_batched)
    filelist_shuffled += filelist_last
    return filelist_shuffled


def dict2col(dct, csvname='filename.csv', order=None, verbose=True):
    keys= list(dct.keys())
    if order == 'ascend': keys = sorted(keys)
    elif order == 'descend': keys = sorted(keys, reverse=True)
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(keys)
        n = len(dct[keys[0]])
        for i in range(n):
            row = [dct[k][i] for k in keys]
            csv_out.writerow(row)
    if verbose:
        print('{} saved!'.format(csvname))


def dict2row(dct, csvname='filename.csv', delimiter=',', order=None, verbose=True):
    keys= list(dct.keys())
    if order == 'ascend': keys = sorted(keys)
    elif order == 'descend': keys = sorted(keys, reverse=True)
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f, delimiter=delimiter)
        for k in keys:
            row = [ k, str(dct[k])]
            csv_out.writerow(row)
    if verbose:
        print('{} saved!'.format(csvname))


def list2csv(lst, csvname='filename.csv', delimiter=',', verbose=True):
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f, delimiter=delimiter)
        n = len(lst)
        for i in range(n):
            csv_out.writerow(lst[i])
    if verbose:
        print('{} saved!'.format(csvname))


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


def csv2dlist(csvname, delimiter=','):
    """extract rows in csv file to a dictionary list"""
    lines = open(csvname, 'r').readlines()
    header = lines[0].rstrip().split(delimiter)
    lines = lines[1:]
    nlines = len(lines)

    dict_list = [{} for _ in range(nlines)]
    for i, line in enumerate(lines):
        line2 = convert_symbol(line.rstrip(), delimiter, '|')
        items = line2.split(delimiter)
        items = [s.replace('|', delimiter) for s in items]
        dict_list[i] = {k:items[j] for j,k in enumerate(header)}

    return dict_list


def dlist2dict(dlist, header=None):
    if not header:
        header = sorted(dlist[0].keys())
    dct = {k:[dlist[i][k] for i in range(len(dlist))] for k in header}
    return dct

def csv2dict(csvname, delimiter=',', header=None):
    dlist = csv2dlist(csvname, delimiter)
    dct = dlist2dict(dlist, header)
    return dct

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


def permute_filelist(filelist, filelist_cols, seed=0, permute_opt='rand',
                     local_rand_factor=0.1):
    if permute_opt == 'rand':
        filelist_permuted = filelist[:]
        np.random.seed(seed)
        np.random.shuffle(filelist_permuted)
        key, noise_range = '', (0,0)
    elif permute_opt == 'semi-sort':
        if 'dur' in filelist_cols:
            key = 'dur'
            key_idx = filelist_cols.index(key)
            key_values = [float(line[key_idx]) for line in filelist]
        else:
            key = 'text'
            key_idx = filelist_cols.index(key)
            key_values = [len(line[key_idx]) for line in filelist]
        keys_idx_value_sorted = [i for i in sorted(
            enumerate(key_values), key=lambda x:x[1], reverse=True)]
        idxs_sorted = [x[0] for x in keys_idx_value_sorted]
        filelist_sorted = [filelist[i] for i in idxs_sorted]
        keys_sorted = [x[1] for x in keys_idx_value_sorted]
        keys_range = np.floor(keys_sorted[-1]), np.ceil(keys_sorted[0])
        noise_upper = (keys_range[1] - keys_range[0]) * local_rand_factor
        noise_range = -noise_upper/2, noise_upper/2
        keys_sorted_noisy = add_rand_noise(keys_sorted, noise_range, seed=seed)
        filelist_permuted = sort_with_noise(filelist_sorted, keys_sorted_noisy)
        # # plot to verify semi-sorted order
        # if 'dur' in filelist_cols:
        #     keys_permuted = [float(line[3]) for line in filelist_permuted]
        # else:
        #     keys_permuted = [len(line[2].split()) for line in filelist_permuted]
        # plt.plot(keys_permuted), plt.savefig('verify.png')
    return filelist_permuted, (key, noise_range)

# import matplotlib.pyplot as plt
# filelist_permuted = permute_filelist(filelist, filelist_cols, seed,
#     permute_opt='semi-sort', local_rand_factor=0.1)
# keys_permuted = [float(line[3]) for line in filelist_permuted]
# fig = plt.figure()
# plt.plot(keys_permuted)
# plt.xlabel('file index'), plt.ylabel('duration')
# plt.title('semi-sorted durations with rand noise within +/-{}'.format(
#     noise_upper/2))
# fig.savefig('verify.png'); plt.close()


def permute_batch(filelist, batch_size, seed=0):
    num_files = len(filelist)
    num_batches = int(num_files/batch_size)
    filelist_batch = [filelist[i*batch_size:(i+1)*batch_size] for i in
                      range(num_batches)]
    np.random.seed(seed)
    np.random.shuffle(filelist_batch)
    filelist_shuffled = flatten_list(filelist_batch)
    filelist_last = filelist[num_batches*batch_size:]
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


def dict2row(dct, csvname='filename.csv', order=None, verbose=True):
    keys= list(dct.keys())
    if order == 'ascend': keys = sorted(keys)
    elif order == 'descend': keys = sorted(keys, reverse=True)
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f)
        for k in keys:
            row = [ k, str(dct[k])]
            csv_out.writerow(row)
    if verbose:
        print('{} saved!'.format(csvname))

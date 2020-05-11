import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import htk_io
import numpy as np
import subprocess

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Reshape a variable of size N, T, D, goes through the forward process, and convert back to N, T, D1
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x, input_sizes_list = None, longest_length = None):
        """
        x is of size (N, T, D)
        """
        if self.module is None:
           assert input_sizes_list is None and longest_length is not None, "input_sizes_list must be None, longest_length must not be None"
           T_max = x.size(1)
           pad_width = longest_length - T_max
           if pad_width > 0:
              x = F.pad(x, pad = (0, 0, 0, pad_width, 0, 0), mode='constant', value = 0)
              return x
           else:
              return x

        assert input_sizes_list is not None, "input_sizes_list must not be None"
        if longest_length is None:
           T_max = input_sizes_list[0]
        else:
           T_max = longest_length

        x = torch.cat([x[i, 0 : input_sizes_list[i]] for i in range(len(input_sizes_list))], dim = 0)    # x: T_sum, D
        x = self.module(x)       # pass the module processing
        start = 0
        out = []
        for length in input_sizes_list:
           x_i = x[start : start + length]
           num_pad = T_max - length
           if num_pad > 0:
              x_i = F.pad(x_i, pad = (0, 0, 0, num_pad), mode='constant', value = 0)
           out.append(x_i)
           start += length
        out = torch.stack(out, dim = 0)     # N, T, D

        return out

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True, batch_first = True)
        self.bidirectional = bidirectional

    def forward(self, x, input_sizes_list):
        if self.batch_norm is not None:
            x = self.batch_norm(x, input_sizes_list)
        x = pack_padded_sequence(x, input_sizes_list, batch_first = True)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first = True)
        x = x.contiguous()

        if self.bidirectional:
           x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)

        return x


class Layered_RNN(nn.Module):
    def __init__(self, rnn_input_size=123, rnn_type=nn.LSTM, rnn_hidden_size=512, nb_layers=3, bidirectional=True, batch_norm=True, out_dim=512, num_classes=4, stats_emr='stats_emr'):
        super(Layered_RNN, self).__init__()

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=False)
        rnns.append(rnn)

        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=batch_norm)
            rnns.append(rnn)

        self.rnns = nn.ModuleList(rnns)
        self.fc = nn.Linear(rnn_hidden_size, out_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = SequenceWise(nn.BatchNorm1d(rnn_hidden_size))
        self.cls = nn.Linear(out_dim, num_classes)
        self.pad_layer = SequenceWise(None)

        self.global_mean_emr, self.global_var_emr = self.read_mv(stats_emr)

    def read_mv(self, stat):
        mean_flag = var_flag = False
        m = v = None
        with open(stat) as s:
            for line in s:
                line = line.strip()
                if len(line) < 1: continue
                if "MEAN" in line:
                    mean_flag = True
                    continue
                if mean_flag:
                    m = list(map(float, line.split()))
                    mean_flag = False
                    continue
                if "VARIANCE" in line:
                    var_flag = True
                    continue
                if var_flag:
                    v = list(map(float, line.split()))
                    var_flag = False
                    continue
        return np.array(m, dtype=np.float64), np.array(v, dtype=np.float64)

    def _HCopy(self, cfg, wav):
        output = None
        htk = wav[:-4] + '.htk'
        try:
            output = subprocess.check_output(["HCopy", "-C", cfg, "-T", "1", wav, htk])
        except subprocess.CalledProcessError as e:
            print ('EXC {}'.format(e))
        return output

    def get_embed(self, wav_file, cfg_file):
        htk_feat_file = wav_file[:-4] + '.htk'
        gpu_dtype = torch.FloatTensor
        if self._HCopy(cfg_file, wav_file) is not None:
            io_src = htk_io.fopen(htk_feat_file)
            utt_feat = io_src.getall()
            num_frms = utt_feat.shape[0]
            utt_feat -= self.global_mean_emr
            utt_feat /= (np.sqrt(self.global_var_emr) + 1e-8)
            utt_feat = torch.FloatTensor(utt_feat[np.newaxis, :, :])
            with torch.no_grad():
                utt_feat = Variable(utt_feat).type(gpu_dtype)

            x = utt_feat.cuda()
            for i in range(len(self.rnns)):
                x = self.rnns[i](x, [num_frms])

            x = self.batch_norm(x, [num_frms])            

            x_embed = self.fc(x)
            x_cls = self.cls(x_embed)
            x_prob = nn.Softmax(dim=2)(x_cls)

            x_label = x_prob.max(2)[1].data.cpu().numpy()
            x_prob = np.squeeze(x_prob.data.cpu().numpy())
            x_embed = np.squeeze(x_embed.data.cpu().numpy())

            x_ave = torch.cat([torch.mean(x[0, 0:num_frms, :], dim = 0, keepdim = True)], dim = 0)

            x_embed_g = self.fc(x_ave)
            x_cls_g = self.cls(x_embed_g)
            x_prob_g = nn.Softmax(dim=1)(x_cls_g)

            x_label_g = x_prob_g.max(1)[1].data.cpu().numpy()
            x_prob_g = np.squeeze(x_prob_g.data.cpu().numpy())
            x_embed_g = np.squeeze(x_embed_g.data.cpu().numpy())

        return x_prob, x_embed, x_prob_g, x_embed_g


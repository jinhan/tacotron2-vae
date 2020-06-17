import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from layers import LinearNorm
from CoordConv import CoordConv2d

class VAE_GST(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(hparams)
        self.fc1 = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        self.fc2 = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        self.fc3 = nn.Linear(hparams.z_latent_dim, hparams.E)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, inputs):
        enc_out = self.ref_encoder(inputs) # [batch_size, E//2]
        mu = self.fc1(enc_out) # [batch_size, z_latent_dim]
        logvar = self.fc2(enc_out) # [batch_size, z_latent_dim]
        z = self.reparameterize(mu, logvar) # [batch_size, z_latent_dim]
        style_embed = self.fc3(z) # [batch_size, E]

        return style_embed, mu, logvar, z

    
class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_input_dim*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hparams):
        super().__init__()
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters

        # It is said that using CoordConv as the first layer preserves positional information well.
        # https://arxiv.org/pdf/1811.02122.pdf
        convs = [CoordConv2d(hparams.fp16_run, in_channels=filters[0],
                           out_channels=filters[0 + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1), with_r=True)]
        convs2 = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(1,K)]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i]) for i in range(K)])

        if hparams.vae_input_type == 'mel':
            self.n_input_dim = hparams.n_mel_channels
        elif hparams.vae_input_type == 'emo':
            self.n_input_dim = hparams.emo_emb_dim

        out_channels = self.calculate_channels(self.n_input_dim, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
                          hidden_size=hparams.E // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.contiguous().view(N, 1, -1, self.n_input_dim)  # [N, 1, Ty, n_input_dim]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_input_dim//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_input_dim//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_input_dim//2^K]

        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

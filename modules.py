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
        enc_out = self.ref_encoder(inputs)
        mu = self.fc1(enc_out)
        logvar = self.fc2(enc_out)
        z = self.reparameterize(mu, logvar)
        style_embed = self.fc3(z)

        return style_embed, mu, logvar, z

    
class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hparams):
        super().__init__()
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters
        # 첫번째 레이어로 CoordConv를 사용하는 것이 positional 정보를 잘 보존한다고 함. https://arxiv.org/pdf/1811.02122.pdf
        convs = [CoordConv2d(in_channels=filters[0],
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

        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
                          hidden_size=hparams.E // 2,
                          batch_first=True)
        self.n_mels = hparams.n_mel_channels

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.contiguous().view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
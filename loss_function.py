from torch import nn
import torch
import numpy as np

class Tacotron2Loss_VAE(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss_VAE, self).__init__()
        self.anneal_function = hparams.anneal_function
        self.lag = hparams.anneal_lag
        self.k = hparams.anneal_k
        self.x0 = hparams.anneal_x0
        self.upper = hparams.anneal_upper
        self.constant = hparams.anneal_constant

    def kl_anneal_function(self, anneal_function, lag, step, k, x0, upper, constant):
        if step >= lag:
            if anneal_function == 'logistic':
                return float(upper / (1 + np.exp(-k * (step-x0))))
            elif anneal_function == 'linear':
                return min(upper, (step-lag) / x0)
            elif anneal_function == 'constant':
                return constant
        else:
            return 0

    def forward(self, model_output, targets, step):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, mu, logvar, _, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = self.kl_anneal_function(self.anneal_function, self.lag,
                    step, self.k, self.x0, self.upper, self.constant)

        recon_loss = mel_loss + gate_loss
        weighted_kl_loss = kl_weight * kl_loss
        total_loss = recon_loss + weighted_kl_loss

        return total_loss, recon_loss, kl_loss, kl_weight

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

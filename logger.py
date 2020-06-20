import random
import torch
#import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy, plot_scatter, plot_tsne


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, use_vae):
        super(Tacotron2Logger, self).__init__(logdir, use_vae)
        self.use_vae = use_vae
        #self.dataformat = 'CHW' # default argument for SummaryWriter.add_image
        self.dataformat = 'HWC' # NVIDIA

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     padding_rate_txt, max_len_txt, padding_rate_mel, max_len_mel,
                     iteration, recon_loss='', kl_div='', kl_weight=''):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)
            self.add_scalar("padding.rate.txt", padding_rate_txt, iteration)
            self.add_scalar("max.len.txt", max_len_txt, iteration)
            self.add_scalar("padding.rate.mel", padding_rate_mel, iteration)
            self.add_scalar("max.len.mel", max_len_mel, iteration)
            if self.use_vae:
                self.add_scalar("kl_div", kl_div, iteration)
                self.add_scalar("kl_weight", kl_weight, iteration)
                self.add_scalar("weighted_kl_loss", kl_weight*kl_div, iteration)
                self.add_scalar("recon_loss", recon_loss, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        if self.use_vae:
            _, mel_outputs, gate_outputs, alignments, mus, _, _, emotions = y_pred
        else:
            _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y
        #print('emotion:\n{}'.format(emotions))

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats=self.dataformat)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats=self.dataformat)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats=self.dataformat)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats=self.dataformat)
        if self.use_vae:
            self.add_image(
                "latent_dim (regular)",
                plot_scatter(mus, emotions),
                iteration, dataformats=self.dataformat)
            self.add_image(
                "latent_dim (t-sne)",
                plot_tsne(mus, emotions),
                iteration, dataformats=self.dataformat)

# import io
# import os
# import re
# import librosa
# import argparse
# import numpy as np
# from glob import glob
# from tqdm import tqdm
# import tensorflow as tf
# from functools import partial

# from hparams import hparams
# from models import create_model, get_most_recent_checkpoint
# from audio import save_audio, inv_spectrogram, inv_preemphasis, \
#                   inv_spectrogram_tensorflow
# from utils import plot, PARAMS_NAME, load_json, load_hparams, \
#                   add_prefix, add_postfix, get_time, parallel_run, makedirs, str2bool

# from text.korean import tokenize
# from text import text_to_sequence, sequence_to_text
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from train import load_model
from text import text_to_sequence

from utils import load_wav_to_torch
from scipy.io.wavfile import write
import os
import time
import librosa

# from sklearn.manifold import TSNE
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pylab as plt
# %matplotlib inline
# import IPython.display as ipd
from tqdm import tqdm

class Synthesizer(object):
    def __init__(self):
        super().__init__()
        self.hparams = create_hparams()
        self.hparams.sampling_rate = 16000
        self.hparams.max_decoder_steps = 600

        self.stft = TacotronSTFT(
            self.hparams.filter_length, self.hparams.hop_length, self.hparams.win_length,
            self.hparams.n_mel_channels, self.hparams.sampling_rate, self.hparams.mel_fmin,
            self.hparams.mel_fmax)

    def load_mel(self, path):
        audio, sampling_rate = load_wav_to_torch(path)
        if sampling_rate != self.hparams.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.cuda()
        return melspec

    # def close(self):
    #     tf.reset_default_graph()
    #     self.sess.close()

    def load(self, checkpoint_path, waveglow_path):
        self.model = load_model(self.hparams)
        self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = self.model.eval()

        self.waveglow = torch.load(waveglow_path)['model']
        self.waveglow.cuda()

        path = './filelists/koemo_spk_emo_all_test.txt'
        with open(path, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split("|") for line in f]
        
        base_path = os.path.dirname(checkpoint_path)
        data_path = os.path.basename(checkpoint_path) + '_' + path.rsplit('_', 1)[1].split('.')[0] + '.npz'
        npz_path = os.path.join(base_path, data_path)
        
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            zs = d['zs']
            emotions = d['emotions']
        else:
            emotions = []
            zs = []
            for audio_path, _, _, emotion in tqdm(filepaths_and_text):
                melspec = self.load_mel(audio_path)
                _, _, _, z = self.model.vae_gst(melspec)
                zs.append(z.cpu().data)
                emotions.append(int(emotion))
            emotions = np.array(emotions) # list이면 안됨 -> ndarray
            zs = torch.cat(zs, dim=0).data.numpy()
            d = {'zs':zs, 'emotions':emotions}
            np.savez(npz_path, **d)

        self.neu = np.mean(zs[emotions==0,:], axis=0)
        self.sad = np.mean(zs[emotions==1,:], axis=0)
        self.ang = np.mean(zs[emotions==2,:], axis=0)
        self.hap = np.mean(zs[emotions==3,:], axis=0)

    def synthesize(self, text, path, condition_on_ref, ref_audio, ratios):
        print(ratios)
        sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        inputs = self.model.parse_input(sequence)
        transcript_embedded_inputs = self.model.transcript_embedding(inputs).transpose(1,2)
        transcript_outputs = self.model.encoder.inference(transcript_embedded_inputs)
        print(condition_on_ref)

        if condition_on_ref:
            #ref_audio = '/data1/jinhan/KoreanEmotionSpeech/wav/hap/hap_00000001.wav'
            ref_audio_mel = self.load_mel(ref_audio)
            latent_vector, _, _, _ = self.model.vae_gst(ref_audio_mel)
            latent_vector = latent_vector.unsqueeze(1).expand_as(transcript_outputs)
        
        else: # condition on emotion ratio
            latent_vector = ratios[0] * self.neu + ratios[1] * self.sad + \
                        ratios[2] * self.hap + ratios[3] * self.ang
            latent_vector = torch.FloatTensor(latent_vector).cuda()
            latent_vector = self.model.vae_gst.fc3(latent_vector)

        encoder_outputs = transcript_outputs + latent_vector

        decoder_input = self.model.decoder.get_go_frame(encoder_outputs)
        self.model.decoder.initialize_decoder_states(encoder_outputs, mask=None)
        mel_outputs, gate_outputs, alignments = [], [], []

        while True:
            decoder_input = self.model.decoder.prenet(decoder_input)
            mel_output, gate_output, alignment = self.model.decoder.decode(decoder_input)

            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.hparams.gate_threshold:
                # print(torch.sigmoid(gate_output.data), gate_output.data)
                break
            if len(mel_outputs) == self.hparams.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.model.decoder.parse_decoder_outputs(
                mel_outputs, gate_outputs, alignments)
        mel_outputs_postnet = self.model.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        # print(mel_outputs_postnet.shape)

        with torch.no_grad():
            synth = self.waveglow.infer(mel_outputs, sigma=0.666)
        
        # return synth[0].data.cpu().numpy()
        # path = add_postfix(path, idx)
        # print(path)
        librosa.output.write_wav(path, synth[0].data.cpu().numpy(), 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--sample_path', default="samples")
    parser.add_argument('--text', required=True)
    parser.add_argument('--num_speakers', default=1, type=int)
    parser.add_argument('--speaker_id', default=0, type=int)
    parser.add_argument('--checkpoint_step', default=None, type=int)
    parser.add_argument('--is_korean', default=True, type=str2bool)
    config = parser.parse_args()

    makedirs(config.sample_path)

    synthesizer = Synthesizer()
    synthesizer.load(config.load_path, config.num_speakers, config.checkpoint_step)

    audio = synthesizer.synthesize(
            texts=[config.text],
            base_path=config.sample_path,
            speaker_ids=[config.speaker_id],
            attention_trim=False,
            isKorean=config.is_korean)[0]
import numpy as np
import torch
import torch.utils.data
import os

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
# for individual & batch level permuting
from utils import permute_filelist, permute_batch_from_filelist
# for pre-batching
from utils import batching, get_batch_sizes, permute_batch_from_batch
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, shuffle_plan, hparams, epoch=0,
                 speaker_ids=None, emotion_ids=None):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.shuffle_audiopaths = shuffle_plan['shuffle-audiopath']
        self.shuffle_batches = shuffle_plan['shuffle-batch']
        self.permute_opt = shuffle_plan['permute-opt']
        self.pre_batching = shuffle_plan['pre-batching']
        self.prep_trainset_per_epoch = hparams.prep_trainset_per_epoch
        self.filelist_cols = hparams.filelist_cols
        self.local_rand_factor = hparams.local_rand_factor
        self.include_emo_emb = hparams.include_emo_emb
        self.emo_emb_dim = hparams.emo_emb_dim
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.n_speakers = hparams.n_speakers
        self.n_emotions = hparams.n_emotions
        self.label_type = hparams.label_type
        self.use_vae = hparams.use_vae

        if hparams.override_sample_size:
            self.hop_length = int(np.ceil(hparams.hop_time/1000*hparams.sampling_rate))
            self.win_length = int(np.ceil(hparams.win_time/1000*hparams.sampling_rate))
            self.filter_length = int(2**np.ceil(np.log2(self.win_length)))
        else:
            self.hop_length = hparams.hop_length
            self.win_length = hparams.win_length
            self.filter_length = hparams.filter_length
        self.stft = layers.TacotronSTFT(
            self.filter_length, self.hop_length, self.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        audiopaths_and_text_ori = self.audiopaths_and_text[:]
        if self.prep_trainset_per_epoch:
            seed = hparams.seed + epoch
        else:
            seed = hparams.seed
        if self.shuffle_audiopaths:
            self.audiopaths_and_text = permute_filelist(self.audiopaths_and_text,
                self.filelist_cols, seed, self.permute_opt, self.local_rand_factor)[0]
        if self.pre_batching:
            batch_sizes = get_batch_sizes(self.audiopaths_and_text,
                                 hparams.filelist_cols, hparams.batch_size)
            assert sum(batch_sizes) == len(self.audiopaths_and_text),\
                "check: not all samples get batched in pre-batching!"
            self.audiopaths_and_text = batching(self.audiopaths_and_text, batch_sizes)
        if self.shuffle_batches:
            if self.pre_batching:
                self.audiopaths_and_text = permute_batch_from_batch(
                    self.audiopaths_and_text, seed)
            else:
                self.audiopaths_and_text = permute_batch_from_filelist(
                    self.audiopaths_and_text, hparams.batch_size, seed)

        self.speaker_ids = speaker_ids
        if not self.speaker_ids:
            self.speaker_ids = self.create_lookup(audiopaths_and_text_ori, 'speaker')

        self.emotion_ids = emotion_ids
        if not self.emotion_ids:
            self.emotion_ids = self.create_lookup(audiopaths_and_text_ori, 'emotion')

    def parse_filelist_line(self, audiopath_and_text):
        # parse basic cols
        audiopath = audiopath_and_text[self.filelist_cols.index('audiopath')]
        text = audiopath_and_text[self.filelist_cols.index('text')]
        # parse optional cols
        emoembpath, dur, speaker, emotion = '', '', '', ''
        if 'emoembpath' in self.filelist_cols:
            emoembpath = audiopath_and_text[self.filelist_cols.index('emoembpath')]
        if 'dur' in self.filelist_cols:
            dur = float(audiopath_and_text[self.filelist_cols.index('dur')])
        if 'speaker' in self.filelist_cols:
            speaker = audiopath_and_text[self.filelist_cols.index('speaker')]
        if 'emotion' in self.filelist_cols:
            emotion = audiopath_and_text[self.filelist_cols.index('emotion')]
        return audiopath, emoembpath, text, dur, speaker, emotion

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        emoemb, speaker, emotion = '', '', ''
        audiopath, emoembpath, text, dur, speaker, emotion = \
            self.parse_filelist_line(audiopath_and_text)
        text = self.get_text(text)  # int_tensor[char_index, ....]
        mel = self.get_mel(audiopath)  # []
        if self.use_vae:
            if self.include_emo_emb:
                emoemb = self.get_emoemb(emoembpath)
            speaker = self.get_speaker(speaker, self.label_type)
            emotion = self.get_emotion(emotion, self.label_type)
        audioid = os.path.splitext(os.path.basename(audiopath))[0]
        return (text, mel, emoemb, speaker, emotion, dur, audioid)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm) # 1 X n_mel_channels X n_frames
            melspec = torch.squeeze(melspec, 0) # n_mel_channels X n_frames
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_emoemb(self, filename):
        emoemb = torch.from_numpy(np.load(filename)).T
        assert emoemb.size(0) == self.emo_emb_dim, (
            'Emotion embedding dimension mismatch: given {}, expected {}'.format(
                emoemb.size(0), self.emo_emb_dim))
        return emoemb

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def create_lookup(self, audiopaths_and_text, attribute):
        a2i = {'speaker':-2, 'emotion':-1}
        ids = sorted(set(x[a2i[attribute]] for x in audiopaths_and_text))
        d = {ids[i]: i for i in range(len(ids))}
        return d

    def get_speaker(self, speaker, label_type='one-hot'):
        if label_type == 'one-hot':
            speaker_vector = np.zeros(self.n_speakers)
            speaker_vector[self.speaker_ids[speaker]] = 1
            output = torch.Tensor(speaker_vector.astype(dtype=np.float32))
        elif label_type == 'id':
            output = torch.tensor([self.speaker_ids[speaker]])
        return output

    def get_emotion(self, emotion, label_type='one-hot'):
        if label_type == 'one-hot':
            emotion_vector = np.zeros(self.n_emotions)
            emotion_vector[self.emotion_ids[emotion]] = 1
            output = torch.Tensor(emotion_vector.astype(dtype=np.float32))
        elif label_type == 'id':
            output = torch.tensor([self.emotion_ids[emotion]])
        return output

    def __getitem__(self, index):
        if self.pre_batching:
            audiopaths_and_text = self.audiopaths_and_text[index]
            pairs = [self.get_mel_text_pair(audiopath_and_text) for
                     audiopath_and_text in audiopaths_and_text]
        else:
            pairs = self.get_mel_text_pair(self.audiopaths_and_text[index])
        return pairs

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, hparams, pre_batching=False):
        self.pre_batching = pre_batching
        self.n_frames_per_step = hparams.n_frames_per_step
        self.label_type = hparams.label_type
        self.use_vae = hparams.use_vae

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        e.g.
            import itertools
            batch = list(itertools.islice(train_loader.dataset, hparams.batch_size))
        """

        if self.pre_batching:
            batch = batch[0]

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        if self.use_vae:
            if self.label_type == 'one-hot':
                speakers = torch.LongTensor(len(batch), len(batch[0][3]))
                for i in range(len(ids_sorted_decreasing)):
                    speaker = batch[ids_sorted_decreasing[i]][3]
                    speakers[i, :] = speaker
                emotions = torch.LongTensor(len(batch), len(batch[0][4]))
                for i in range(len(ids_sorted_decreasing)):
                    emotion = batch[ids_sorted_decreasing[i]][4]
                    emotions[i, :] = emotion
            elif self.label_type == 'id':
                speakers = torch.LongTensor(len(batch))
                emotions = torch.LongTensor(len(batch))
                for i in range(len(ids_sorted_decreasing)):
                    speakers[i] = batch[ids_sorted_decreasing[i]][3]
                    emotions[i] = batch[ids_sorted_decreasing[i]][4]
        else:
            speakers = emotions = ''

        durs = [[] for _ in range(len(batch))]
        audioids = [[] for _ in range(len(batch))]
        for i in range(len(ids_sorted_decreasing)):
            durs[i] = batch[ids_sorted_decreasing[i]][5]
            audioids[i] = batch[ids_sorted_decreasing[i]][6]

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len1 = max([x[1].size(1) for x in batch])

        if len(batch[0][2]) > 0:
          num_emoembs = batch[0][2].size(0)
          max_target_len2 = max([x[2].size(1) for x in batch])

        max_target_len = max_target_len1
        # todo: uniform wintime/hoptime of mel and emoemb so max_target_len will be the same

        # increment max_target_len to the multiples of n_frames_per_step
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
            # todo: to support n_frames_per_step > 1

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        if len(batch[0][2]) > 0:
            emoemb_padded = torch.FloatTensor(len(batch), num_emoembs, max_target_len)
            emoemb_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                emoemb = batch[ids_sorted_decreasing[i]][2]
                emoemb_nframes = min(emoemb.size(1), max_target_len)
                emoemb_padded[i, :, :emoemb_nframes] = emoemb[:, :emoemb_nframes]
        else:
            emoemb_padded = ''

        return text_padded, input_lengths, mel_padded, emoemb_padded, \
            gate_padded, output_lengths, speakers, emotions, durs, audioids

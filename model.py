from math import sqrt, ceil
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, ConvNorm2D, LinearNorm
from utils import to_gpu, get_mask_from_lengths
#from fp16_optimizer import fp32_to_fp16, fp16_to_fp32
from modules import VAE_GST

drop_rate = 0.5
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=drop_rate, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), drop_rate, self.training)
        x = F.dropout(self.convolutions[-1](x), drop_rate, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.use_vae = hparams.use_vae
        if self.use_vae:
            self.encoder_embedding_dim = self.get_encoder_augmented_dim(hparams)
        else:
            self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_encoder_augmented_dim(self, hparams):
        encoder_embedding_dim = hparams.encoder_embedding_dim
        if hparams.embedding_variation > 1:
            encoder_embedding_dim += hparams.speaker_embedding_dim + \
                                     hparams.emotion_embedding_dim
        if hparams.embedding_variation > 2:
            encoder_embedding_dim += hparams.E
        return encoder_embedding_dim

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)

        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) == 1:
            gate_outputs.unsqueeze_(1)
        gate_outputs = gate_outputs.transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        #self.attention_cell = F.dropout(
        #   self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        #self.decoder_cell = F.dropout(
        #    self.decoder_cell, self.p_decoder_dropout, self.training)
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.use_vae = hparams.use_vae
        self.embedding_variation = hparams.embedding_variation
        self.label_type = hparams.label_type
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.symbols_embedding_dim = hparams.symbols_embedding_dim
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.emotion_embedding_dim = hparams.emotion_embedding_dim
        self.transcript_embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        if self.use_vae:
            if self.label_type == 'one-hot':
                self.speaker_embedding = LinearNorm(
                    hparams.n_speakers, hparams.speaker_embedding_dim, bias=True,
                    w_init_gain='tanh')
                self.emotion_embedding = LinearNorm(
                    hparams.n_emotions, hparams.emotion_embedding_dim, bias=True,
                    w_init_gain='tanh')
            elif self.label_type == 'id':
                self.speaker_embedding = nn.Embedding(
                    hparams.n_speakers, hparams.speaker_embedding_dim)
                self.emotion_embedding = nn.Embedding(
                    hparams.n_emotions, hparams.emotion_embedding_dim)
        self.vae_input_type = hparams.vae_input_type
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.transcript_embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.vae_gst = VAE_GST(hparams)


    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, emoemb_padded, gate_padded, \
            output_lengths, speakers, emotions, durs, audioids = batch
        text_padded = to_gpu(text_padded).long()
        if self.use_vae:
            if self.label_type == 'one-hot':
                speakers = to_gpu(speakers).float()
                emotions = to_gpu(emotions).float()
            elif self.label_type == 'id':
                speakers = to_gpu(speakers).long()
                emotions = to_gpu(emotions).long()
            if self.vae_input_type == 'emo':
              emoemb_padded = to_gpu(emoemb_padded).float()
        else:
            speakers, emotions = '', ''
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        x = (text_padded, input_lengths, mel_padded, emoemb_padded, max_len,
             output_lengths, speakers, emotions, durs, audioids)
        y = (mel_padded, gate_padded)

        return (x,y)

    # def parse_input(self, inputs):
    #     inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
    #     return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        #outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def combine_encoder_output(self, transcript_outputs, prosody_outputs,
            speaker_embeddings, emotion_embeddings, embedding_variation):
        if embedding_variation == 0: # original
            encoder_outputs = transcript_outputs + prosody_outputs
        else:
            speaker_embeddings = speaker_embeddings.unsqueeze_(1)
            emotion_embeddings = emotion_embeddings.unsqueeze_(1)
            if embedding_variation == 1: # add all
                num_repeat_speaker = self.symbols_embedding_dim // self.speaker_embedding_dim
                speaker_embedded_inputs = speaker_embeddings.repeat(1,
                    transcript_outputs.size(1), num_repeat_speaker)
                num_repeat_emotion = self.symbols_embedding_dim // self.emotion_embedding_dim
                emotion_embedded_inputs = emotion_embeddings.repeat(1,
                    transcript_outputs.size(1), num_repeat_emotion)
                encoder_outputs = transcript_outputs + prosody_outputs + \
                                  speaker_embedded_inputs + emotion_embedded_inputs
            elif embedding_variation == 2: # add (transcript, prosody) then concat (speaker, emotion)
                encoder_outputs = transcript_outputs + prosody_outputs
                encoder_outputs = self.concat_embedding(encoder_outputs, speaker_embeddings)
                encoder_outputs = self.concat_embedding(encoder_outputs, emotion_embeddings)
            elif embedding_variation == 3: # concat all
                encoder_outputs = torch.cat([transcript_outputs, prosody_outputs], dim=-1)
                encoder_outputs = self.concat_embedding(encoder_outputs, speaker_embeddings)
                encoder_outputs = self.concat_embedding(encoder_outputs, emotion_embeddings)
        return encoder_outputs

    def concat_embedding(self, outputs, embeddings):
        embeddings_ = embeddings.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embeddings_], dim=-1)
        return outputs

    def forward(self, inputs):
        # parsed arguments
        #  - inputs: padded text inputs (batch_size, max(input_lengths))
        #  - input_lengths: text inputs lengths (batch_size)
        #  - targets: mel outputs (batch_size, n_mel_channels, max(output_lengths))
        #  - output_lengths: mel outputs lengths (batch_size)

        inputs, input_lengths, targets, emoembs, _, output_lengths, speakers, \
            emotions, _, _ = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        # get embedded text inputs (batch_size, symbols_embedding_dim, max(input_lengths))
        transcript_embedded_inputs = self.transcript_embedding(inputs).transpose(1, 2)

        # [N, transcript_T, int(encoder_dim/2)] -- JH
        # get transcript output (batch_size, max(input_lengths), encoder_embedding_dim)
        transcript_outputs = self.encoder(transcript_embedded_inputs, input_lengths)

        # get embedded prosody outputs (batch_size, E)
        # get mu, logvar, z, all in the size of (batch_size, z_latent_dim)
        # get z
        if self.use_vae:
            if self.vae_input_type == 'mel':
                prosody_outputs, mu, logvar, z = self.vae_gst(targets)
            elif self.vae_input_type == 'emo':
                prosody_outputs, mu, logvar, z = self.vae_gst(emoembs)
            prosody_outputs = prosody_outputs.unsqueeze(1).expand_as(transcript_outputs)
            # get speaker and emotion embeddings
            speaker_embeddings = self.speaker_embedding(speakers)
            emotion_embeddings = self.emotion_embedding(emotions)
            # combine encoder outputs from transcript (-1,1), prosody outputs, etc.
            encoder_outputs = self.combine_encoder_output(transcript_outputs,
                prosody_outputs, speaker_embeddings, emotion_embeddings,
                self.embedding_variation)
        else:
            encoder_outputs = transcript_outputs

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if self.use_vae:
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mu, \
                logvar, z, emotions], output_lengths)
        else:
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
                output_lengths)

    def inference(self, inputs):

        inputs, input_lengths, targets, emoembs, _, output_lengths, speakers, \
            emotions, _ = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        transcript_embedded_inputs = self.transcript_embedding(inputs).transpose(1, 2)
        transcript_outputs = self.encoder.inference(transcript_embedded_inputs)

        if self.use_vae:
            if self.vae_input_type == 'mel':
                prosody_outputs, mu, logvar, z = self.vae_gst(targets)
            elif self.vae_input_type == 'emo':
                prosody_outputs, mu, logvar, z = self.vae_gst(emoembs)
            prosody_outputs = prosody_outputs.unsqueeze(1).expand_as(transcript_outputs)

            # get speaker and emotion embeddings
            speaker_embeddings = self.speaker_embedding(speakers)
            emotion_embeddings = self.emotion_embedding(emotions)

            encoder_outputs = self.combine_encoder_output(transcript_outputs,
                prosody_outputs, speaker_embeddings, emotion_embeddings,
                self.embedding_variation)
        else:
            encoder_outputs = transcript_outputs

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

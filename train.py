import os
import sys
import time
import argparse
import math
from numpy import finfo
import imageio
sys.path.append(os.getcwd())

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from plotting_utils import plot_scatter, plot_tsne, plot_kl_weight
from utils import get_kl_weight, get_text_padding_rate, get_mel_padding_rate
from utils import dict2col, dict2row, list2csv, csv2dict, flatten_list
from loss_function import Tacotron2Loss_VAE, Tacotron2Loss
from logger import Tacotron2Logger

from hparams import create_hparams, hparams_debug_string # for LJSpeech
#from hparams_soe import create_hparams, hparams_debug_string # for SOE


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams, epoch=0, valset=None, collate_fn=None):
    # Get data, data loaders and collate function ready

    # prepare train set
    print('preparing train set for epoch {}'.format(epoch))
    shuffle_train = {'shuffle-audiopath': hparams.shuffle_audiopaths,
        'shuffle-batch': hparams.shuffle_batches, 'permute-opt': hparams.permute_opt,
        'pre-batching': hparams.pre_batching}
    trainset = TextMelLoader(hparams.training_files, shuffle_train,
                             hparams, epoch)
    #print('\n'.join(['{}, {}'.format(line[0],line[2]) for line in \
    #                 trainset.audiopaths_and_text[:5]]))
    if valset is None:
        # prepare val set (different shuffle plan compared with train set)
        print('preparing val set for epoch {}'.format(epoch))
        shuffle_val = {'shuffle-audiopath': hparams.shuffle_audiopaths,
            'shuffle-batch': False, 'permute-opt': 'rand', 'pre-batching': False}
        valset = TextMelLoader(hparams.validation_files, shuffle_val, hparams)
    if collate_fn is None:
        collate_fn = {'train': TextMelCollate(hparams, pre_batching=hparams.pre_batching),
                      'val': TextMelCollate(hparams, pre_batching=False)}

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset, shuffle=hparams.shuffle_samples)
    else:
        train_sampler = None

    shuffle = (train_sampler is None) and hparams.shuffle_samples
    batch_size = 1 if hparams.pre_batching else hparams.batch_size
    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
        sampler=train_sampler, batch_size=batch_size, pin_memory=False,
        drop_last=True, collate_fn=collate_fn['train'])
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank,
                                   use_vae=False):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory),
                                 use_vae)
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict.get('iteration', 0)
    epoch = checkpoint_dict.get('epoch', 0)
    step = checkpoint_dict.get('step', 0)
    if epoch == 0:
        print("Loaded checkpoint '{}' from iter {}" .format(
            checkpoint_path, iteration))
    else:
        print("Loaded checkpoint '{}' from iter {} (epoch {}, step {})" .format(
            checkpoint_path, iteration, epoch, step))
    return model, optimizer, learning_rate, iteration, epoch, step


def save_checkpoint(model, optimizer, learning_rate, iteration, epoch, step, filepath):
    print("Saving model and optimizer state at iter {} (epoch {}, step {}) to {}".format(
        iteration, epoch, step, filepath))
    torch.save({'iteration': iteration,
                'epoch': epoch,
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def track_seq(track, input_lengths, gate_padded, metadata, verbose=False):
    padding_rate_txt, max_len_txt, top_len_txt = get_text_padding_rate(input_lengths)
    padding_rate_mel, max_len_mel, top_len_mel = get_mel_padding_rate(gate_padded)
    batch_size, batch_length = gate_padded.shape
    batch_area = batch_size * batch_length
    mem_all = torch.cuda.memory_allocated() / (1024**2)
    mem_cached = torch.cuda.memory_cached() / (1024**2)
    mem_use = mem_all + mem_cached
    duration, iteration, epoch, step = metadata
    if verbose:
        print('{} ({}:{}) {:.1f}Sec, '.format(
            iteration, epoch, step, duration), end='')
        print('batch cap {} ({}X{}) '.format(
            batch_area, batch_size, batch_length), end='')
        print('mem {:.1f}MiB ({:.1f}+{:.1f}) '.format(
            mem_use, mem_all, mem_cached), end='')
        print('text (pad%: {0:.1f}%, top3: {1}), '.format(
            padding_rate_txt*100, top_len_txt), end='')
        print('mel (pad%: {0:.1f}%, top3: {1})'.format(
            padding_rate_mel*100, top_len_mel))
    track['padding-rate-txt'].append(padding_rate_txt)
    track['max-len-txt'].append(max_len_txt)
    track['top-len-txt'].append(top_len_txt)
    track['padding-rate-mel'].append(padding_rate_mel)
    track['max-len-mel'].append(max_len_mel)
    track['top-len-mel'].append(top_len_mel)
    track['batch-size'].append(batch_size)
    track['batch-length'].append(batch_length)
    track['batch-area'].append(batch_area)
    track['mem-use'].append(mem_use)
    track['mem-all'].append(mem_all)
    track['mem-cached'].append(mem_cached)
    track['duration'].append(duration)
    track['iteration'].append(iteration)
    track['epoch'].append(epoch)
    track['step'].append(step)


def validate(model, criterion, valset, iteration, batch_size, n_gpus, collate_fn,
             logger, distributed_run, rank, use_vae=False, pre_batching=False):
    """Handles all the validation scoring and printing"""
    model.eval()
    #torch.set_grad_enabled(False)
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        batch_size = 1 if pre_batching else batch_size
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        y0, y_pred0 = '', ''
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            # save first batch (with full batch size) for logging later
            if not y0 and not y_pred0:
              y0, y_pred0 = y, y_pred
            if use_vae:
                loss, _, _, _ = criterion(y_pred, y, iteration)
            else:
                loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            if rank == 0:
                this_batch_size, this_batch_length = batch[0].size(0), batch[2].size(2)
                this_batch_area = this_batch_size * this_batch_length
                mem_all = torch.cuda.memory_allocated()
                mem_cached = torch.cuda.memory_cached()
                mem_use = mem_all + mem_cached
                print('{}/{}: '.format(i, len(val_loader)), end='')
                print('Batch: {} ({}X{}) '.format(this_batch_area, this_batch_size,
                    this_batch_length), end='')
                print('Mem: {:.2f} ({:.2f}+{:.2f}) '.format(mem_use/(1024**2),
                    mem_all/(1024**2), mem_cached/(1024**2)), end='')
                print('Val loss {:.3f}'.format(reduced_val_loss))
        val_loss = val_loss / (i + 1)
    #torch.set_grad_enabled(True)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:.3f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y0, y_pred0, iteration)

    if use_vae:
        mus, emotions = y_pred0[4], y_pred0[7]
    else:
        mus, emotions = '', ''

    return val_loss, (mus, emotions)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    if hparams.use_vae:
        criterion = Tacotron2Loss_VAE(hparams)
    else:
        criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank, hparams.use_vae)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    valset_csv = os.path.join(output_directory, log_directory, 'valset.csv')
    # list2csv(flatten_list(valset.audiopaths_and_text), valset_csv, delimiter='|')
    list2csv(valset.audiopaths_and_text, valset_csv, delimiter='|')

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration, epoch, step = \
                load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            if epoch == 0:
                iteration += 1  # next iteration is iteration + 1
                epoch_offset = max(0, int(iteration / len(train_loader)))
            else:
                epoch_offset = epoch
            print('epoch offset: {}'.format(epoch_offset))
            train_loader = prepare_dataloaders(hparams, epoch_offset, valset,
                collate_fn['train'])[0]
        print('completing loading model ...')

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    track_csv = os.path.join(output_directory, log_directory, 'track.csv')
    track_header = ['padding-rate-txt', 'max-len-txt', 'top-len-txt',
        'padding-rate-mel', 'max-len-mel', 'top-len-mel', 'batch-size',
        'batch-length', 'batch-area', 'mem-use', 'mem-all', 'mem-cached',
        'duration', 'iteration', 'epoch', 'step']
    if os.path.isfile(track_csv) and checkpoint_path is not None:
        print('loading existing {} ...'.format(track_csv))
        track = csv2dict(track_csv, header=track_header)
    else:
        track = {k:[] for k in track_header}

    print('start training in epoch {} ~ {} ...'.format(epoch_offset, hparams.epochs))
    nbatches = len(train_loader)
    for epoch in range(epoch_offset, hparams.epochs):
        #if epoch >= 10: break
        print("Epoch: {}, #batches: {}".format(epoch, nbatches))
        batch_sizes, batch_lengths = [0] * nbatches, [0] * nbatches
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            if hparams.use_vae:
                loss, recon_loss, kl, kl_weight = criterion(y_pred, y, iteration)
            else:
                loss = criterion(y_pred, y)

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                batch_sizes[i], batch_lengths[i] = batch[0].size(0), batch[2].size(2)
                batch_capacity = batch_sizes[i] * batch_lengths[i]
                mem_all = torch.cuda.memory_allocated() / (1024**2)
                mem_cached = torch.cuda.memory_cached() / (1024**2)
                mem_use = mem_all + mem_cached
                print("{} ({}:{}/{}): ".format(iteration, epoch, i, nbatches), end='')
                print("Batch {} ({}X{}) ".format(batch_capacity, batch_sizes[i],
                    batch_lengths[i]), end='')
                print("Mem {:.1f} ({:.1f}+{:.1f}) ".format(mem_use, mem_all,
                    mem_cached), end='')
                print("Train loss {:.3f} Grad Norm {:.3f} {:.2f}s/it".format(
                    reduced_loss, grad_norm, duration))
                input_lengths, gate_padded = batch[1], batch[4]
                metadata = (duration, iteration, epoch, i)
                track_seq(track, input_lengths, gate_padded, metadata)
                padding_rate_txt = track['padding-rate-txt'][-1]
                max_len_txt = track['max-len-txt'][-1]
                padding_rate_mel = track['padding-rate-mel'][-1]
                max_len_mel = track['max-len-mel'][-1]
                if hparams.use_vae:
                    logger.log_training(
                        reduced_loss, grad_norm, learning_rate, duration,
                        padding_rate_txt, max_len_txt, padding_rate_mel,
                        max_len_mel, iteration, recon_loss, kl, kl_weight)
                else:
                    logger.log_training(
                        reduced_loss, grad_norm, learning_rate, duration,
                        padding_rate_txt, max_len_txt, padding_rate_mel,
                        max_len_mel, iteration)

            check_by_iter = (hparams.check_by == 'iter') and \
                            (iteration % hparams.iters_per_checkpoint == 0)
            check_by_epoch = (hparams.check_by == 'epoch') and i == 0 and \
                             (epoch % hparams.epochs_per_checkpoint == 0)
            if not is_overflow and (check_by_iter or check_by_epoch):
                dict2col(track, track_csv, verbose=True)
                val_loss, (mus, emotions) = validate(model, criterion, valset,
                     iteration, hparams.batch_size, n_gpus, collate_fn['val'], logger,
                     hparams.distributed_run, rank, hparams.use_vae, pre_batching=False)
                if rank == 0:
                    checkpoint_path = os.path.join(output_directory,
                        "checkpoint_{}-{}-{}_{:.3f}".format(iteration, epoch, i, val_loss))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                         epoch, i, checkpoint_path)
                    if hparams.use_vae:
                        image_scatter_path = os.path.join(output_directory,
                             "checkpoint_{0}_scatter_val.png".format(iteration))
                        image_tsne_path = os.path.join(output_directory,
                             "checkpoint_{0}_tsne_val.png".format(iteration))
                        imageio.imwrite(image_scatter_path, plot_scatter(mus, emotions))
                        imageio.imwrite(image_tsne_path, plot_tsne(mus, emotions))

            iteration += 1

        if hparams.prep_trainset_per_epoch:
            train_loader = prepare_dataloaders(hparams, epoch+1, valset,
                collate_fn['train'])[0]
            nbatches = len(train_loader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--gpu', type=int, default=0,
                        required=False, help='current gpu device id')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    return parser.parse_args()


def log_args(args, argnames, logfile):
    dct = {name: eval('args.{}'.format(name)) for name in argnames}
    dict2row(dct, logfile, order='ascend')


if __name__ == '__main__':

    # # runtime mode
    # args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()
    args.output_directory = 'outdir/ljspeech/ssb-dbs1' # check
    args.log_directory = 'logdir'
    args.checkpoint_path = None # fresh run
    args.warm_start = False
    args.n_gpus = 1
    args.rank = 0
    args.gpu = 1 # check
    args.group_name = 'group_name'
    hparams = ["training_files=filelists/ljspeech/ljspeech_wav_train.txt",
               "validation_files=filelists/ljspeech/ljspeech_wav_valid.txt",
               "filelist_cols=[audiopath,text,dur,speaker,emotion]",
               "shuffle_audiopaths=True",
               "seed=0000",
               "shuffle_batches=True",
               "shuffle_samples=False",
               "permute_opt=semi-sort",
               "local_rand_factor=0.1",
               "pre_batching=True",
               "prep_trainset_per_epoch=True",
               "override_sample_size=False",
               "text_cleaners=[english_cleaners]",
               "use_vae=False",
               "anneal_function=logistic",
               "use_saved_learning_rate=True",
               "load_mel_from_disk=False",
               "include_emo_emb=False",
               "vae_input_type=mel",
               "fp16_run=False",
               "embedding_variation=0",
               "label_type=one-hot",
               "distributed_run=False",
               "batch_size=16",
               "iters_per_checkpoint=2000",
               "anneal_x0=100000",
               "anneal_k=0.0001"]
    args.hparams = ','.join(hparams)

    # create log directory due to saving files before training starts
    if not os.path.isdir(args.output_directory):
        print('creating dir: {} ...'.format(args.output_directory))
        os.makedirs(args.output_directory)
        os.chmod(args.output_directory, 0o775)

    argnames = ['output_directory', 'log_directory', 'checkpoint_path',
                'warm_start', 'n_gpus', 'rank', 'gpu', 'group_name']
    args_csv = os.path.join(args.output_directory, 'args.csv')
    log_args(args, argnames, args_csv)

    hparams = create_hparams(args.hparams)
    hparams_csv = os.path.join(args.output_directory, 'hparams.csv')
    print(hparams_debug_string(hparams, hparams_csv))

    if args.n_gpus == 1:
        # set current GPU device
        torch.cuda.set_device(args.gpu)
    print('current GPU: {}'.format(torch.cuda.current_device()))

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("shuffle audiopaths:", hparams.shuffle_audiopaths)
    print("permute option:", hparams.permute_opt)
    print("local_rand_factor:", hparams.local_rand_factor)
    print("pre_batching:", hparams.pre_batching)
    print("prep trainset per epoch:", hparams.prep_trainset_per_epoch)
    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("Override Sample Size:", hparams.override_sample_size)
    print("Load Mel from Disk:", hparams.load_mel_from_disk)
    print("Use VAE:", hparams.use_vae)
    print("Include Emotion Embedding:", hparams.include_emo_emb)
    print("Label Type:", hparams.label_type)
    print("VAE Input Type:", hparams.vae_input_type)
    print("Embedding Variation:", hparams.embedding_variation)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    # log kl weights
    if hparams.use_vae:
        af = hparams.anneal_function
        lag = hparams.anneal_lag
        k = hparams.anneal_k
        x0 = hparams.anneal_x0
        upper = hparams.anneal_upper
        constant = hparams.anneal_constant
        kl_weights = get_kl_weight(af, lag, k, x0, upper, constant, nsteps=250000)
        imageio.imwrite(os.path.join(args.output_directory, 'kl_weights.png'),
                        plot_kl_weight(kl_weights, af, lag,k, x0, upper, constant))

    output_directory = args.output_directory
    log_directory = args.log_directory
    checkpoint_path = args.checkpoint_path
    warm_start = args.warm_start
    n_gpus = args.n_gpus
    rank = args.rank
    group_name = args.group_name

    train(output_directory, log_directory, checkpoint_path,
          warm_start, n_gpus, rank, group_name, hparams)

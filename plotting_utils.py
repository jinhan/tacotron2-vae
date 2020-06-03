import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from sklearn.manifold import TSNE

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_scatter(mus, y, figsize=(8,8)):
    """
    Scatter plot can be drawn on tensorboardX
    """
    colors = 'r', 'b', 'g', 'y'
    labels = 'neutral', 'happy', 'angry', 'sad'

    mus = mus.cpu().numpy()
    y = y.cpu().numpy()
    y = np.argmax(y, 1)

    # sort mus by its variance in descending order and get the first 2 indices
    idx = sorted(np.argsort(np.std(mus, 0))[::-1][:2])

    fig, ax = plt.subplots(figsize=figsize)
    for i, (c, label) in enumerate(zip(colors, labels)):
        ax.scatter(mus[y==i, idx[0]], mus[y==i, idx[1]], c=c, label=label, alpha=0.5)
    plt.xlabel('dim {}'.format(idx[0])), plt.ylabel('dim {}'.format(idx[1]))
    plt.title('scatter plot of mus with emotion labels, ' +
              'dim: {}, show {}, {}'.format(mus.shape[1], idx[0], idx[1]))
    plt.grid(True), plt.legend(loc='upper left')

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_tsne(mus, y, figsize=(8,8)):
  """
  t-SNE scatter plot can be drawn on tensorboardX
  """
  colors = 'r', 'b', 'g', 'y'
  labels = 'neutral', 'happy', 'angry', 'sad'

  mus = mus.cpu().numpy()
  y = y.cpu().numpy()
  y = np.argmax(y, 1)

  tsne_model = TSNE(n_components=2, random_state=0, init='random')
  mus_transformed = tsne_model.fit_transform(mus)

  fig, ax = plt.subplots(figsize=figsize)
  for i, (c, label) in enumerate(zip(colors, labels)):
    ax.scatter(mus_transformed[y==i, 0], mus_transformed[y==i, 1],
               c=c, label=label, alpha=0.5)
  plt.xlabel('dim 0'), plt.ylabel('dim 1')
  plt.title('t-SNE plot of mus with emotion labels, dim: {}'.format(mus.shape[1]))
  plt.grid(True), plt.legend(loc='upper left')

  fig.canvas.draw()
  data = save_figure_to_numpy(fig)
  plt.close()
  return data
